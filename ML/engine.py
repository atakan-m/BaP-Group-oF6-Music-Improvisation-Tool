"""Realtime inference engine for the jazz improvisation tool.

This module hosts the runtime that drives the trained LSTM during play:

    MicTickAggregator   maps audio-block pitch detections to one
                        REST / HOLD / NOTE_x token per 16th-note window.
    make_sampler        builds a sampling closure with HOLD/REST penalties
                        and a hard cap on consecutive HOLDs.
    JazzImprov          the stateful per-16th-tick engine. Maintains the
                        LSTM's persistent state, a rollout buffer of
                        upcoming predictions, and a post-deviation lock
                        counter. Its commit() method is the per-tick API.
    make_engine         one-call factory: loads the checkpoint, builds the
                        engine, and primes its initial rollout.

Piano_display.py uses make_engine via its compatibility alias
`make_realtime_engine` re-exported from generate.py.
"""

import json
import os

import numpy as np
import torch

from chord_utils import build_chord_tables, chord_str_to_ids
from model import (
    load_model,
    TOK_REST, TOK_HOLD, TOK_NOTE_BASE,
    PITCH_LOW, PITCH_HIGH,
    TICKS_PER_BAR,
)


# =====================================================================
# Token <-> pitch helpers
# =====================================================================

def pitch_to_note_token(midi_pitch):
    """Map a MIDI pitch to its NOTE_x token id, clipped to the vocab range."""
    p = max(PITCH_LOW, min(PITCH_HIGH, int(midi_pitch)))
    return TOK_NOTE_BASE + (p - PITCH_LOW)


def token_to_pitch(tok):
    """Map a NOTE_x token to its MIDI pitch; returns None for REST/HOLD."""
    tok = int(tok)
    return PITCH_LOW + (tok - TOK_NOTE_BASE) if tok >= TOK_NOTE_BASE else None


_NOTE_NAME_TO_PC = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
                    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def _note_name_to_midi(name):
    """Parse a note-name string like 'C4' or 'F#3' into a MIDI integer.

    Returns None for empty input or unparseable strings. Used by the mic
    aggregator when an SP detector exposes its detection as a note name
    instead of a numeric MIDI value.
    """
    if not name:
        return None
    i = 1
    if i < len(name) and name[i] == "#":
        i += 1
    try:
        return _NOTE_NAME_TO_PC[name[:i]] + (int(name[i:]) + 1) * 12
    except (KeyError, ValueError):
        return None


# =====================================================================
# Microphone aggregator
# =====================================================================

class MicTickAggregator:
    """Aggregate per-audio-block pitch detections into per-tick tokens.

    Sits between the SP detector (which fires every ~11.6 ms) and the
    engine (which commits every 16th note). Inside each 16th-note window,
    every audio callback observes one MIDI value (or None for silence) via
    `observe(...)`. At the tick boundary `consume_token()` resolves the
    window to a single REST / HOLD / NOTE_x token.

    Typical wiring (set up before the audio stream starts):

        agg = MicTickAggregator()
        original = detector.callback
        def wrapped(indata, frames, time_info, status):
            original(indata, frames, time_info, status)
            agg.observe(detector.last_midi)
        detector.callback = wrapped

        # then once per 16th note:
        tok = agg.consume_token()
        engine.commit(tok)
    """

    DEFAULT_MIN_MIDI = 48     # C3 — lowest pitch accepted from the mic
    DEFAULT_MAX_MIDI = 83     # B5 — highest pitch accepted from the mic

    def __init__(self, min_midi=None, max_midi=None):
        self._midis = []                                   # observations this window
        self._prev_pitch = None                            # dominant pitch of the last tick
        self.min_midi = self.DEFAULT_MIN_MIDI if min_midi is None else int(min_midi)
        self.max_midi = self.DEFAULT_MAX_MIDI if max_midi is None else int(max_midi)

    def _gate(self, midi):
        """Return midi if it's an integer inside [min_midi, max_midi], else None.

        Used to drop residual HPS octave errors and out-of-range noise that
        survive the SP-side filtering.
        """
        if midi is None:
            return None
        try:
            m = int(midi)
        except (TypeError, ValueError):
            return None
        return m if self.min_midi <= m <= self.max_midi else None

    def observe(self, midi_or_none):
        """Record one observation. Called once per audio block.

        Accepts None (silence), an int / float MIDI value, or a note-name
        string. Anything else is treated as silence.
        """
        if midi_or_none is None:
            self._midis.append(None)
        elif isinstance(midi_or_none, (int, float)):
            self._midis.append(self._gate(midi_or_none))
        elif isinstance(midi_or_none, str):
            self._midis.append(self._gate(_note_name_to_midi(midi_or_none)))
        else:
            self._midis.append(None)

    def consume_token(self):
        """Resolve the just-finished window and return a single token.

        Behaviour:
            * fewer than half the observations have a valid pitch → REST;
            * dominant pitch differs from last tick's dominant pitch → emit
              NOTE_<dominant> (a new attack);
            * dominant pitch equals last tick's → emit HOLD (sustain).
        Empties the observation buffer for the next window.
        """
        observations = self._midis
        self._midis = []
        valid = [m for m in observations if m is not None]
        # If less than half the window had a valid pitch, call it silence.
        if not valid or len(valid) * 2 < len(observations):
            self._prev_pitch = None
            return TOK_REST
        counts = {}
        for m in valid:
            counts[m] = counts.get(m, 0) + 1
        dominant = max(counts.items(), key=lambda kv: kv[1])[0]
        is_new_attack = dominant != self._prev_pitch
        self._prev_pitch = dominant
        return pitch_to_note_token(dominant) if is_new_attack else TOK_HOLD


# =====================================================================
# Sampler
# =====================================================================

def make_sampler(temperature=1.0, hold_penalty=0.0, rest_penalty=0.0,
                 max_consec_holds=None, rng=None):
    """Build a token-sampling closure with the configured penalties.

    The returned closure has the signature:

        sample(logits, consec_holds=0) -> token_id

    Logits are modified in this order before softmaxing:
        1. HOLD logit -= hold_penalty                (soft bias)
        2. REST logit -= rest_penalty                (soft bias)
        3. HOLD logit -> -inf  if consec_holds >= max_consec_holds
           (hard cap that prevents arbitrarily long sustains)
        4. divide by temperature, softmax, multinomial sample.
    """
    if rng is None:
        rng = np.random.default_rng()
    inv_temp = 1.0 / max(float(temperature), 1e-6)

    def sample(logits, consec_holds=0):
        l = logits.detach().cpu().numpy().astype(np.float64)
        if hold_penalty != 0.0:
            l[TOK_HOLD] -= hold_penalty
        if rest_penalty != 0.0:
            l[TOK_REST] -= rest_penalty
        if max_consec_holds is not None and consec_holds >= max_consec_holds:
            l[TOK_HOLD] = -1e9
        l *= inv_temp
        l -= l.max()
        p = np.exp(l)
        p /= p.sum()
        return int(rng.choice(len(p), p=p))

    return sample


# =====================================================================
# Realtime engine
# =====================================================================

DEFAULT_CURRENT_PITCH = 60     # what the engine assumes the player is on at reset


class JazzImprov:
    """Stateful realtime engine wrapped around a trained JazzLSTM.

    The engine maintains three conceptually distinct pieces of state:

      Persistent state
          h_persistent, prev_actual, current_tick, _current_pitch, _prev_pitch.
          Represents what the player has actually committed so far. Advances
          by exactly one tick per commit().

      Rollout buffer
          _rollout_buf, _h_rollout, _prev_rollout_tok, _rollout_next_tick.
          The model's prediction for the upcoming ~rollout_ticks ticks. On
          a match or silence-as-match the buffer shifts forward by one and
          extends at the back; on a deviation it is rebuilt with the lock
          HOLDs followed by fresh predictions.

      Lock counter
          _locked_ticks. Counts down the forced-HOLD ticks that follow a
          deviation — during this period player input is treated as a
          sustained HOLD unless they attack a new pitch (which triggers a
          fresh deviation).

    Public API (used from Piano_display): set_progression, reset, commit,
    rollout, current_pitch, is_done, last_action.
    """

    def __init__(self, model, *, temperature=1.0,
                 rollout_ticks=30, deviation_lock_ticks=1,
                 hold_penalty=1.0, rest_penalty=1.5,
                 max_consec_holds=None,
                 deviation_initial_fresh=2,
                 rollout_extend_per_tick=4,
                 seed=0, device="cpu"):
        """Configure the engine. Call `set_progression` and `reset` before
        the first `commit`.

        Args:
            model                   : a JazzLSTM (typically int8-quantised
                                      on the Pi).
            temperature             : sampling temperature.
            rollout_ticks           : target rollout length in ticks.
            deviation_lock_ticks    : total quarter-/half-note lock length
                                      (n_lock = this − 1 forced HOLDs).
            hold_penalty            : soft logit penalty on HOLD.
            rest_penalty            : soft logit penalty on REST.
            max_consec_holds        : hard cap on consecutive HOLDs.
            deviation_initial_fresh : tokens sampled fresh on the deviation
                                      tick itself. The rest of the rollout
                                      is refilled by catchup over the
                                      following ticks.
            rollout_extend_per_tick : maximum extra tokens sampled per
                                      catchup commit while the rollout is
                                      still shorter than rollout_ticks.
            seed                    : RNG seed for reproducibility.
            device                  : torch device.
        """
        self.model = model
        self.device = torch.device(device)
        self.deviation_lock_ticks = int(deviation_lock_ticks)
        self.rollout_ticks = int(rollout_ticks)
        # Deviation cost amortisation. On a deviation we sample only this
        # many fresh tokens instead of the full rollout_ticks − n_lock, and
        # the rest of the rollout refills by `rollout_extend_per_tick`
        # tokens per subsequent commit. Trades partial visual lookahead for
        # a much smaller per-tick LSTM spike on the Raspberry Pi.
        self.deviation_initial_fresh = int(deviation_initial_fresh)
        self.rollout_extend_per_tick = int(rollout_extend_per_tick)

        self._rng = np.random.default_rng(seed)
        self._sample = make_sampler(
            temperature=temperature, hold_penalty=hold_penalty,
            rest_penalty=rest_penalty, max_consec_holds=max_consec_holds,
            rng=self._rng,
        )

        # Progression-derived per-tick features (populated by set_progression).
        self.chord_root = None      # numpy int64 array, one entry per tick
        self.chord_qual = None      # numpy int64 array, one entry per tick
        self.bar_chords = None      # list of chord strings, one per bar
        self.total_ticks = 0

        # Runtime state (populated by reset).
        self.current_tick = 0
        self.prev_actual = TOK_REST
        self.h_persistent = None
        self._rollout_buf = []
        self._h_rollout = None
        self._prev_rollout_tok = TOK_REST
        self._rollout_next_tick = 0
        self._locked_ticks = 0
        self._current_pitch = DEFAULT_CURRENT_PITCH
        # Pitch the player was on *before* the most recent pitch change.
        # Lets the deviation check tolerate a late attack (player playing
        # the previous planned note one tick after the engine has moved on).
        self._prev_pitch = DEFAULT_CURRENT_PITCH
        # One of: "init", "match", "silence", "deviation", "locked", "done".
        # Updated at the end of every commit() so callers can bucket timing
        # measurements or log per-tick behaviour.
        self.last_action = "init"

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def current_pitch(self):
        """The MIDI pitch the engine currently believes the player is on."""
        return self._current_pitch

    @property
    def is_done(self):
        """True once the engine has committed all `total_ticks` ticks."""
        return self.current_tick >= self.total_ticks

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def set_progression(self, chord_progression, total_bars, qual_to_id):
        """Lay out the per-tick chord_root and chord_qual feature arrays.

        The chord progression is cycled to fill `total_bars` bars, then
        expanded so every tick within a bar carries the bar's chord ids.

        Args:
            chord_progression : list of (chord_str, n_bars) tuples,
                                cycled if shorter than total_bars.
            total_bars        : total length of the song in bars.
            qual_to_id        : quality-string → id lookup obtained from
                                build_chord_tables (chord_utils.py).
        """
        bar_chords = []
        for chord_str, nb in chord_progression:
            bar_chords.extend([chord_str] * int(nb))
        cycle = list(bar_chords) or ["NC"]
        while len(bar_chords) < total_bars:
            bar_chords += cycle
        bar_chords = bar_chords[:total_bars]

        n_total = total_bars * TICKS_PER_BAR
        cr = np.zeros(n_total, dtype=np.int64)
        cq = np.zeros(n_total, dtype=np.int64)
        for bi, c in enumerate(bar_chords):
            r, q = chord_str_to_ids(c, qual_to_id)
            t0 = bi * TICKS_PER_BAR
            cr[t0:t0 + TICKS_PER_BAR] = r
            cq[t0:t0 + TICKS_PER_BAR] = q

        self.chord_root = cr
        self.chord_qual = cq
        self.bar_chords = bar_chords
        self.total_ticks = n_total

    def reset(self):
        """Reset all runtime state and rebuild the initial rollout.

        Must be called after set_progression and before the first commit.
        """
        if self.chord_root is None:
            raise RuntimeError("Call set_progression() before reset()")
        self.current_tick = 0
        self.prev_actual = TOK_REST
        self.h_persistent = None
        self._rollout_buf = []
        self._h_rollout = None
        self._prev_rollout_tok = TOK_REST
        self._rollout_next_tick = 0
        self._locked_ticks = 0
        self._current_pitch = DEFAULT_CURRENT_PITCH
        self._prev_pitch = DEFAULT_CURRENT_PITCH
        self.last_action = "init"
        self._refresh_rollout()

    # ------------------------------------------------------------------
    # LSTM primitives
    # ------------------------------------------------------------------
    def _forward_one(self, prev_token, t, hidden):
        """Run one LSTM step predicting the token at tick `t`.

        Reads the chord features at `t` from the precomputed arrays and
        feeds them alongside `prev_token` and `t % TICKS_PER_BAR` into the
        model. Returns (logits[VOCAB_SIZE], new_hidden_state).
        """
        cr = int(self.chord_root[t])
        cq = int(self.chord_qual[t])
        bp = t % TICKS_PER_BAR
        d = self.device
        pt  = torch.tensor([[int(prev_token)]], dtype=torch.long, device=d)
        crt = torch.tensor([[cr]],              dtype=torch.long, device=d)
        cqt = torch.tensor([[cq]],              dtype=torch.long, device=d)
        bpt = torch.tensor([[bp]],              dtype=torch.long, device=d)
        logits, hidden = self.model(pt, crt, cqt, bpt, hidden)
        return logits[0, 0], hidden

    @staticmethod
    def _clone_hidden(h):
        """Deep-copy an LSTM hidden tuple (h, c) so the rollout state can be
        forked off without aliasing the persistent state."""
        if h is None:
            return None
        return (h[0].clone(), h[1].clone())

    @staticmethod
    def _count_trailing_holds(buf):
        """Number of TOK_HOLD tokens at the end of `buf`.

        Used to seed the consec_holds counter for the sampler so the hard
        max-consec-holds cap respects HOLDs that are already in the rollout.
        """
        n = 0
        for tok in reversed(buf):
            if int(tok) == TOK_HOLD:
                n += 1
            else:
                break
        return n

    # ------------------------------------------------------------------
    # Pitch tracking — HOLD and REST resolve to the engine's current pitch
    # so the deviation check can compare MIDI numbers directly.
    # ------------------------------------------------------------------
    def _token_to_pitch(self, tok):
        """Map a token to a MIDI pitch (NOTE_x → its pitch, HOLD/REST → current)."""
        if int(tok) >= TOK_NOTE_BASE:
            return PITCH_LOW + (int(tok) - TOK_NOTE_BASE)
        return self._current_pitch

    def _update_current_pitch(self, tok):
        """Advance pitch tracking on a NOTE attack.

        If `tok` is a NOTE token and its pitch differs from current_pitch,
        moves current_pitch into _prev_pitch and stores the new pitch as
        current. No-op for HOLD/REST or for same-pitch re-attacks.
        """
        if int(tok) >= TOK_NOTE_BASE:
            new_pitch = PITCH_LOW + (int(tok) - TOK_NOTE_BASE)
            if new_pitch != self._current_pitch:
                self._prev_pitch = self._current_pitch
                self._current_pitch = new_pitch

    def _next_expected_pitch(self):
        """The next planned pitch change in the rollout.

        Walks the rollout until it finds a pitch different from
        current_pitch and returns it, or None if the whole rollout sits on
        the current pitch. Used by the deviation tolerance buffer so the
        player can attack the next planned note one tick early without
        triggering a re-plan.
        """
        for tok in self._rollout_buf:
            p = self._token_to_pitch(tok)
            if p != self._current_pitch:
                return p
        return None

    # ------------------------------------------------------------------
    # Rollout machinery — every sampling path ultimately calls _sample_n.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sample_n(self, h, prev, n, start_tick, consec_holds=0):
        """Sample up to `n` new tokens autoregressively.

        Stops early if the song would run past `total_ticks`. Returns
        (sampled_tokens, final_hidden, final_prev_token, final_consec_holds)
        so callers can keep chaining additional samples without re-deriving
        the state.
        """
        tokens = []
        end_tick = min(start_tick + n, self.total_ticks)
        for t in range(start_tick, end_tick):
            logits, h = self._forward_one(prev, t, h)
            nxt = self._sample(logits, consec_holds)
            consec_holds = consec_holds + 1 if nxt == TOK_HOLD else 0
            tokens.append(nxt)
            prev = nxt
        return tokens, h, prev, consec_holds

    @torch.no_grad()
    def _refresh_rollout(self):
        """Build the rollout from scratch using a clone of the persistent state.

        Called once from reset() to prime the initial lookahead so the very
        first commit already has a rollout to consume.
        """
        h = self._clone_hidden(self.h_persistent)
        tokens, h, prev, _ = self._sample_n(
            h, self.prev_actual, self.rollout_ticks, self.current_tick,
        )
        self._rollout_buf = tokens
        self._h_rollout = h
        self._prev_rollout_tok = prev
        self._rollout_next_tick = self.current_tick + len(tokens)

    @torch.no_grad()
    def _extend_rollout_by_one(self):
        """Sample one additional token and append it at the back of the rollout.

        Used by the match / silence / locked commit paths to maintain the
        rollout length as the front is popped each tick.
        """
        if self._rollout_next_tick >= self.total_ticks:
            return
        consec = self._count_trailing_holds(self._rollout_buf)
        tokens, h, prev, _ = self._sample_n(
            self._h_rollout, self._prev_rollout_tok, 1,
            self._rollout_next_tick, consec,
        )
        if tokens:
            self._rollout_buf.append(tokens[0])
            self._h_rollout = h
            self._prev_rollout_tok = prev
            self._rollout_next_tick += 1

    @torch.no_grad()
    def _extend_rollout_by_n(self, n):
        """Sample `n` additional tokens in one batched call and append them.

        Faster than calling _extend_rollout_by_one in a loop because all
        forwards share a single Python overhead per batch. Used by
        _catchup_extend during the refill phase that follows a deviation.
        """
        if n <= 0 or self._rollout_next_tick >= self.total_ticks:
            return
        n = min(n, self.total_ticks - self._rollout_next_tick)
        consec = self._count_trailing_holds(self._rollout_buf)
        tokens, h, prev, _ = self._sample_n(
            self._h_rollout, self._prev_rollout_tok, n,
            self._rollout_next_tick, consec,
        )
        if tokens:
            self._rollout_buf.extend(tokens)
            self._h_rollout = h
            self._prev_rollout_tok = prev
            self._rollout_next_tick += len(tokens)

    def _catchup_extend(self):
        """Top the rollout up toward target length by a bounded amount.

        Called from the match / silence / locked commit paths. Does nothing
        once the rollout is back at rollout_ticks length, so steady-state
        commits pay no extra cost. During the post-deviation refill window
        it extends by up to rollout_extend_per_tick tokens per commit,
        spreading the deviation's total work across multiple ticks instead
        of bunching it all into the deviation tick itself.
        """
        deficit = self.rollout_ticks - len(self._rollout_buf)
        if deficit <= 0:
            return
        self._extend_rollout_by_n(min(deficit, self.rollout_extend_per_tick))

    @torch.no_grad()
    def _build_deviation_rollout(self, n_lock, initial_fresh=None):
        """Rebuild the rollout for the start of a post-deviation lock period.

        The new rollout begins with `n_lock` forced HOLDs (which represent
        the player sustaining the deviation note) and is then extended by
        `initial_fresh` newly sampled tokens. The remaining capacity up to
        rollout_ticks is filled in over the following ticks by
        _catchup_extend so the deviation commit itself doesn't have to
        sample the whole rollout in one go.

        Args:
            n_lock        : forced-HOLD ticks to simulate up front. These
                            are walked through the LSTM (no sampling) so
                            the post-lock state reflects the sustain.
            initial_fresh : number of predicted tokens sampled on top of
                            the lock. Defaults to deviation_initial_fresh.
                            Pass `rollout_ticks - n_lock` to fall back to
                            the original "build the full rollout right
                            away" behaviour.
        """
        if initial_fresh is None:
            initial_fresh = self.deviation_initial_fresh
        # Don't sample past the song's end.
        initial_fresh = max(0, min(
            initial_fresh,
            self.total_ticks - (self.current_tick + n_lock),
            self.rollout_ticks - n_lock,
        ))

        h = self._clone_hidden(self.h_persistent)
        prev = self.prev_actual
        # Advance through (deviation_note, HOLD, HOLD, …) without sampling
        for k in range(n_lock):
            t = self.current_tick + k
            if t >= self.total_ticks:
                n_lock = k
                break
            _, h = self._forward_one(prev, t, h)
            prev = TOK_HOLD
        # Then sample only `initial_fresh` tokens. Consec_holds starts at
        # n_lock so the max-consec cap sees the forced HOLDs as part of
        # the chain.
        new_buf = [TOK_HOLD] * n_lock
        if initial_fresh > 0:
            fresh, h, prev, _ = self._sample_n(
                h, prev, initial_fresh,
                self.current_tick + n_lock, consec_holds=n_lock,
            )
            new_buf.extend(fresh)
        self._rollout_buf = new_buf
        self._h_rollout = h
        self._prev_rollout_tok = prev
        self._rollout_next_tick = self.current_tick + len(new_buf)

    # ------------------------------------------------------------------
    # Persistent state advance + deviation entry point
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _advance_persistent(self, played_token):
        """Advance h_persistent by one LSTM step using the played token.

        Forwards the LSTM with the *previous* prev_actual at the *current*
        engine tick, stores the new hidden state, then records
        `played_token` as the new prev_actual and increments current_tick.
        This is the only place h_persistent and current_tick change.
        """
        _, h = self._forward_one(self.prev_actual, self.current_tick, self.h_persistent)
        self.h_persistent = h
        self.prev_actual = int(played_token)
        self.current_tick += 1

    def _start_deviation(self, played_token):
        """Shared helper for entering a deviation.

        Used both for a fresh deviation from the normal path and for the
        "new pitch attack breaks the lock" path during a locked period.
        Advances the persistent state with the player's token, rebuilds
        the rollout around the new pitch, and arms the lock counter.
        """
        self._advance_persistent(played_token)
        self._update_current_pitch(played_token)
        n_lock = max(0, self.deviation_lock_ticks - 1)
        self._build_deviation_rollout(n_lock)
        self._locked_ticks = n_lock
        self.last_action = "deviation"

    # ------------------------------------------------------------------
    # Public commit — the per-tick API used by Piano_display.py.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def commit(self, played_token):
        """Process the player's input for one 16th-note tick.

        Decision tree:
            * Locked + new pitch attack → break the lock with a fresh
              deviation (rebuilds the rollout around the new pitch).
            * Locked + same pitch / HOLD / REST → continue the lock; commit
              a forced HOLD and shift the rollout forward.
            * REST → silence-as-match: commit the model's expected next
              token and shift the rollout forward.
            * Pitch matches expected (with tolerance for prev / next planned
              pitch) → match; commit the player's token and shift forward.
            * Pitch differs from all of {current, prev, next} → deviation;
              rebuild the rollout and start a new lock period.

        After all paths the engine's `last_action` is set so callers can
        bucket timings or render-side updates by the kind of commit.
        Returns a snapshot of the updated rollout (a list copy).
        """
        if self.current_tick >= self.total_ticks:
            self.last_action = "done"
            return list(self._rollout_buf)

        played = int(played_token)

        # ---- locked period ----
        if self._locked_ticks > 0:
            # A new pitch attack breaks out of the lock (e.g. a 2-3 note ladder)
            if played >= TOK_NOTE_BASE:
                new_pitch = PITCH_LOW + (played - TOK_NOTE_BASE)
                if new_pitch != self._current_pitch:
                    self._start_deviation(played)
                    return list(self._rollout_buf)
            # otherwise continue the lock
            self._advance_persistent(TOK_HOLD)
            if self._rollout_buf:
                self._rollout_buf.pop(0)
            self._extend_rollout_by_one()
            # Spread leftover deviation work across the lock period.
            self._catchup_extend()
            self._locked_ticks -= 1
            self.last_action = "locked"
            return list(self._rollout_buf)

        # ---- normal tick ----
        if played == TOK_REST and self._rollout_buf:
            # Silence is never a deviation. Follow the plan.
            played = int(self._rollout_buf[0])
            matched = True
            self.last_action = "silence"
        else:
            expected_pitch = (self._token_to_pitch(self._rollout_buf[0])
                              if self._rollout_buf else None)
            played_pitch = self._token_to_pitch(played)
            # Full-note tolerance buffer: a played pitch is considered a
            # match if it equals any of:
            #   * current expected pitch (the planned note right now),
            #   * previous pitch        (player attacked one tick late),
            #   * next expected pitch   (player attacked one tick early).
            # This stops the engine from re-planning when the player is just
            # slightly off-beat on a note that's still in the song.
            next_pitch = self._next_expected_pitch()
            matched = (
                played_pitch == expected_pitch
                or played_pitch == self._prev_pitch
                or (next_pitch is not None and played_pitch == next_pitch)
            )

        self._advance_persistent(played)
        self._update_current_pitch(played)

        if matched:
            self._rollout_buf.pop(0)
            self._extend_rollout_by_one()
            # Top up the rollout if a recent deviation built a short one.
            # No-op once we're back at target length.
            self._catchup_extend()
            if self.last_action != "silence":
                self.last_action = "match"
        else:
            # Deviation: persistent state was already advanced above, just
            # (re)build the rollout from there and arm the lock. Builds
            # only `deviation_initial_fresh` fresh tokens; the rest will be
            # filled in by _catchup_extend over subsequent commits.
            n_lock = max(0, self.deviation_lock_ticks - 1)
            self._build_deviation_rollout(n_lock)
            self._locked_ticks = n_lock
            self.last_action = "deviation"

        return list(self._rollout_buf)

    def rollout(self):
        """Return a read-only copy of the current rollout buffer.

        Piano_display reads this every commit to refresh the on-screen
        figures and the upcoming-spawn sheet.
        """
        return list(self._rollout_buf)


# =====================================================================
# Factory: load a checkpoint and build a ready-to-use engine in one call.
# =====================================================================

_HERE       = os.path.dirname(os.path.abspath(__file__))

# ============================================================
#   ↓↓↓  CHANGE THIS LINE TO SWITCH WHICH MODEL IS LOADED  ↓↓↓
# ============================================================
DEFAULT_CHECKPOINT = os.path.join(_HERE, "models", "jazz_lstm_v2.pt")
# ============================================================

_VOCAB_PATH = os.path.join(_HERE, "data", "processed", "chord_vocab.json")


def make_engine(chord_progression, *,
                total_bars=64,
                temperature=1.0, rollout_ticks=30,
                deviation_lock_ticks=1,
                hold_penalty=1.0, rest_penalty=1.5,
                max_consec_holds=None,
                deviation_initial_fresh=2,
                rollout_extend_per_tick=4,
                checkpoint_path=None, vocab_path=None,
                quantize=True, jit=False,
                seed=None, device="cpu"):
    """Load the trained model and build a ready-to-use JazzImprov engine.

    Convenience wrapper around `load_model` + `JazzImprov` construction +
    `set_progression` + `reset`. The returned engine is primed with its
    initial rollout and ready for the first `commit()` call.

    Defaults that are worth knowing:
        quantize = True
            Applies dynamic int8 quantisation to nn.LSTM and nn.Linear.
            Gives roughly a 3× speedup on a Raspberry Pi 4B with no
            measurable accuracy loss. Set False to keep full fp32.
        jit = False
            Set True to additionally TorchScript-compile the model.
            Removes Python dispatch overhead per forward; stacks with
            quantisation.

    The deviation_initial_fresh + rollout_extend_per_tick pair controls
    how the deviation cost is spread across ticks:
        * deviation_initial_fresh fresh tokens are sampled on the
          deviation tick itself (plus the n_lock forced HOLDs);
        * each subsequent commit extends the rollout by up to
          rollout_extend_per_tick tokens until it is back at
          rollout_ticks length.
    """
    checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
    vocab_path      = vocab_path      or _VOCAB_PATH

    model, _ = load_model(checkpoint_path, device=device,
                          quantize=quantize, jit=jit)

    with open(vocab_path) as f:
        chord_vocab = json.load(f)
    _, _, _, qual_to_id = build_chord_tables(chord_vocab)

    if seed is None:
        seed = int(np.random.SeedSequence().entropy & 0xFFFFFFFF)

    eng = JazzImprov(
        model,
        temperature=temperature,
        rollout_ticks=rollout_ticks,
        deviation_lock_ticks=deviation_lock_ticks,
        hold_penalty=hold_penalty, rest_penalty=rest_penalty,
        max_consec_holds=max_consec_holds,
        deviation_initial_fresh=deviation_initial_fresh,
        rollout_extend_per_tick=rollout_extend_per_tick,
        seed=seed, device=device,
    )
    eng.set_progression(list(chord_progression), total_bars=total_bars,
                        qual_to_id=qual_to_id)
    eng.reset()
    return eng
