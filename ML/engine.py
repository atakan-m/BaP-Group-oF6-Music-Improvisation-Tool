"""Realtime inference engine for the BaP improvisation tool.

What lives here:
    - `MicTickAggregator`      maps SP audio-block detections -> tokens per 16th
    - `make_sampler`           closure that does penalized + capped sampling
    - `JazzImprov`             stateful per-16th-tick wrapper around JazzLSTM
    - `make_engine`            one-call factory: load model + build engine

The public entry point used by Piano_display.py is `make_engine` (re-exported
from `generate.py` as `make_realtime_engine` for backwards-compat).
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
    """MIDI pitch -> NOTE_x token, clipped to the model's vocab range."""
    p = max(PITCH_LOW, min(PITCH_HIGH, int(midi_pitch)))
    return TOK_NOTE_BASE + (p - PITCH_LOW)


def token_to_pitch(tok):
    """NOTE_x token -> MIDI pitch, or None for REST/HOLD."""
    tok = int(tok)
    return PITCH_LOW + (tok - TOK_NOTE_BASE) if tok >= TOK_NOTE_BASE else None


_NOTE_NAME_TO_PC = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
                    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def _note_name_to_midi(name):
    """'C4' -> 60. None / unparseable -> None. (For SP detectors that
    expose `last_note` as a string instead of `last_midi` as an int.)"""
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
    """Collects pitch detections during one 16th-note window and emits
    a single REST / HOLD / NOTE token at the tick boundary.

    Connect by wrapping the SP detector callback:

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

    DEFAULT_MIN_MIDI = 48     # C3
    DEFAULT_MAX_MIDI = 83     # B5

    def __init__(self, min_midi=None, max_midi=None):
        self._midis = []
        self._prev_pitch = None
        self.min_midi = self.DEFAULT_MIN_MIDI if min_midi is None else int(min_midi)
        self.max_midi = self.DEFAULT_MAX_MIDI if max_midi is None else int(max_midi)

    def _gate(self, midi):
        """Reject out-of-range pitches (typically HPS octave errors or noise)."""
        if midi is None:
            return None
        try:
            m = int(midi)
        except (TypeError, ValueError):
            return None
        return m if self.min_midi <= m <= self.max_midi else None

    def observe(self, midi_or_none):
        if midi_or_none is None:
            self._midis.append(None)
        elif isinstance(midi_or_none, (int, float)):
            self._midis.append(self._gate(midi_or_none))
        elif isinstance(midi_or_none, str):
            self._midis.append(self._gate(_note_name_to_midi(midi_or_none)))
        else:
            self._midis.append(None)

    def consume_token(self):
        """Take majority vote over the just-finished window, reset, return token."""
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
    """Build a sampling closure with fixed hyperparams. Returns:

        sample(logits, consec_holds=0) -> token_id

    Applied (in order) before softmax:
        - HOLD logit -= hold_penalty                (soft bias)
        - REST logit -= rest_penalty                (soft bias)
        - HOLD logit -> -inf  if consec_holds >= max_consec_holds  (hard cap)
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
    """Three pieces of state, conceptually:

      * Persistent  (h_persistent, prev_actual, current_tick, _current_pitch)
        = the player's real history. Only advances on commit().

      * Rollout buffer (_rollout_buf + _h_rollout + _prev_rollout_tok)
        = the model's prediction for the next ~10s. Shifts on matches,
        rebuilt from scratch on deviations.

      * Lock counter (_locked_ticks)
        = N ticks during which player input is forced to HOLD (the half/quarter-
        note display behavior after a deviation), unless they attack a new pitch.

    The public surface is set_progression / reset / commit / rollout.
    """

    def __init__(self, model, *, temperature=1.0,
                 rollout_ticks=30, deviation_lock_ticks=1,
                 hold_penalty=1.0, rest_penalty=1.5,
                 max_consec_holds=None,
                 seed=0, device="cpu"):
        self.model = model
        self.device = torch.device(device)
        self.deviation_lock_ticks = int(deviation_lock_ticks)
        self.rollout_ticks = int(rollout_ticks)

        self._rng = np.random.default_rng(seed)
        self._sample = make_sampler(
            temperature=temperature, hold_penalty=hold_penalty,
            rest_penalty=rest_penalty, max_consec_holds=max_consec_holds,
            rng=self._rng,
        )

        # progression-derived per-tick features (set by set_progression)
        self.chord_root = None
        self.chord_qual = None
        self.bar_chords = None
        self.total_ticks = 0

        # runtime state (set by reset)
        self.current_tick = 0
        self.prev_actual = TOK_REST
        self.h_persistent = None
        self._rollout_buf = []
        self._h_rollout = None
        self._prev_rollout_tok = TOK_REST
        self._rollout_next_tick = 0
        self._locked_ticks = 0
        self._current_pitch = DEFAULT_CURRENT_PITCH
        self.last_action = "init"   # "match" | "silence" | "deviation" | "locked" | "done" | "init"

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def current_pitch(self):
        return self._current_pitch

    @property
    def is_done(self):
        return self.current_tick >= self.total_ticks

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def set_progression(self, chord_progression, total_bars, qual_to_id):
        """Lay out the per-tick chord_root + chord_qual feature arrays.

        Args:
            chord_progression : list of (chord_str, n_bars). Cycled to fill.
            total_bars        : how many bars the song will last.
            qual_to_id        : the quality lookup from build_chord_tables.
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
        self.last_action = "init"
        self._refresh_rollout()

    # ------------------------------------------------------------------
    # LSTM primitives
    # ------------------------------------------------------------------
    def _forward_one(self, prev_token, t, hidden):
        """One LSTM step that predicts the token at tick `t`. Returns
        (logits[vocab_size], new_hidden)."""
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
        if h is None:
            return None
        return (h[0].clone(), h[1].clone())

    @staticmethod
    def _count_trailing_holds(buf):
        n = 0
        for tok in reversed(buf):
            if int(tok) == TOK_HOLD:
                n += 1
            else:
                break
        return n

    # ------------------------------------------------------------------
    # Pitch tracking (so HOLD/REST resolve to a pitch for the match check)
    # ------------------------------------------------------------------
    def _token_to_pitch(self, tok):
        if int(tok) >= TOK_NOTE_BASE:
            return PITCH_LOW + (int(tok) - TOK_NOTE_BASE)
        return self._current_pitch

    def _update_current_pitch(self, tok):
        if int(tok) >= TOK_NOTE_BASE:
            self._current_pitch = PITCH_LOW + (int(tok) - TOK_NOTE_BASE)

    # ------------------------------------------------------------------
    # Rollout machinery — all three paths funnel through `_sample_n`
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sample_n(self, h, prev, n, start_tick, consec_holds=0):
        """Sample up to `n` new tokens starting from given hidden state.
        Returns (tokens, final_h, final_prev, final_consec_holds)."""
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
        """Build rollout from scratch using a clone of the persistent state."""
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
        """Append one new prediction at the end of the rollout."""
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
    def _build_deviation_rollout(self, n_lock):
        """Rebuild rollout starting with `n_lock` forced HOLDs (visual sustain
        of the deviation note) followed by fresh predictions sampled from a
        state that has been simulated forward through those HOLDs."""
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
        # Then sample the remainder. Consec_holds starts at n_lock so the
        # max-consec cap sees the forced HOLDs as part of the chain.
        new_buf = [TOK_HOLD] * n_lock
        fresh, h, prev, _ = self._sample_n(
            h, prev, self.rollout_ticks - n_lock,
            self.current_tick + n_lock, consec_holds=n_lock,
        )
        new_buf.extend(fresh)
        self._rollout_buf = new_buf
        self._h_rollout = h
        self._prev_rollout_tok = prev
        self._rollout_next_tick = self.current_tick + len(new_buf)

    # ------------------------------------------------------------------
    # Persistent state advance
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _advance_persistent(self, played_token):
        """Advance h_persistent by one LSTM step using the actually-played token."""
        _, h = self._forward_one(self.prev_actual, self.current_tick, self.h_persistent)
        self.h_persistent = h
        self.prev_actual = int(played_token)
        self.current_tick += 1

    def _start_deviation(self, played_token):
        """Common path for both 'fresh deviation' and 'deviation during lock':
        advance persistent state, rebuild rollout, restart lock."""
        self._advance_persistent(played_token)
        self._update_current_pitch(played_token)
        n_lock = max(0, self.deviation_lock_ticks - 1)
        self._build_deviation_rollout(n_lock)
        self._locked_ticks = n_lock
        self.last_action = "deviation"

    # ------------------------------------------------------------------
    # Public commit
    # ------------------------------------------------------------------
    @torch.no_grad()
    def commit(self, played_token):
        """Tell the engine what the player played at the current tick.

        Behaviour:
          * **locked + new pitch attack**: break out → fresh deviation
          * **locked + same pitch / HOLD / REST**: continue lock, shift rollout
          * **REST**: silence is never a deviation; commit the model's
            expected next token instead → rollout shifts normally
          * **pitch matches expected**: match → rollout shifts + extends
          * **pitch differs**: deviation → rebuild rollout, start lock

        Returns the updated rollout (snapshot of the next `rollout_ticks` tokens).
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
            matched = played_pitch == expected_pitch

        self._advance_persistent(played)
        self._update_current_pitch(played)

        if matched:
            self._rollout_buf.pop(0)
            self._extend_rollout_by_one()
            if self.last_action != "silence":
                self.last_action = "match"
        else:
            # Deviation: persistent state was already advanced above, just
            # (re)build the rollout from there and arm the lock.
            n_lock = max(0, self.deviation_lock_ticks - 1)
            self._build_deviation_rollout(n_lock)
            self._locked_ticks = n_lock
            self.last_action = "deviation"

        return list(self._rollout_buf)

    def rollout(self):
        """Read-only copy of the current rollout buffer."""
        return list(self._rollout_buf)


# =====================================================================
# Factory: load checkpoint + build engine in one call
# =====================================================================

_HERE       = os.path.dirname(os.path.abspath(__file__))

# ============================================================
#   ↓↓↓  CHANGE THIS LINE TO SWITCH WHICH MODEL IS LOADED  ↓↓↓
# ============================================================
DEFAULT_CHECKPOINT = os.path.join(_HERE, "models", "jazz_lstm.pt")
# ============================================================

_VOCAB_PATH = os.path.join(_HERE, "data", "processed", "chord_vocab.json")


def make_engine(chord_progression, *,
                total_bars=64,
                temperature=1.0, rollout_ticks=30,
                deviation_lock_ticks=1,
                hold_penalty=1.0, rest_penalty=1.5,
                max_consec_holds=None,
                checkpoint_path=None, vocab_path=None,
                quantize=True, jit=False,
                seed=None, device="cpu"):
    """Load the trained model and build a ready-to-use JazzImprov engine.

    Default `quantize=True` applies int8 dynamic quantization for fast Pi
    inference (~3x speedup, no measurable quality loss). Set False to keep
    full fp32. Set `jit=True` to additionally TorchScript-compile.
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
        seed=seed, device=device,
    )
    eng.set_progression(list(chord_progression), total_bars=total_bars,
                        qual_to_id=qual_to_id)
    eng.reset()
    return eng
