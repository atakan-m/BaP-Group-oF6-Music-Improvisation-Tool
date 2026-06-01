"""
Real-time inference engine for the BaP jazz improvisation tool.

The engine wraps a trained JazzLSTM and supports:
  * stateful per-16th-note advancement
  * a maintained "next 10 seconds" rollout buffer for the synthesia display
  * cheap re-planning when the player deviates from the predicted note

Token vocab (matches process_to_grid.py and generate.py):
    0       REST
    1       HOLD (continuation of previous note)
    2..58   NOTE_<midi>  for midi in [40, 96]

Typical usage from the SP/HW side:

    engine = JazzImprov("ML/checkpoints/jazz_lstm.pt")
    engine.set_progression([("Dm7", 1), ("G7", 1), ("Cj7", 2)], total_bars=64)
    engine.reset()

    rollout = engine.rollout()       # tokens for ticks 0..52  -> draw on display

    # Then on every 16th note from the SP layer:
    while not engine.is_done:
        played_token = sp_get_player_token_for_current_tick()
        rollout = engine.commit(played_token)   # tokens for ticks t..t+52
        # send updated rollout to the display
"""

import os
import json
import argparse

import numpy as np
import torch

from model import (
    JazzLSTM,
    parse_chord,
    UNK_ROOT_ID,
)


# ----------------------------------------------------------------------
# Vocab constants (keep in sync with process_to_grid.py)
# ----------------------------------------------------------------------
TOK_REST       = 0
TOK_HOLD       = 1
TOK_NOTE_BASE  = 2
PITCH_LOW      = 40
PITCH_HIGH     = 96
TICKS_PER_BEAT = 4
TICKS_PER_BAR  = 16


# ----------------------------------------------------------------------
# Tiny utilities the SP layer can use to convert between MIDI and tokens
# ----------------------------------------------------------------------
def pitch_to_note_token(midi_pitch):
    """MIDI pitch -> NOTE_x token, clipped to vocab range."""
    p = int(np.clip(int(midi_pitch), PITCH_LOW, PITCH_HIGH))
    return TOK_NOTE_BASE + (p - PITCH_LOW)


def token_to_pitch(token):
    """NOTE_x token -> MIDI pitch (or None for REST/HOLD)."""
    if int(token) >= TOK_NOTE_BASE:
        return PITCH_LOW + (int(token) - TOK_NOTE_BASE)
    return None


def tokens_to_events(tokens, base_tick=0):
    """Token sequence -> list of (onset_tick, midi_pitch, duration_ticks).
    Useful for translating a rollout into something the display can draw."""
    events, i, n = [], 0, len(tokens)
    while i < n:
        tok = int(tokens[i])
        if tok >= TOK_NOTE_BASE:
            midi = PITCH_LOW + (tok - TOK_NOTE_BASE)
            dur = 1
            j = i + 1
            while j < n and int(tokens[j]) == TOK_HOLD:
                dur += 1
                j += 1
            events.append((base_tick + i, midi, dur))
            i = j
        else:
            i += 1
    return events


# ----------------------------------------------------------------------
# Chord <-> ids
# ----------------------------------------------------------------------
def build_qual_map(chord_vocab):
    qualities = set()
    for c in chord_vocab:
        p = parse_chord(c)
        if p is not None:
            qualities.add(p[1])
    qualities = sorted(qualities) + ["__UNK_Q__"]
    return {q: i for i, q in enumerate(qualities)}


def chord_to_root_qual(chord_str, qual_to_id):
    unk_qid = qual_to_id["__UNK_Q__"]
    p = parse_chord(chord_str)
    if p is None:
        return UNK_ROOT_ID, unk_qid
    root, q = p
    return root, qual_to_id.get(q, unk_qid)


# ----------------------------------------------------------------------
# The engine
# ----------------------------------------------------------------------
class JazzImprov:
    """Stateful real-time wrapper around a trained JazzLSTM.

    Three pieces of state are kept:
      * Persistent LSTM state (h_persistent, prev_actual, current_tick) reflecting
        what the *player actually played* so far.
      * Rollout buffer (a list of predicted future tokens) for the display.
      * Rollout LSTM state (h_rollout, prev_rollout_tok) so we can cheaply extend
        the rollout by one when the player matches the prediction.
    """

    def __init__(
        self,
        checkpoint_path,
        data_dir="ML/data/processed",
        temperature=1.0,
        rollout_ticks=53,            # ~10s at 80bpm, 16ths per beat
        deviation_lock_ticks=4,      # quarter-note lock after a deviation
        seed=0,
        device=None,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.temperature = float(temperature)
        self.rollout_ticks = int(rollout_ticks)
        self.deviation_lock_ticks = int(deviation_lock_ticks)
        self._rng = np.random.default_rng(seed)

        # --- load model
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model = JazzLSTM(
            vocab_size=ckpt["vocab_size"],
            n_qualities=ckpt["n_qualities"],
            hidden=ckpt["hidden"],
            n_layers=ckpt["n_layers"],
            dropout=ckpt["dropout"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.vocab_size = ckpt["vocab_size"]

        # --- chord vocab
        with open(os.path.join(data_dir, "chord_vocab.json")) as f:
            chord_vocab = json.load(f)
        self.qual_to_id = build_qual_map(chord_vocab)

        # --- progression / per-tick features (set by set_progression)
        self.chord_root = None
        self.chord_qual = None
        self.bar_chords = None
        self.total_ticks = 0

        # --- runtime state (set by reset)
        self.current_tick = 0
        self.prev_actual = TOK_REST
        self.h_persistent = None
        self._rollout_buf = []
        self._h_rollout = None
        self._prev_rollout_tok = TOK_REST
        self._rollout_next_tick = 0
        # When > 0, the next `_locked_ticks` commits are forced to behave as
        # if the player is holding the deviation note (no new re-plan; the
        # rollout is shifted/extended each tick). Set by a deviation commit.
        self._locked_ticks = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def set_progression(self, chord_progression, total_bars):
        """chord_progression: list of (chord_str, n_bars). Cycles to fill total_bars.

        Chord strings should be expressed *as if the song were in C* — the model
        was trained on data transposed to C-tonic.
        """
        bar_chords = []
        for chord_str, nb in chord_progression:
            for _ in range(int(nb)):
                bar_chords.append(chord_str)
        cycle = list(bar_chords) or ["NC"]
        while len(bar_chords) < total_bars:
            bar_chords += cycle
        bar_chords = bar_chords[:total_bars]

        n_total = int(total_bars) * TICKS_PER_BAR
        cr = np.zeros(n_total, dtype=np.int64)
        cq = np.zeros(n_total, dtype=np.int64)
        for bi, c in enumerate(bar_chords):
            r, q = chord_to_root_qual(c, self.qual_to_id)
            t0 = bi * TICKS_PER_BAR
            cr[t0:t0 + TICKS_PER_BAR] = r
            cq[t0:t0 + TICKS_PER_BAR] = q

        self.chord_root = cr
        self.chord_qual = cq
        self.bar_chords = bar_chords
        self.total_ticks = n_total

    def reset(self):
        """Clear all state and prime an initial rollout starting at tick 0."""
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
        self._refresh_rollout()

    # ------------------------------------------------------------------
    # LSTM primitives
    # ------------------------------------------------------------------
    def _features_for_tick(self, t):
        return (
            int(self.chord_root[t]),
            int(self.chord_qual[t]),
            int(t % TICKS_PER_BAR),
        )

    def _forward_one(self, prev_token, t, hidden):
        """One LSTM step that predicts the token at tick t.
        Returns (logits[vocab_size], new_hidden)."""
        cr, cq, bp = self._features_for_tick(t)
        pt  = torch.tensor([[int(prev_token)]], dtype=torch.long, device=self.device)
        crt = torch.tensor([[cr]],              dtype=torch.long, device=self.device)
        cqt = torch.tensor([[cq]],              dtype=torch.long, device=self.device)
        bpt = torch.tensor([[bp]],              dtype=torch.long, device=self.device)
        logits, hidden = self.model(pt, crt, cqt, bpt, hidden)
        return logits[0, 0], hidden

    def _sample(self, logits):
        l = logits.detach().cpu().numpy().astype(np.float64)
        l /= max(self.temperature, 1e-6)
        l -= l.max()
        p = np.exp(l)
        p /= p.sum()
        return int(self._rng.choice(len(p), p=p))

    @staticmethod
    def _clone_hidden(hidden):
        if hidden is None:
            return None
        h, c = hidden
        return (h.clone(), c.clone())

    # ------------------------------------------------------------------
    # Rollout management (private)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _refresh_rollout(self):
        """Rebuild rollout buffer from the current persistent state."""
        h = self._clone_hidden(self.h_persistent)
        prev = self.prev_actual
        buf = []
        end_tick = min(self.current_tick + self.rollout_ticks, self.total_ticks)
        for t in range(self.current_tick, end_tick):
            logits, h = self._forward_one(prev, t, h)
            nxt = self._sample(logits)
            buf.append(nxt)
            prev = nxt
        self._rollout_buf = buf
        self._h_rollout = h
        self._prev_rollout_tok = prev
        self._rollout_next_tick = end_tick

    @torch.no_grad()
    def _extend_rollout_by_one(self):
        t = self._rollout_next_tick
        if t >= self.total_ticks:
            return
        logits, h = self._forward_one(self._prev_rollout_tok, t, self._h_rollout)
        nxt = self._sample(logits)
        self._rollout_buf.append(nxt)
        self._h_rollout = h
        self._prev_rollout_tok = nxt
        self._rollout_next_tick = t + 1

    @torch.no_grad()
    def _advance_persistent(self, played_token):
        """Advance persistent state by one tick using the actually-played token.

        Feeds the LSTM (prev_actual, chord_at_current_tick, ...) which advances
        h_persistent into the state ready to predict the *next* tick. The newly
        observed played_token then becomes the next prev_actual.
        """
        t = self.current_tick
        _logits, h_next = self._forward_one(self.prev_actual, t, self.h_persistent)
        self.h_persistent = h_next
        self.prev_actual = int(played_token)
        self.current_tick = t + 1

    @torch.no_grad()
    def _build_deviation_rollout(self, n_lock):
        """Build a new rollout buffer that begins with `n_lock` HOLD tokens
        (the visual continuation of the just-played deviation note) followed
        by fresh predictions resumed from a state that has been simulated
        forward through those HOLDs.

        Called right after `_advance_persistent(played_token)` on a deviation,
        so `current_tick` is one past the deviation tick and `prev_actual`
        equals the deviation note. The persistent state is *not* modified
        here — the simulated advance happens on a clone."""
        # Simulate the next `n_lock` HOLD inputs in a clone so the rollout
        # state ends up at "ready to predict (current_tick + n_lock)".
        h_sim = self._clone_hidden(self.h_persistent)
        prev = self.prev_actual
        for k in range(n_lock):
            t = self.current_tick + k
            if t >= self.total_ticks:
                n_lock = k
                break
            _logits, h_sim = self._forward_one(prev, t, h_sim)
            prev = TOK_HOLD

        new_buf = [TOK_HOLD] * n_lock
        rollout_start = self.current_tick + n_lock
        n_more = self.rollout_ticks - n_lock
        end_tick = min(rollout_start + n_more, self.total_ticks)
        for t in range(rollout_start, end_tick):
            logits, h_sim = self._forward_one(prev, t, h_sim)
            nxt = self._sample(logits)
            new_buf.append(nxt)
            prev = nxt

        self._rollout_buf = new_buf
        self._h_rollout = h_sim
        self._prev_rollout_tok = prev
        self._rollout_next_tick = end_tick

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def commit(self, played_token):
        """Commit what the player actually played at the *current* tick.

        Returns the updated rollout (list of token ids) covering the next
        `rollout_ticks` ticks starting from the new current_tick.

        Behaviour:
          * **Match** (`played_token == rollout[0]`): the rollout shifts by
            one — the displayed future stays stable for the player.
          * **Deviation** (mismatch): the next `deviation_lock_ticks - 1`
            ticks are *locked* to display HOLDs (the deviation note is held
            as a quarter note). The remainder of the rollout is regenerated
            from a state simulated forward through those HOLDs. During the
            lock, subsequent `commit()` calls ignore the player's input and
            simply shift the rollout — giving the player time to read the
            new future and the model time to recover.
        """
        if self.current_tick >= self.total_ticks:
            return list(self._rollout_buf)

        # --- locked period: ignore played, advance with HOLD, shift+extend
        if self._locked_ticks > 0:
            self._advance_persistent(TOK_HOLD)
            if self._rollout_buf:
                self._rollout_buf.pop(0)
            self._extend_rollout_by_one()
            self._locked_ticks -= 1
            return list(self._rollout_buf)

        # --- normal tick
        matched = (
            len(self._rollout_buf) > 0
            and int(played_token) == int(self._rollout_buf[0])
        )
        self._advance_persistent(played_token)

        if matched:
            self._rollout_buf.pop(0)
            self._extend_rollout_by_one()
        else:
            # Deviation: lock the next (deviation_lock_ticks - 1) ticks and
            # rebuild the rollout starting from the state at tick T+lock.
            n_lock = max(0, self.deviation_lock_ticks - 1)
            self._build_deviation_rollout(n_lock)
            self._locked_ticks = n_lock

        return list(self._rollout_buf)

    def rollout(self):
        """Snapshot of the current rollout buffer (no state change)."""
        return list(self._rollout_buf)

    @property
    def is_done(self):
        return self.current_tick >= self.total_ticks


# ----------------------------------------------------------------------
# Demo: simulate a "perfect follower" loop and optionally a one-shot
# deviation, dump everything to MIDI so you can listen.
# ----------------------------------------------------------------------
def _demo():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",  default="ML/checkpoints/jazz_lstm.pt")
    ap.add_argument("--chords",      default="Dm7,G7,Cj7,Cj7")
    ap.add_argument("--chord_bars",  type=int, default=1)
    ap.add_argument("--bars",        type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--deviate_at",  type=int, default=-1,
                    help="if >=0, override the player's token at this tick with a random NOTE")
    ap.add_argument("--deviate_pitch", type=int, default=-1,
                    help="if --deviate_at is set, force this MIDI pitch (else random)")
    ap.add_argument("--out",         default="realtime_demo.mid")
    args = ap.parse_args()

    engine = JazzImprov(
        args.checkpoint,
        temperature=args.temperature,
        seed=args.seed,
    )
    prog = [(c.strip(), args.chord_bars) for c in args.chords.split(",") if c.strip()]
    engine.set_progression(prog, total_bars=args.bars)
    engine.reset()

    print(f"checkpoint   : {args.checkpoint}")
    print(f"progression  : {[c for c, _ in prog]}  x {args.chord_bars} bar(s) each")
    print(f"total ticks  : {engine.total_ticks}")
    print(f"rollout size : {engine.rollout_ticks} ticks (~10s @ 80bpm)")
    print(f"first 20 of initial rollout: {engine.rollout()[:20]}")
    print()

    # Simulate a perfect player: at each tick, play whatever the model predicts.
    # Optionally swap in a forced note at one specific tick to test re-planning.
    played_sequence = []
    n_refreshes = 0
    rng = np.random.RandomState(args.seed + 1)

    while not engine.is_done:
        rb = engine.rollout()
        if not rb:
            break
        predicted = rb[0]
        played = predicted
        if engine.current_tick == args.deviate_at:
            if args.deviate_pitch >= 0:
                played = pitch_to_note_token(args.deviate_pitch)
            else:
                played = TOK_NOTE_BASE + int(rng.randint(PITCH_HIGH - PITCH_LOW + 1))
            print(f"  deviation @ tick {engine.current_tick}: "
                  f"predicted={predicted}, playing={played} "
                  f"(midi={token_to_pitch(played)})")
            n_refreshes += 1
        played_sequence.append(played)
        engine.commit(played)

    print(f"\nsimulated {len(played_sequence)} ticks, "
          f"{n_refreshes} forced deviations")

    # Stats
    pseq = np.array(played_sequence)
    n = len(pseq)
    n_rest = int((pseq == TOK_REST).sum())
    n_hold = int((pseq == TOK_HOLD).sum())
    n_note = n - n_rest - n_hold
    print(f"  REST: {100*n_rest/n:.1f}%   HOLD: {100*n_hold/n:.1f}%   "
          f"NOTE: {100*n_note/n:.1f}%")

    # Write MIDI of the committed sequence so we can listen
    from generate import write_midi
    events = tokens_to_events(played_sequence)
    write_midi(events, engine.bar_chords, args.out, tempo_bpm=80)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    _demo()
