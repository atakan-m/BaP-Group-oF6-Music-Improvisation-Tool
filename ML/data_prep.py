"""Convert the Weimar Jazz Database into a 16th-note grid dataset.

This is the one-off offline preprocessing step that produces the tensors
train.py later samples from. The pipeline is:

    1. Load the melody, beats, and solo_info tables from wjazzd.db.
    2. Keep only 4/4 solos with a parseable key signature.
    3. Forward-fill missing chord cells within each solo so every beat
       has a chord assigned.
    4. For each note, compute a fractional beat position from the actual
       beat onsets and quantise it to the 16th-note grid.
    5. Transpose every solo (notes and chord roots together) so its tonic
       becomes C — the model only ever sees C-tonic data.
    6. Build a per-tick token sequence per solo. Each tick carries:
            token   ∈ {REST, HOLD, NOTE_<midi>}
            chord   chord string covering the tick
            bar_pos position within the bar (0..15)

Input:
    ML/data/raw/wjazzd.db

Outputs (to ML/data/processed/):
    grid_solos.parquet   one row per tick: (melid, tick, token, chord, bar_pos)
    chord_vocab.json     sorted chord-string list + 'UNK'
    token_vocab.json     token names indexed by id
    metadata.json        vocab sizes, pitch range, grid resolution, etc.
"""

import json
import os
import sqlite3
from collections import Counter

import numpy as np
import pandas as pd

from chord_utils import parse_key, transpose_chord
from model import (
    TOK_REST, TOK_HOLD, TOK_NOTE_BASE,
    PITCH_LOW, PITCH_HIGH, VOCAB_SIZE,
    TICKS_PER_BEAT, TICKS_PER_BAR,
)


CHORD_MIN_COUNT = 10                                     # rare chords -> UNK
RAW_DIR  = os.path.join("ML", "data", "raw")
OUT_DIR  = os.path.join("ML", "data", "processed")
DB_PATH  = os.path.join(RAW_DIR, "wjazzd.db")

# Human-readable token names indexed by id. Written to token_vocab.json so
# debugging tools can map a token id back to a recognisable string.
TOKEN_NAMES = ["REST", "HOLD"] + [f"NOTE_{p}" for p in range(PITCH_LOW, PITCH_HIGH + 1)]


# ----------------------------------------------------------------------
# Load + filter
# ----------------------------------------------------------------------

def load_db():
    """Read the three relevant tables from the Weimar Jazz DB.

    Returns:
        (melody, beats, solo_info) — three pandas DataFrames, ordered as the
        SQL queries leave them so the downstream merges are deterministic.
    """
    conn = sqlite3.connect(DB_PATH)
    melody = pd.read_sql(
        "SELECT melid, onset, pitch, duration, beatdur, bar, beat "
        "FROM melody ORDER BY melid, onset", conn)
    beats = pd.read_sql(
        "SELECT melid, onset AS beat_onset, bar, beat, chord "
        "FROM beats ORDER BY melid, bar, beat", conn)
    solo_info = pd.read_sql(
        "SELECT melid, key, signature FROM solo_info", conn)
    conn.close()
    return melody, beats, solo_info


def filter_valid_solos(solo_info):
    """Return the set of melid values we keep for training.

    A solo is kept iff its time signature is 4/4 and the key field is
    parseable. The 4/4 restriction lets us use the fixed 16-tick bar
    throughout, and the key parse is required by the C-tonic transposition
    step downstream.
    """
    keep = set()
    for _, row in solo_info.iterrows():
        if row["signature"] != "4/4":
            continue
        if parse_key(row["key"]) is None:
            continue
        keep.add(int(row["melid"]))
    return keep


# ----------------------------------------------------------------------
# Per-note prep: forward-fill chords, compute fractional beat pos
# ----------------------------------------------------------------------

def prep_beats(beats):
    """Clean up the beats table and compute per-row metadata.

    Three operations:
        * forward-fill missing chord cells within each solo so every beat
          has a chord assigned;
        * compute beat_idx, the cumulative beat number within the solo;
        * compute beat_dur, the duration (in seconds) of each beat, with the
          median per solo used as a fallback on the trailing beat where the
          forward-difference is undefined.
    """
    beats = beats.copy()
    beats["chord"] = beats["chord"].astype(str).str.strip()
    beats.loc[beats["chord"] == "", "chord"] = pd.NA
    beats["chord"] = beats.groupby("melid")["chord"].ffill()
    beats = beats.dropna(subset=["chord"]).copy()

    beats = beats.sort_values(["melid", "bar", "beat"]).reset_index(drop=True)
    beats["beat_idx"] = beats.groupby("melid").cumcount()
    beats["beat_dur"] = beats.groupby("melid")["beat_onset"].diff().shift(-1)
    median_dur = beats.groupby("melid")["beat_dur"].transform("median")
    beats["beat_dur"] = beats["beat_dur"].fillna(median_dur)
    return beats


def align_notes(melody, beats):
    """Join the melody and beats tables, mapping every note to a grid tick.

    For each note we compute:
        frac      — its fractional position within the containing beat in
                    [0, 1);
        beat_pos  — absolute beat position (beat_idx + frac);
        dur_beats — duration in beats (duration / beatdur);
        tick      — quantised onset on the 16th-note grid;
        dur_ticks — quantised duration in 16ths (clamped to ≥ 1).
    """
    m = melody.dropna(subset=["bar", "beat", "pitch", "duration", "beatdur"]).copy()
    m["bar"]   = m["bar"].astype(int)
    m["beat"]  = m["beat"].astype(int)
    m["pitch"] = m["pitch"].astype(int)
    m = m.merge(beats[["melid", "bar", "beat", "beat_onset", "beat_dur", "beat_idx"]],
                on=["melid", "bar", "beat"], how="left")
    m = m.dropna(subset=["beat_onset", "beat_dur"]).copy()

    m["frac"]      = ((m["onset"] - m["beat_onset"]) / m["beat_dur"]).clip(0.0, 0.9999)
    m["beat_pos"]  = m["beat_idx"] + m["frac"]
    m["dur_beats"] = m["duration"] / m["beatdur"]
    m["tick"]      = (m["beat_pos"] * TICKS_PER_BEAT).round().astype(int)
    m["dur_ticks"] = (m["dur_beats"] * TICKS_PER_BEAT).round().clip(lower=1).astype(int)
    return m


# ----------------------------------------------------------------------
# Per-solo: build the tick-indexed token + chord + bar_pos sequence
# ----------------------------------------------------------------------

def make_solo_grid(notes, beats_by_idx):
    """Build the per-tick DataFrame for one solo.

    The notes must already be transposed to C tonic, pitch-clipped to the
    model's MIDI range, and sorted by tick. The tick origin is shifted so
    the first note lands at tick 0, and the total length is rounded up to a
    whole bar.

    Each row in the returned frame holds:
        tick     — 16th-note index 0..n_ticks-1;
        token    — REST initially, becomes NOTE_<midi> at note onsets and
                   HOLD on subsequent ticks for the note's duration;
        chord    — the chord covering the beat that tick falls in, with
                   "NC" as the fallback when no chord is known;
        bar_pos  — tick % 16.

    Returns None if the solo is too short to keep.
    """
    if len(notes) < 10:
        return None

    t0 = int(notes["tick"].min())
    notes = notes.assign(tick=notes["tick"] - t0)
    last_tick_end = int(notes["tick"].iloc[-1] + notes["dur_ticks"].iloc[-1])
    n_ticks = ((last_tick_end + TICKS_PER_BAR - 1) // TICKS_PER_BAR) * TICKS_PER_BAR
    if n_ticks < TICKS_PER_BAR:
        return None

    tokens = np.full(n_ticks, TOK_REST, dtype=np.int16)
    for tick, dur_ticks, pitch in zip(notes["tick"], notes["dur_ticks"], notes["pitch"]):
        t = int(tick)
        if t < 0 or t >= n_ticks:
            continue
        tokens[t] = TOK_NOTE_BASE + (int(pitch) - PITCH_LOW)
        for k in range(1, int(dur_ticks)):
            tt = t + k
            if tt >= n_ticks or tokens[tt] != TOK_REST:
                break
            tokens[tt] = TOK_HOLD

    # chord per tick: use the chord at the beat the tick falls in
    chord_per_tick = []
    last_chord = None
    beat0 = t0 // TICKS_PER_BEAT
    for tk in range(n_ticks):
        c = beats_by_idx.get(beat0 + tk // TICKS_PER_BEAT, last_chord)
        if c is not None:
            last_chord = c
        chord_per_tick.append(last_chord if last_chord else "NC")

    return pd.DataFrame({
        "tick":    np.arange(n_ticks, dtype=np.int32),
        "token":   tokens.astype(np.int16),
        "chord":   chord_per_tick,
        "bar_pos": (np.arange(n_ticks, dtype=np.int8) % TICKS_PER_BAR),
    })


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    """End-to-end data-prep driver. Reads the DB, writes the four outputs."""
    print(f"[1] Loading {DB_PATH}")
    melody, beats, solo_info = load_db()
    print(f"    melody    : {len(melody):>7d}")
    print(f"    beats     : {len(beats):>7d}")
    print(f"    solo_info : {len(solo_info):>7d}")

    keep = filter_valid_solos(solo_info)
    print(f"[2] 4/4 + key parseable: {len(keep)} solos")

    melody = melody[melody.melid.isin(keep)].copy()
    beats  = beats [beats.melid .isin(keep)].copy()
    beats  = prep_beats(beats)
    m      = align_notes(melody, beats)
    print(f"[3] After beat alignment: {len(m)} notes")

    # Transpose every solo so its tonic becomes C. shift = (0 - key) mod 12,
    # so e.g. a G-major solo shifts up by 5 semitones (G+5 = C). Both the
    # melody pitches and the chord roots are shifted by the same amount.
    key_lookup = dict(zip(solo_info.melid, solo_info.key))
    shifts = {mid: (0 - parse_key(key_lookup[mid])) % 12 for mid in keep}
    m["shift"]  = m["melid"].map(shifts)
    m["pitch"]  = (m["pitch"] + m["shift"]).astype(int)
    beats["shift"] = beats["melid"].map(shifts)
    beats["chord"] = beats.apply(
        lambda r: transpose_chord(r["chord"], int(r["shift"])), axis=1)

    n_before = len(m)
    m = m[(m["pitch"] >= PITCH_LOW) & (m["pitch"] <= PITCH_HIGH)].copy()
    print(f"[4] Dropped {n_before - len(m)} notes outside [{PITCH_LOW}, {PITCH_HIGH}]")

    # Build the per-solo grid frames. beats_by_solo is a beat-index → chord
    # lookup used by make_solo_grid to assign a chord to every tick.
    beats_by_solo = {mid: dict(zip(g["beat_idx"], g["chord"]))
                     for mid, g in beats.groupby("melid")}

    frames = []
    for melid, notes in m.groupby("melid"):
        grid = make_solo_grid(notes.sort_values("tick"), beats_by_solo.get(melid, {}))
        if grid is None:
            continue
        grid.insert(0, "melid", np.int32(melid))
        frames.append(grid)
    big = pd.concat(frames, ignore_index=True)
    big["chord"] = big["chord"].astype("string")
    print(f"[5] Built grids for {big.melid.nunique()} solos, {len(big)} ticks total")

    # Chord vocabulary cutoff: chords seen at least CHORD_MIN_COUNT times
    # keep their string; everything rarer is collapsed to "UNK" so the
    # quality embedding doesn't waste capacity on near-singletons.
    counts = Counter(big["chord"].tolist())
    common = {c for c, n in counts.items() if n >= CHORD_MIN_COUNT and c != "UNK"}
    big["chord"] = big["chord"].where(big["chord"].isin(common), "UNK")
    chord_vocab = sorted(common) + ["UNK"]
    print(f"[6] Chord vocab: {len(chord_vocab)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    big.to_parquet(os.path.join(OUT_DIR, "grid_solos.parquet"), index=False)
    json.dump(chord_vocab,  open(os.path.join(OUT_DIR, "chord_vocab.json"), "w"), indent=2)
    json.dump(TOKEN_NAMES,  open(os.path.join(OUT_DIR, "token_vocab.json"), "w"), indent=2)
    json.dump({
        "n_solos":          int(big["melid"].nunique()),
        "n_ticks_total":    int(len(big)),
        "n_chords":         len(chord_vocab),
        "pitch_low":        PITCH_LOW,
        "pitch_high":       PITCH_HIGH,
        "ticks_per_beat":   TICKS_PER_BEAT,
        "ticks_per_bar":    TICKS_PER_BAR,
        "token_vocab_size": VOCAB_SIZE,
        "tok_rest":         TOK_REST,
        "tok_hold":         TOK_HOLD,
        "tok_note_base":    TOK_NOTE_BASE,
        "chord_min_count":  CHORD_MIN_COUNT,
    }, open(os.path.join(OUT_DIR, "metadata.json"), "w"), indent=2)
    print(f"[7] Wrote outputs to {OUT_DIR}/")

    n = len(big)
    n_rest = int((big["token"] == TOK_REST).sum())
    n_hold = int((big["token"] == TOK_HOLD).sum())
    n_note = n - n_rest - n_hold
    print(f"\nToken distribution: "
          f"REST {n_rest/n*100:.1f}% / HOLD {n_hold/n*100:.1f}% / "
          f"NOTE {n_note/n*100:.1f}%")


if __name__ == "__main__":
    main()
