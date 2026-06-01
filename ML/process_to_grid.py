"""
Process the Weimar Jazz Database into a 16th-note grid dataset.

Pipeline:
  1. Load melody + beats + solo_info from wjazzd.db
  2. Keep only 4/4 solos with a parseable key and >= 10 notes
  3. Forward-fill empty chord cells within each solo
  4. For each note: compute a fractional beat position using real beat onsets
     from the beats table, then round to the nearest 16th-note tick
  5. Transpose every solo (pitches + chord roots) to C
  6. Build a per-tick token sequence per solo:
        token = REST | HOLD | NOTE_<midi_pitch>
     plus per-tick chord and position-within-bar (0..15)
  7. Save:
        ML/data/processed/grid_solos.parquet  (melid, tick, token, chord, bar_pos)
        ML/data/processed/chord_vocab.json    (sorted chord strings + 'UNK')
        ML/data/processed/token_vocab.json    (token names by id)
        ML/data/processed/metadata.json       (pitch range, vocab sizes, etc.)

Input  : ML/data/raw/wjazzd.db
"""

import os
import json
import sqlite3
from collections import Counter

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Chord / key parsing  (lifted from process_wjd.py and unchanged)
# ----------------------------------------------------------------------
NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def parse_root(token):
    if not token:
        return None
    letter = token[0].upper()
    if letter not in NOTE_TO_PC:
        return None
    pc = NOTE_TO_PC[letter]
    i = 1
    if i < len(token):
        if token[i] == "#":
            pc = (pc + 1) % 12
            i += 1
        elif token[i] == "b":
            pc = (pc - 1) % 12
            i += 1
    return pc, token[i:]


def parse_key(s):
    if not s or not str(s).strip():
        return None
    head = str(s).split("-")[0].strip()
    parsed = parse_root(head)
    return None if parsed is None else parsed[0]


def normalize_quality(q):
    if not q:
        return ""
    for a, b in [
        ("maj7", "j7"), ("Maj7", "j7"), ("MAJ7", "j7"), ("M7", "j7"),
        ("Δ", "j"), ("maj", "j"), ("Maj", "j"),
        ("min7", "-7"), ("Min7", "-7"), ("min", "-"), ("Min", "-"),
        ("dim", "o"), ("aug", "+"),
    ]:
        q = q.replace(a, b)
    if q.startswith("m") and (len(q) == 1 or not q[1].isalpha()):
        q = "-" + q[1:]
    return q


def parse_chord(s):
    s = (s or "").strip()
    if not s or s == "NC":
        return None
    parsed = parse_root(s)
    if parsed is None:
        return None
    pc, rest = parsed
    return pc, normalize_quality(rest)


def chord_str(pc, quality):
    return PITCH_NAMES[pc] + quality


def transpose_chord(chord_string, semitones):
    parsed = parse_chord(chord_string)
    if parsed is None:
        return chord_string
    pc, q = parsed
    return chord_str((pc + semitones) % 12, q)


# ----------------------------------------------------------------------
# Grid parameters
# ----------------------------------------------------------------------
TICKS_PER_BEAT = 4                       # 16th notes in 4/4
TICKS_PER_BAR = 4 * TICKS_PER_BEAT        # 16

PITCH_LOW, PITCH_HIGH = 40, 96            # 57 pitches (E2..C7)
N_PITCHES = PITCH_HIGH - PITCH_LOW + 1

# Token vocabulary:
#   0       = REST
#   1       = HOLD (continuation of previous note)
#   2..58   = NOTE_<midi>  for midi in [PITCH_LOW, PITCH_HIGH]
TOK_REST = 0
TOK_HOLD = 1
TOK_NOTE_BASE = 2
TOKEN_VOCAB_SIZE = 2 + N_PITCHES

TOKEN_NAMES = ["REST", "HOLD"] + [f"NOTE_{p}" for p in range(PITCH_LOW, PITCH_HIGH + 1)]

CHORD_MIN_COUNT = 10                      # rare chords -> UNK

RAW_DIR = os.path.join("ML", "data", "raw")
OUT_DIR = os.path.join("ML", "data", "processed")
DB_PATH = os.path.join(RAW_DIR, "wjazzd.db")


# ----------------------------------------------------------------------
# 1. Load source data
# ----------------------------------------------------------------------
print(f"[1] Loading {DB_PATH}")
conn = sqlite3.connect(DB_PATH)

melody = pd.read_sql(
    "SELECT melid, onset, pitch, duration, beatdur, bar, beat "
    "FROM melody ORDER BY melid, onset", conn)
beats = pd.read_sql(
    "SELECT melid, onset AS beat_onset, bar, beat, chord "
    "FROM beats ORDER BY melid, bar, beat", conn)
solo_info = pd.read_sql(
    "SELECT melid, key, signature, avgtempo FROM solo_info", conn)
conn.close()
print(f"    melody    : {len(melody):>7d} notes")
print(f"    beats     : {len(beats):>7d} beat rows")
print(f"    solo_info : {len(solo_info):>7d} solos")


# ----------------------------------------------------------------------
# 2. Pick the solos we'll keep:  4/4 + parseable key
# ----------------------------------------------------------------------
key_lookup = dict(zip(solo_info.melid, solo_info.key))
sig_lookup = dict(zip(solo_info.melid, solo_info.signature))

valid_melids = []
for mid in solo_info.melid:
    if sig_lookup.get(mid) != "4/4":
        continue
    if parse_key(key_lookup.get(mid)) is None:
        continue
    valid_melids.append(int(mid))
print(f"[2] 4/4 + key parseable: {len(valid_melids)} solos")

melody = melody[melody.melid.isin(valid_melids)].copy()
beats  = beats[beats.melid.isin(valid_melids)].copy()


# ----------------------------------------------------------------------
# 3. Forward-fill chord cells within each solo, drop heads with no chord
# ----------------------------------------------------------------------
beats["chord"] = beats["chord"].astype(str).str.strip()
beats.loc[beats["chord"] == "", "chord"] = pd.NA
beats["chord"] = beats.groupby("melid")["chord"].ffill()
beats = beats.dropna(subset=["chord"]).copy()


# ----------------------------------------------------------------------
# 4. Per-beat helpers: sequential beat index + duration to next beat
# ----------------------------------------------------------------------
beats = beats.sort_values(["melid", "bar", "beat"]).reset_index(drop=True)
beats["beat_idx"] = beats.groupby("melid").cumcount()                       # 0,1,2,...
beats["beat_dur"] = beats.groupby("melid")["beat_onset"].diff().shift(-1)    # next - this
# Fill the last beat of each solo with the solo's median beat duration
median_dur = beats.groupby("melid")["beat_dur"].transform("median")
beats["beat_dur"] = beats["beat_dur"].fillna(median_dur)


# ----------------------------------------------------------------------
# 5. Drop notes outside any beat row, compute fractional beat position
# ----------------------------------------------------------------------
melody = melody.dropna(subset=["bar", "beat", "pitch", "duration", "beatdur"]).copy()
melody["bar"]  = melody["bar"].astype(int)
melody["beat"] = melody["beat"].astype(int)
melody["pitch"] = melody["pitch"].astype(int)

m = melody.merge(
    beats[["melid", "bar", "beat", "beat_onset", "beat_dur", "beat_idx"]],
    on=["melid", "bar", "beat"], how="left")
m = m.dropna(subset=["beat_onset", "beat_dur"]).copy()

# Fractional position within beat using real timing
m["frac"] = (m["onset"] - m["beat_onset"]) / m["beat_dur"]
m["frac"] = m["frac"].clip(lower=0.0, upper=0.9999)        # don't spill into next beat
m["beat_pos"] = m["beat_idx"] + m["frac"]                  # in beats from solo start

# Duration in beats (use note's local beatdur)
m["dur_beats"] = m["duration"] / m["beatdur"]

# Quantize onset to 16th-note tick, duration to integer ticks (>=1)
m["tick"]      = (m["beat_pos"] * TICKS_PER_BEAT).round().astype(int)
m["dur_ticks"] = (m["dur_beats"] * TICKS_PER_BEAT).round().clip(lower=1).astype(int)
print(f"[5] After alignment: {len(m)} notes")


# ----------------------------------------------------------------------
# 6. Transpose: shift every solo so its tonic becomes C
# ----------------------------------------------------------------------
shifts = {mid: (0 - parse_key(key_lookup[mid])) % 12 for mid in valid_melids}

m["shift"]  = m["melid"].map(shifts)
m["pitch"]  = (m["pitch"] + m["shift"]).astype(int)

beats = beats.merge(
    pd.Series(shifts, name="shift"), left_on="melid", right_index=True, how="left")
beats["chord"] = beats.apply(
    lambda r: transpose_chord(r["chord"], int(r["shift"])), axis=1)

# Drop notes outside our pitch window
n_before = len(m)
m = m[(m["pitch"] >= PITCH_LOW) & (m["pitch"] <= PITCH_HIGH)].copy()
print(f"[6] Dropped {n_before - len(m)} notes outside [{PITCH_LOW},{PITCH_HIGH}] "
      f"({100*(n_before-len(m))/max(1,n_before):.2f}%)")


# ----------------------------------------------------------------------
# 7. Build per-solo grid sequences
# ----------------------------------------------------------------------
records = []
n_total_ticks = 0
n_kept_solos = 0

# Pre-index beats per solo:  beat_idx -> chord
beats_by_solo = {
    mid: dict(zip(g["beat_idx"], g["chord"]))
    for mid, g in beats.groupby("melid")
}

for melid, notes in m.groupby("melid"):
    notes = notes.sort_values("tick")
    if len(notes) < 10:
        continue

    # Make ticks start at 0 (some solos have negative bars / pre-bar notes)
    t0 = int(notes["tick"].min())
    notes = notes.assign(tick=notes["tick"] - t0)

    # Total length: last note onset + its duration, rounded up to bar boundary
    last_tick_end = int(notes["tick"].iloc[-1] + notes["dur_ticks"].iloc[-1])
    n_ticks = ((last_tick_end + TICKS_PER_BAR - 1) // TICKS_PER_BAR) * TICKS_PER_BAR
    if n_ticks < TICKS_PER_BAR:
        continue

    tokens = np.full(n_ticks, TOK_REST, dtype=np.int16)
    for tick, dur_ticks, pitch in zip(
            notes["tick"].to_numpy(),
            notes["dur_ticks"].to_numpy(),
            notes["pitch"].to_numpy()):
        t = int(tick)
        d = int(dur_ticks)
        if t < 0 or t >= n_ticks:
            continue
        # New note onset (overwrites any prior HOLD or REST at this tick)
        tokens[t] = TOK_NOTE_BASE + (int(pitch) - PITCH_LOW)
        # Fill following ticks with HOLD, but stop if we hit another note
        for k in range(1, d):
            tt = t + k
            if tt >= n_ticks:
                break
            if tokens[tt] != TOK_REST:
                break
            tokens[tt] = TOK_HOLD

    # Chord per tick: chord at beat_idx = tick // TICKS_PER_BEAT, adjusted for t0
    bdict = beats_by_solo.get(melid, {})
    if not bdict:
        continue
    beat0 = t0 // TICKS_PER_BEAT
    chord_per_tick = []
    last_chord = None
    for tk in range(n_ticks):
        bidx = beat0 + (tk // TICKS_PER_BEAT)
        c = bdict.get(bidx, last_chord)
        if c is not None:
            last_chord = c
        chord_per_tick.append(last_chord if last_chord else "NC")

    bar_pos = np.arange(n_ticks, dtype=np.int8) % TICKS_PER_BAR

    df_solo = pd.DataFrame({
        "melid":   np.full(n_ticks, melid, dtype=np.int32),
        "tick":    np.arange(n_ticks, dtype=np.int32),
        "token":   tokens.astype(np.int16),
        "chord":   chord_per_tick,
        "bar_pos": bar_pos.astype(np.int8),
    })
    records.append(df_solo)
    n_total_ticks += n_ticks
    n_kept_solos += 1

print(f"[7] Built grids for {n_kept_solos} solos, {n_total_ticks} ticks total "
      f"({n_total_ticks * (60 / 80 / TICKS_PER_BEAT) / 60:.1f} minutes at 80bpm)")

big = pd.concat(records, ignore_index=True)
big["chord"] = big["chord"].astype("string")


# ----------------------------------------------------------------------
# 8. Chord vocab: chords seen >= CHORD_MIN_COUNT ticks; rest -> 'UNK'
# ----------------------------------------------------------------------
chord_counts = Counter(big["chord"].tolist())
common = {c for c, n in chord_counts.items() if n >= CHORD_MIN_COUNT and c != "UNK"}
big["chord"] = big["chord"].where(big["chord"].isin(common), "UNK")

chord_vocab = sorted(common) + ["UNK"]
print(f"[8] Chord vocab: {len(chord_vocab)} chords "
      f"(threshold = {CHORD_MIN_COUNT} ticks)")


# ----------------------------------------------------------------------
# 9. Save outputs
# ----------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

parquet_path = os.path.join(OUT_DIR, "grid_solos.parquet")
big.to_parquet(parquet_path, index=False)
print(f"[9a] Wrote {parquet_path}  ({os.path.getsize(parquet_path) / 1024:.0f} KB)")

with open(os.path.join(OUT_DIR, "chord_vocab.json"), "w") as f:
    json.dump(chord_vocab, f, indent=2)
print(f"[9b] Wrote chord_vocab.json")

with open(os.path.join(OUT_DIR, "token_vocab.json"), "w") as f:
    json.dump(TOKEN_NAMES, f, indent=2)
print(f"[9c] Wrote token_vocab.json")

meta = {
    "n_solos":          int(big["melid"].nunique()),
    "n_ticks_total":    int(len(big)),
    "n_chords":         len(chord_vocab),
    "pitch_low":        PITCH_LOW,
    "pitch_high":       PITCH_HIGH,
    "ticks_per_beat":   TICKS_PER_BEAT,
    "ticks_per_bar":    TICKS_PER_BAR,
    "token_vocab_size": TOKEN_VOCAB_SIZE,
    "tok_rest":         TOK_REST,
    "tok_hold":         TOK_HOLD,
    "tok_note_base":    TOK_NOTE_BASE,
    "chord_min_count":  CHORD_MIN_COUNT,
}
with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"[9d] Wrote metadata.json")


# ----------------------------------------------------------------------
# 10. Sanity checks
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("SANITY CHECKS")
print("=" * 60)
tok_counts = big["token"].value_counts().sort_index()
n = len(big)
n_rest = int(tok_counts.get(TOK_REST, 0))
n_hold = int(tok_counts.get(TOK_HOLD, 0))
n_note = n - n_rest - n_hold
print(f"  ticks total : {n}")
print(f"  REST share  : {n_rest/n*100:5.1f}%   ({n_rest})")
print(f"  HOLD share  : {n_hold/n*100:5.1f}%   ({n_hold})")
print(f"  NOTE share  : {n_note/n*100:5.1f}%   ({n_note})")
print()
print("  Top 5 pitches in NOTE_x tokens:")
note_only = big[big["token"] >= TOK_NOTE_BASE]
pitch_counts = (note_only["token"] - TOK_NOTE_BASE + PITCH_LOW).value_counts().head(5)
for p, c in pitch_counts.items():
    print(f"    MIDI {int(p):3d} ({PITCH_NAMES[int(p) % 12]:2s}{int(p)//12 - 1}): {c}")
print()
print("  Top 10 chords:")
for c, n in big["chord"].value_counts().head(10).items():
    print(f"    {c:>10s} : {n}")
print()
print("  Solo length distribution (ticks):")
solo_lens = big.groupby("melid").size()
print(f"    min/median/max : {solo_lens.min()} / {int(solo_lens.median())} / {solo_lens.max()}")
print(f"    seconds @ 80bpm: "
      f"{solo_lens.min()*60/80/4:.1f} / "
      f"{int(solo_lens.median())*60/80/4:.1f} / "
      f"{solo_lens.max()*60/80/4:.1f}")
print()
print("  Pitch-class histogram on dominant 7 chords (should look bluesy):")
for chord_name in ["C7", "G7", "F7", "Cj7", "C-7"]:
    if chord_name not in chord_vocab:
        continue
    sub = note_only[big.loc[note_only.index, "chord"] == chord_name]
    pcs = ((sub["token"] - TOK_NOTE_BASE + PITCH_LOW) % 12).value_counts().sort_index()
    top = pcs.sort_values(ascending=False).head(5).index.tolist()
    print(f"    {chord_name:>5s} -> {[PITCH_NAMES[int(p)] for p in top]}")
