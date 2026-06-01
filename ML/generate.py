"""
Generate a solo from a trained JazzLSTM and write it to a .mid file.

The model was trained on data transposed to C-tonic, so the chord progression
you pass in should be expressed *as if the song were in C*.  Examples:
    ii-V-I in C major : Dm7, G7, Cmaj7
    minor blues       : C-7, F-7, C-7, G7
    rhythm changes A  : C, A7, D-7, G7

CLI:
    python ML/generate.py --chords "Dm7,G7,Cj7,Cj7" --bars 16 --temperature 1.0
"""

import os
import json
import struct
import argparse

import numpy as np
import torch

from model import (
    JazzLSTM,
    parse_chord,
    UNK_ROOT_ID,
)


# Token vocab (must match process_to_grid.py)
TOK_REST = 0
TOK_HOLD = 1
TOK_NOTE_BASE = 2
PITCH_LOW = 40
TICKS_PER_BEAT = 4
TICKS_PER_BAR = 16


# ----------------------------------------------------------------------
# Chord -> (root, qual_id) using the same parsing as training
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
# Build per-tick chord/bar arrays from a chord progression
# ----------------------------------------------------------------------
def build_track(chord_progression, n_bars, qual_to_id):
    """
    chord_progression: list of (chord_str, n_bars_for_this_chord)
    n_bars           : total bars to fill (we cycle the progression)
    """
    bar_chords = []
    for chord_str, nb in chord_progression:
        for _ in range(nb):
            bar_chords.append(chord_str)
    cycle = list(bar_chords)
    while len(bar_chords) < n_bars:
        bar_chords += cycle
    bar_chords = bar_chords[:n_bars]

    n_total_ticks = n_bars * TICKS_PER_BAR
    chord_root = np.zeros(n_total_ticks, dtype=np.int64)
    chord_qual = np.zeros(n_total_ticks, dtype=np.int64)
    for bar_idx, c in enumerate(bar_chords):
        r, q = chord_to_root_qual(c, qual_to_id)
        t0 = bar_idx * TICKS_PER_BAR
        chord_root[t0:t0 + TICKS_PER_BAR] = r
        chord_qual[t0:t0 + TICKS_PER_BAR] = q
    return chord_root, chord_qual, bar_chords


# ----------------------------------------------------------------------
# Autoregressive sampling
# ----------------------------------------------------------------------
@torch.no_grad()
def generate_tokens(model, chord_root, chord_qual, temperature=1.0, seed=0):
    device = next(model.parameters()).device
    n = len(chord_root)
    bar_pos = np.arange(n, dtype=np.int64) % TICKS_PER_BAR

    rng = np.random.default_rng(seed)
    tokens = np.zeros(n, dtype=np.int64)
    prev_token = TOK_REST
    hidden = None

    for t in range(n):
        pt = torch.tensor([[prev_token]],         dtype=torch.long, device=device)
        cr = torch.tensor([[int(chord_root[t])]], dtype=torch.long, device=device)
        cq = torch.tensor([[int(chord_qual[t])]], dtype=torch.long, device=device)
        bp = torch.tensor([[int(bar_pos[t])]],    dtype=torch.long, device=device)
        logits, hidden = model(pt, cr, cq, bp, hidden)
        l = logits[0, 0].cpu().numpy().astype(np.float64) / max(temperature, 1e-6)
        l -= l.max()
        p = np.exp(l)
        p /= p.sum()
        nxt = int(rng.choice(len(p), p=p))
        tokens[t] = nxt
        prev_token = nxt
    return tokens, bar_pos


# ----------------------------------------------------------------------
# Token stream -> (onset_tick, midi, duration_tick) events
# ----------------------------------------------------------------------
def tokens_to_events(tokens):
    events = []
    i, n = 0, len(tokens)
    while i < n:
        tok = int(tokens[i])
        if tok >= TOK_NOTE_BASE:
            midi = PITCH_LOW + (tok - TOK_NOTE_BASE)
            dur = 1
            j = i + 1
            while j < n and tokens[j] == TOK_HOLD:
                dur += 1
                j += 1
            events.append((i, midi, dur))
            i = j
        else:
            i += 1
    return events


# ----------------------------------------------------------------------
# Chord voicing for the backing track
# ----------------------------------------------------------------------
CHORD_INTERVALS = {
    "":     [0, 4, 7],
    "6":    [0, 4, 7, 9],
    "j":    [0, 4, 7, 11],
    "j7":   [0, 4, 7, 11],
    "j9":   [0, 4, 7, 11, 14],
    "7":    [0, 4, 7, 10],
    "9":    [0, 4, 7, 10, 14],
    "13":   [0, 4, 7, 10, 14, 21],
    "7b9":  [0, 4, 7, 10, 13],
    "7#9":  [0, 4, 7, 10, 15],
    "7b5":  [0, 4, 6, 10],
    "7#5":  [0, 4, 8, 10],
    "-":    [0, 3, 7],
    "-6":   [0, 3, 7, 9],
    "-7":   [0, 3, 7, 10],
    "-9":   [0, 3, 7, 10, 14],
    "-j7":  [0, 3, 7, 11],
    "-7b5": [0, 3, 6, 10],
    "o":    [0, 3, 6],
    "o7":   [0, 3, 6, 9],
    "+":    [0, 4, 8],
    "+7":   [0, 4, 8, 10],
    "sus":  [0, 5, 7],
    "sus4": [0, 5, 7],
    "sus7": [0, 5, 7, 10],
    "sus2": [0, 2, 7],
}

<<<<<<< Updated upstream
            notes.append((midi, beats, chord_id_pg))
            elapsed_time += beats
            prev_pitch_class = pitch_class
            prev_dur = dur
            prev_midi = midi
    output = []
    for midi , beats, chord_id_pg in notes:
        output.append((int(midi), float(beats), chord_progression[chord_id_pg][0]))
    return output
            
BPM = 80
iiVI_cycle = [("Cm7", 4), ("F7", 4), ("Bbmaj7", 4), ("Bbmaj7", 4)]
iiVI_long = iiVI_cycle * 5
notes = generate_music(LSTMmodel, mod.chord_to_id, iiVI_long, temperature=0.75)
print(len(notes))
    
=======
>>>>>>> Stashed changes

def chord_to_voicing(chord_string, bass_min=36, upper_min=48):
    p = parse_chord(chord_string)
    if p is None:
        return []
    root_pc, quality = p
    intervals = CHORD_INTERVALS.get(quality)
    if intervals is None:
        for prefix in ("-7b5", "-j", "-7", "-6", "-9", "-",
                       "j", "o7", "o", "+7", "+", "sus"):
            if quality.startswith(prefix):
                intervals = CHORD_INTERVALS[prefix]
                break
        if intervals is None:
            intervals = CHORD_INTERVALS["7" if "7" in quality else ""]
    bass = bass_min + ((root_pc - bass_min) % 12)
    uppers = []
    for iv in intervals:
        pc = (root_pc + iv) % 12
        m = upper_min + ((pc - upper_min) % 12)
        uppers.append(m)
    return [bass] + sorted(set(uppers))


# ----------------------------------------------------------------------
# MIDI writing (proper delta times -> rests work)
# ----------------------------------------------------------------------
def _vlq(n):
    out = [n & 0x7F]; n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80); n >>= 7
    return bytes(reversed(out))


def _make_track(b):
    return b"MTrk" + struct.pack(">I", len(b)) + bytes(b)


def write_midi(events, bar_chords, out_path, tempo_bpm=80,
               ticks_per_quarter=480,
               melody_velocity=92, chord_velocity=50):
    grid = ticks_per_quarter // TICKS_PER_BEAT
    bar_mt = TICKS_PER_BAR * grid

    # tempo track
    micros = int(60_000_000 / tempo_bpm)
    tempo_evt = bytearray()
    tempo_evt += _vlq(0) + b"\xff\x51\x03" + micros.to_bytes(3, "big")
    tempo_evt += _vlq(0) + b"\xff\x2f\x00"

    # melody track (rests via delta times)
    mel_evt = bytearray()
    last_off = 0
    for onset_g, midi, dur_g in events:
        onset_mt = onset_g * grid
        dur_mt   = max(1, dur_g * grid)
        delta    = onset_mt - last_off
        mel_evt += _vlq(delta) + bytes([0x90, midi, melody_velocity])
        mel_evt += _vlq(dur_mt) + bytes([0x80, midi, 0])
        last_off = onset_mt + dur_mt
    mel_evt += _vlq(0) + b"\xff\x2f\x00"

    # chord backing track (channel 1)
    chord_evt = bytearray()
    for chord_str in bar_chords:
        voicing = chord_to_voicing(chord_str)
        if not voicing:
            chord_evt += _vlq(bar_mt) + bytes([0xb1, 0, 0])
            continue
        for p in voicing:
            chord_evt += _vlq(0) + bytes([0x91, p, chord_velocity])
        for i, p in enumerate(voicing):
            d = bar_mt if i == 0 else 0
            chord_evt += _vlq(d) + bytes([0x81, p, 0])
    chord_evt += _vlq(0) + b"\xff\x2f\x00"

    header = b"MThd" + struct.pack(">IHHH", 6, 1, 3, ticks_per_quarter)
    with open(out_path, "wb") as f:
        f.write(header + _make_track(tempo_evt)
                       + _make_track(mel_evt)
                       + _make_track(chord_evt))


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ML/checkpoints/jazz_lstm.pt")
    ap.add_argument("--chords",     default="Dm7,G7,Cj7,Cj7",
                    help="comma-separated chord progression (in C-relative key)")
    ap.add_argument("--chord_bars", type=int, default=1, help="bars per chord")
    ap.add_argument("--bars",       type=int, default=16, help="total bars to render")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tempo_bpm",  type=int, default=80)
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--out",        default="generated_solo.mid")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    with open("ML/data/processed/chord_vocab.json") as f:
        chord_vocab = json.load(f)
    qual_to_id = build_qual_map(chord_vocab)

    model = JazzLSTM(
        vocab_size=ckpt["vocab_size"],
        n_qualities=ckpt["n_qualities"],
        hidden=ckpt["hidden"],
        n_layers=ckpt["n_layers"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prog = [(c.strip(), args.chord_bars) for c in args.chords.split(",") if c.strip()]
    chord_root, chord_qual, bar_chords = build_track(prog, args.bars, qual_to_id)

    print(f"checkpoint   : {args.checkpoint} (epoch {ckpt.get('epoch','?')})")
    print(f"progression  : {[c for c, _ in prog]}  x {args.chord_bars} bar(s) each")
    print(f"total bars   : {args.bars}  ({args.bars * TICKS_PER_BAR} ticks, "
          f"{args.bars * 4 * 60 / args.tempo_bpm:.1f}s at {args.tempo_bpm}bpm)")
    print(f"temperature  : {args.temperature}   seed: {args.seed}")
    print()

    tokens, _ = generate_tokens(
        model, chord_root, chord_qual,
        temperature=args.temperature, seed=args.seed)

    n = len(tokens)
    n_rest = int((tokens == TOK_REST).sum())
    n_hold = int((tokens == TOK_HOLD).sum())
    n_note = n - n_rest - n_hold
    print("generated token distribution:")
    print(f"  REST: {n_rest:>5d} ({100*n_rest/n:5.1f}%)")
    print(f"  HOLD: {n_hold:>5d} ({100*n_hold/n:5.1f}%)")
    print(f"  NOTE: {n_note:>5d} ({100*n_note/n:5.1f}%)")

    events = tokens_to_events(tokens)
    if events:
        midi_vals = [e[1] for e in events]
        durs      = [e[2] for e in events]
        print(f"\nnote events  : {len(events)}")
        print(f"  pitch range: MIDI {min(midi_vals)} .. {max(midi_vals)}")
        print(f"  duration   : min {min(durs)} / "
              f"median {sorted(durs)[len(durs)//2]} / max {max(durs)} ticks")
    else:
        print("\nWARNING: no NOTE events generated. Try a different seed or temperature.")

    write_midi(events, bar_chords, args.out, tempo_bpm=args.tempo_bpm)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
