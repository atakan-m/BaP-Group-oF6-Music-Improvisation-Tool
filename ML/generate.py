"""Offline-MIDI CLI + a few re-exports for old import paths.

This module exists for two reasons:

1. **Offline MIDI generation** (the `main()` CLI below) — for testing the
   model without booting Piano_display. Renders a solo + chord-comping
   track to a Standard MIDI File you can open in any DAW.

2. **Re-exports from `engine.py`** so older scripts that did
   `from generate import …` still work. Anything new should import
   directly from `engine` and `model`.

The old static-sheet `generate_music(LSTMmodel, chord_to_id, …)` API has
been removed — Piano_display now uses `realtime_sheet.RealtimeSheet`
which drives the engine in realtime. Even when the player isn't playing,
the engine treats silence as "follow the plan" and self-improvises, so
there's no need for a separate offline generator inside Piano_display.
"""

# ----- Re-exports kept for older import paths --------------------------
from engine import (
    JazzImprov,
    MicTickAggregator,
    pitch_to_note_token,
    token_to_pitch,
    make_engine as make_realtime_engine,
)
from model import (
    TOK_REST, TOK_HOLD, TOK_NOTE_BASE,
    PITCH_LOW, PITCH_HIGH, VOCAB_SIZE,
    TICKS_PER_BAR, TICKS_PER_BEAT,
)


# ======================================================================
# Offline MIDI rendering — used by the CLI
# ======================================================================

def _tokens_to_events(tokens):
    """Token sequence -> list of (onset_tick, midi_pitch, duration_ticks)."""
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
            events.append((i, midi, dur))
            i = j
        else:
            i += 1
    return events


_CHORD_INTERVALS = {
    "":   [0, 4, 7],            "6":   [0, 4, 7, 9],
    "j":  [0, 4, 7, 11],        "j7":  [0, 4, 7, 11],
    "7":  [0, 4, 7, 10],        "9":   [0, 4, 7, 10, 14],
    "-":  [0, 3, 7],            "-6":  [0, 3, 7, 9],
    "-7": [0, 3, 7, 10],        "-9":  [0, 3, 7, 10, 14],
    "-j7":[0, 3, 7, 11],        "-7b5":[0, 3, 6, 10],
    "o":  [0, 3, 6],            "o7":  [0, 3, 6, 9],
    "+":  [0, 4, 8],            "+7":  [0, 4, 8, 10],
    "sus":[0, 5, 7],            "sus7":[0, 5, 7, 10],
}


def _chord_voicing(chord_str, bass_min=36, upper_min=48):
    from chord_utils import parse_chord
    p = parse_chord(chord_str)
    if p is None:
        return []
    root, quality = p
    iv = _CHORD_INTERVALS.get(quality)
    if iv is None:
        for prefix in ("-7b5", "-j", "-7", "-6", "-9", "-",
                       "j", "o7", "o", "+", "sus"):
            if quality.startswith(prefix):
                iv = _CHORD_INTERVALS[prefix]
                break
        if iv is None:
            iv = _CHORD_INTERVALS["7" if "7" in quality else ""]
    bass = bass_min + ((root - bass_min) % 12)
    uppers = sorted({upper_min + (((root + k) - upper_min) % 12) for k in iv})
    return [bass] + uppers


def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def _make_track(payload):
    import struct
    return b"MTrk" + struct.pack(">I", len(payload)) + bytes(payload)


def write_midi(tokens, bar_chords, out_path, tempo_bpm=80, ticks_per_quarter=480):
    """Render a token sequence + chord progression into a Standard MIDI File.
    Three tracks: tempo, melody (channel 0), chord comping (channel 1)."""
    import struct
    grid = ticks_per_quarter // TICKS_PER_BEAT     # MIDI ticks per 16th
    bar_mt = TICKS_PER_BAR * grid

    micros = int(60_000_000 / tempo_bpm)
    tempo = bytearray()
    tempo += _vlq(0) + b"\xff\x51\x03" + micros.to_bytes(3, "big")
    tempo += _vlq(0) + b"\xff\x2f\x00"

    mel = bytearray()
    last_off = 0
    for onset, midi, dur in _tokens_to_events(tokens):
        onset_mt = onset * grid
        dur_mt   = max(1, dur * grid)
        mel += _vlq(onset_mt - last_off) + bytes([0x90, midi, 92])
        mel += _vlq(dur_mt) + bytes([0x80, midi, 0])
        last_off = onset_mt + dur_mt
    mel += _vlq(0) + b"\xff\x2f\x00"

    chord = bytearray()
    for c in bar_chords:
        v = _chord_voicing(c)
        if not v:
            chord += _vlq(bar_mt) + bytes([0xb1, 0, 0])
            continue
        for p in v:
            chord += _vlq(0) + bytes([0x91, p, 50])
        for i, p in enumerate(v):
            chord += _vlq(bar_mt if i == 0 else 0) + bytes([0x81, p, 0])
    chord += _vlq(0) + b"\xff\x2f\x00"

    header = b"MThd" + struct.pack(">IHHH", 6, 1, 3, ticks_per_quarter)
    with open(out_path, "wb") as f:
        f.write(header + _make_track(tempo) + _make_track(mel) + _make_track(chord))


# ======================================================================
# CLI: `python ML/generate.py --chords "..." --bars N --out song.mid`
# ======================================================================

def main():
    """Drive the engine into silence (REST is always treated as match) so
    it walks its own plan, then export the result as MIDI."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--chords",      default="Dm7,G7,Cj7,Cj7",
                    help="comma-separated chord progression (expressed in C)")
    ap.add_argument("--bars",        type=int,   default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--hold_penalty",type=float, default=0.5)
    ap.add_argument("--rest_penalty",type=float, default=1.5)
    ap.add_argument("--max_consec_holds", type=int, default=8)
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--tempo_bpm",   type=int,   default=80)
    ap.add_argument("--out",         default="generated_solo.mid")
    args = ap.parse_args()

    prog = [(c.strip(), 1) for c in args.chords.split(",") if c.strip()]
    eng = make_realtime_engine(
        chord_progression=prog,
        total_bars=args.bars,
        temperature=args.temperature,
        hold_penalty=args.hold_penalty,
        rest_penalty=args.rest_penalty,
        max_consec_holds=args.max_consec_holds,
        seed=args.seed,
        quantize=False,           # full fp32 — speed isn't critical offline
    )

    tokens = []
    while not eng.is_done:
        eng.commit(TOK_REST)
        tokens.append(eng.prev_actual)

    write_midi(tokens, eng.bar_chords, args.out, tempo_bpm=args.tempo_bpm)
    print(f"wrote {args.out} ({len(tokens)} ticks)")


if __name__ == "__main__":
    main()
