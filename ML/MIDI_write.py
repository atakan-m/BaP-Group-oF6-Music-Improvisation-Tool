import struct
import generate as gen

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

CHORD_INTERVALS = {
    "":     [0, 4, 7],            # major triad
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


def normalize_quality(q):
    if not q:
        return ""
    replacements = [
        ("maj7", "j7"), ("Maj7", "j7"), ("MAJ7", "j7"), ("M7", "j7"),
        ("Δ", "j"), ("maj", "j"), ("Maj", "j"),
        ("min7", "-7"), ("Min7", "-7"), ("min", "-"), ("Min", "-"),
        ("dim", "o"),
        ("aug", "+"),
    ]
    for a, b in replacements:
        q = q.replace(a, b)
    if q.startswith("m") and (len(q) == 1 or not q[1].isalpha()):
        q = "-" + q[1:]
    return q


def parse_chord(s):
    s = (s or "").strip()
    if not s:
        return None
    parsed = parse_root(s)
    if parsed is None:
        return None
    pc, rest = parsed
    return pc, normalize_quality(rest)

def chord_intervals(quality):
    if quality in CHORD_INTERVALS:
        return CHORD_INTERVALS[quality]
    # heuristic fallbacks
    for prefix in ("-7b5", "-j", "-7", "-6", "-9", "-", "j", "o7", "o", "+7", "+", "sus"):
        if quality.startswith(prefix):
            return CHORD_INTERVALS[prefix]
    if "7" in quality:
        return CHORD_INTERVALS["7"]
    return CHORD_INTERVALS[""]


def chord_to_voicing(chord_string, bass_min=36, upper_min=48):
    """Return [bass_midi, upper_midi_1, ...]. Bass note in low octave,
    chord tones placed in the lowest octave at-or-above upper_min."""
    parsed = parse_chord(chord_string)
    if parsed is None:
        return []
    root_pc, quality = parsed
    intervals = chord_intervals(quality)
    bass = bass_min + ((root_pc - bass_min) % 12)
    uppers = []
    for iv in intervals:
        pc = (root_pc + iv) % 12
        m = upper_min + ((pc - upper_min) % 12)
        uppers.append(m)
    return [bass] + sorted(set(uppers))

def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def _make_track(events_bytes):
    return b"MTrk" + struct.pack(">I", len(events_bytes)) + bytes(events_bytes)






def write_midi(notes, progression, out_path,
               ticks_per_quarter=480, tempo_bpm=140,
               melody_velocity=92, chord_velocity=58,
               bass_min=36, upper_min=48):
    """notes: [(midi, beats, chord_label)]  -- the melody
       progression: [(chord_str, beats)]     -- for the chord comping track
    """
    # Track 1: tempo only
    micros = int(60_000_000 / tempo_bpm)
    tempo_evt = bytearray()
    tempo_evt += _vlq(0) + b"\xff\x51\x03" + micros.to_bytes(3, "big")
    tempo_evt += _vlq(0) + b"\xff\x2f\x00"

    # Track 2: melody on channel 0
    mel_evt = bytearray()
    for pitch, beats, _ in notes:
        ticks = max(1, int(round(beats * ticks_per_quarter)))
        mel_evt += _vlq(0) + bytes([0x90, int(pitch), melody_velocity])
        mel_evt += _vlq(ticks) + bytes([0x80, int(pitch), 0])
    mel_evt += _vlq(0) + b"\xff\x2f\x00"

    # Track 3: chord comping on channel 1
    chord_evt = bytearray()
    for chord_string, beats in progression:
        ticks = max(1, int(round(beats * ticks_per_quarter)))
        voicing = chord_to_voicing(chord_string, bass_min=bass_min, upper_min=upper_min)
        if not voicing:
            chord_evt += _vlq(ticks) + bytes([0xb1, 0, 0])  # silent CC
            continue
        for p in voicing:
            chord_evt += _vlq(0) + bytes([0x91, p, chord_velocity])
        for i, p in enumerate(voicing):
            d = ticks if i == 0 else 0
            chord_evt += _vlq(d) + bytes([0x81, p, 0])
    chord_evt += _vlq(0) + b"\xff\x2f\x00"

    header = b"MThd" + struct.pack(">IHHH", 6, 1, 3, ticks_per_quarter)
    with open(out_path, "wb") as f:
        f.write(header + _make_track(tempo_evt) + _make_track(mel_evt) + _make_track(chord_evt))



print("chords:")
chord_progression = [(str(x),4) for x in input().split(",")]
print(chord_progression)
iiVI_long = chord_progression * 5
notes = gen.generate_music(gen.LSTMmodel, gen.chord_to_id, iiVI_long, temperature=0.3)
write_midi(notes, iiVI_long, "midi_gen_model2_cmajtemp03.mid", tempo_bpm=80)
