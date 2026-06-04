"""Chord-string parsing and chord-vocabulary lookups.

Shared by `data_prep.py` (used during training-data preparation), `engine.py`
(used at inference to resolve user-typed chord strings), and `generate.py`
(used by the offline CLI).

The vocabulary the model trains on stores chords as strings like "Dm7",
"G7", "Cj7" (Jazz-Real-Book style). Each string is parsed into:

    (root_pitch_class, normalized_quality)
        root_pitch_class : int in 0..11 (C, C#, D, …) or 12 = unknown
        normalized_quality : short string like "-7", "j7", "7b9", "o", …

The model takes (root, quality_id) as separate embeddings — splitting it
this way lets the model share knowledge across all chords with the same
quality (every "m7" behaves similarly relative to its root) and shrinks the
vocab from ~300 chord strings to ~80 qualities × 12 roots.
"""

NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
UNK_ROOT_ID = 12       # reserved root id for unparseable chord strings


# ---------------------------------------------------------------
# Parsing primitives
# ---------------------------------------------------------------

def parse_root(token):
    """Split a chord head into (root_pc, remainder_string)."""
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
    """Canonicalize chord-quality strings (maj7 → j7, min → -, etc.)."""
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
    """Chord string → (root_pc, normalized_quality), or None if unparseable."""
    s = (s or "").strip()
    if not s or s in ("UNK", "NC"):
        return None
    p = parse_root(s)
    if p is None:
        return None
    return p[0], normalize_quality(p[1])


def parse_key(s):
    """Tonic of a solo's key string (e.g. 'Bb-maj' → 10)."""
    if not s or not str(s).strip():
        return None
    head = str(s).split("-")[0].strip()
    p = parse_root(head)
    return None if p is None else p[0]


def transpose_chord(chord_string, semitones):
    """Shift a chord string by `semitones`. Returns unchanged if unparseable."""
    p = parse_chord(chord_string)
    if p is None:
        return chord_string
    pc, q = p
    return PITCH_NAMES[(pc + semitones) % 12] + q


# ---------------------------------------------------------------
# Vocab lookups
# ---------------------------------------------------------------

def build_chord_tables(chord_vocab):
    """From a list of chord strings, build the lookup tables the model needs.

    Returns:
        chord_to_root : dict[str -> int]   chord_string -> root id (0..12)
        chord_to_qual : dict[str -> int]   chord_string -> quality id
        n_qualities   : int                size of the quality vocab (incl UNK)
        qual_to_id    : dict[str -> int]   quality_string -> quality id
    """
    qualities = set()
    for c in chord_vocab:
        p = parse_chord(c)
        if p is not None:
            qualities.add(p[1])
    qualities = sorted(qualities) + ["__UNK_Q__"]
    qual_to_id = {q: i for i, q in enumerate(qualities)}

    chord_to_root, chord_to_qual = {}, {}
    for c in chord_vocab:
        p = parse_chord(c)
        if p is None:
            chord_to_root[c] = UNK_ROOT_ID
            chord_to_qual[c] = qual_to_id["__UNK_Q__"]
        else:
            chord_to_root[c] = p[0]
            chord_to_qual[c] = qual_to_id[p[1]]
    return chord_to_root, chord_to_qual, len(qualities), qual_to_id


def chord_str_to_ids(chord_str, qual_to_id):
    """Resolve any chord string (including ones not seen during training)
    to (root_id, quality_id) for the embeddings, with UNK fallback."""
    unk_qid = qual_to_id["__UNK_Q__"]
    p = parse_chord(chord_str)
    if p is None:
        return UNK_ROOT_ID, unk_qid
    root, q = p
    return root, qual_to_id.get(q, unk_qid)
