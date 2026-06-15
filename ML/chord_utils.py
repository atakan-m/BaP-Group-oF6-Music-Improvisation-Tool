"""Chord-string parsing and chord-vocabulary lookups.

Shared by three modules:
    * data_prep.py — used during training-data preparation
    * engine.py    — used at inference to resolve user-typed chord strings
    * generate.py  — used by the offline MIDI-rendering CLI

The vocabulary the model trains on stores chords as strings like "Dm7",
"G7", "Cj7" (Jazz-Real-Book style). Each string is parsed into:

    (root_pitch_class, normalized_quality)
        root_pitch_class   : int in 0..11 (C, C#, D, …) or 12 = unknown
        normalized_quality : short string like "-7", "j7", "7b9", "o", …

The model embeds (root, quality_id) separately. Splitting it this way lets
the model share knowledge across all chords with the same quality (every
"m7" behaves similarly relative to its root) and shrinks the vocab from
~300 raw chord strings to ~80 qualities × 12 roots.
"""

NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
UNK_ROOT_ID = 12       # reserved root id for unparseable chord strings


# ---------------------------------------------------------------
# Parsing primitives
# ---------------------------------------------------------------

def parse_root(token):
    """Split a chord head into (root_pc, remainder_string).

    Accepts a leading note letter optionally followed by a single sharp or
    flat accidental, then returns the pitch class (0..11) and the rest of
    the chord string for downstream quality parsing. Returns None if the
    string doesn't start with a valid note letter.
    """
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
    """Canonicalise chord-quality strings into the short Jazz-Real-Book form.

    Folds aliases like "maj7", "M7", "Δ7" → "j7"; "min7", "Min7" → "-7";
    "dim" → "o"; "aug" → "+"; a leading bare "m" (e.g. "m7") becomes "-7".
    Empty input returns empty string.
    """
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
    """Top-level chord parser.

    Returns (root_pc, normalized_quality) for any chord whose first character
    is a valid note letter. Returns None for blank input or the special
    sentinels "UNK" and "NC" (no chord).
    """
    s = (s or "").strip()
    if not s or s in ("UNK", "NC"):
        return None
    p = parse_root(s)
    if p is None:
        return None
    return p[0], normalize_quality(p[1])


def parse_key(s):
    """Extract the tonic pitch class from a key string.

    Takes strings like "Bb-maj", "A-min" and returns the integer pitch class
    of the tonic (10 and 9 respectively). Returns None if the leading letter
    isn't a valid note. Used by data_prep.py to compute the transposition
    shift that maps each solo to C tonic.
    """
    if not s or not str(s).strip():
        return None
    head = str(s).split("-")[0].strip()
    p = parse_root(head)
    return None if p is None else p[0]


def transpose_chord(chord_string, semitones):
    """Shift a chord by a number of semitones.

    The quality is preserved; only the root letter changes. Sharps are used
    for accidentals in the output. Returns the input string unchanged if it
    can't be parsed (so UNK / NC pass through cleanly).
    """
    p = parse_chord(chord_string)
    if p is None:
        return chord_string
    pc, q = p
    return PITCH_NAMES[(pc + semitones) % 12] + q


# ---------------------------------------------------------------
# Vocab lookups
# ---------------------------------------------------------------

def build_chord_tables(chord_vocab):
    """Build the chord lookup tables used by the model at train time.

    The chord_vocab list typically comes from chord_vocab.json (produced by
    data_prep.py). For every chord we precompute its embedding indices;
    chords that fail to parse fall back to the reserved UNK ids.

    Returns:
        chord_to_root : dict[str -> int]  chord_string -> root id (0..12)
        chord_to_qual : dict[str -> int]  chord_string -> quality id
        n_qualities   : int               size of the quality vocab (incl UNK)
        qual_to_id    : dict[str -> int]  quality_string -> quality id
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
    """Resolve a chord string to (root_id, quality_id) at inference time.

    Used by engine.set_progression to encode the user-typed chord
    progression. Out-of-vocabulary qualities and unparseable chords fall
    back to the UNK ids so the model sees a defined input either way.
    """
    unk_qid = qual_to_id["__UNK_Q__"]
    p = parse_chord(chord_str)
    if p is None:
        return UNK_ROOT_ID, unk_qid
    root, q = p
    return root, qual_to_id.get(q, unk_qid)
