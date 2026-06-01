"""
JazzLSTM: per-16th-note next-token prediction model.

At each tick we predict the *current* token given:
    - the token at the previous tick     (one of REST | HOLD | NOTE_<midi>)
    - the chord active at this tick      (split into root pc + quality id)
    - the position in the current bar    (0..15)

Token vocabulary (matches process_to_grid.py):
    0 = REST
    1 = HOLD (continuation of previous note)
    2..58 = NOTE_<midi>  for midi in [40, 96]
"""

import torch
import torch.nn as nn


# ------------------------------------------------------------------
# Chord parsing helpers
# (kept in sync with process_to_grid.py; small duplication is fine)
# ------------------------------------------------------------------
NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
UNK_ROOT_ID = 12          # we use index 12 for "no/unknown root"


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
    if not s or s in ("UNK", "NC"):
        return None
    parsed = parse_root(s)
    if parsed is None:
        return None
    pc, rest = parsed
    return pc, normalize_quality(rest)


def build_chord_tables(chord_vocab):
    """
    Convert a list of chord strings -> (root_pc, quality_id) lookups.

    Returns:
        chord_to_root: dict[str -> int in 0..12]   (12 = UNK)
        chord_to_qual: dict[str -> int in 0..Q-1]
        n_qualities:   int                          (includes an UNK quality)
    """
    qualities = set()
    for c in chord_vocab:
        parsed = parse_chord(c)
        if parsed is not None:
            qualities.add(parsed[1])
    qualities = sorted(qualities) + ["__UNK_Q__"]
    qual_to_id = {q: i for i, q in enumerate(qualities)}

    chord_to_root, chord_to_qual = {}, {}
    for c in chord_vocab:
        parsed = parse_chord(c)
        if parsed is None:
            chord_to_root[c] = UNK_ROOT_ID
            chord_to_qual[c] = qual_to_id["__UNK_Q__"]
        else:
            chord_to_root[c] = parsed[0]
            chord_to_qual[c] = qual_to_id[parsed[1]]
    return chord_to_root, chord_to_qual, len(qualities)


# ------------------------------------------------------------------
# The model
# ------------------------------------------------------------------
class JazzLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_qualities,
        d_token=64,
        d_root=8,
        d_qual=16,
        d_bar=8,
        hidden=256,
        n_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.n_layers = n_layers

        self.token_embed = nn.Embedding(vocab_size, d_token)
        self.root_embed = nn.Embedding(13, d_root)            # 12 roots + UNK
        self.qual_embed = nn.Embedding(n_qualities, d_qual)
        self.bar_embed = nn.Embedding(16, d_bar)              # 16 ticks per 4/4 bar

        in_dim = d_token + d_root + d_qual + d_bar
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, prev_token, chord_root, chord_qual, bar_pos, hidden=None):
        """
        All inputs are long tensors of shape (B, T).

            prev_token: token at the previous tick
            chord_root: chord root pc at the current tick (predict-tick)
            chord_qual: chord quality id at the current tick
            bar_pos:    0..15 at the current tick
            hidden:     optional initial (h, c) for stateful inference

        Returns:
            logits  (B, T, vocab_size)
            hidden  ((n_layers, B, hidden), (n_layers, B, hidden))
        """
        x = torch.cat(
            [
                self.token_embed(prev_token),
                self.root_embed(chord_root),
                self.qual_embed(chord_qual),
                self.bar_embed(bar_pos),
            ],
            dim=-1,
        )
        out, hidden = self.lstm(x, hidden)
        return self.head(out), hidden
