"""The JazzLSTM next-token prediction model and its loader.

Token vocab (REST=0, HOLD=1, NOTE_<midi> for midi in [40, 96]) is defined
here as constants so engine.py, train.py and data_prep.py can all share it.

At inference time `load_model()` can apply dynamic int8 quantization
(`quantize=True` — the big speedup on ARM/Pi) and/or TorchScript JIT
(`jit=True`). Both are safe no-ops at quality level for a model this small.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------
# Token vocab — shared across the whole ML/ stack
# ---------------------------------------------------------------
TOK_REST       = 0
TOK_HOLD       = 1
TOK_NOTE_BASE  = 2
PITCH_LOW      = 40
PITCH_HIGH     = 96
N_PITCHES      = PITCH_HIGH - PITCH_LOW + 1
VOCAB_SIZE     = 2 + N_PITCHES        # 59

# 16th-note grid in 4/4
TICKS_PER_BEAT = 4
TICKS_PER_BAR  = 16


# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------

class JazzLSTM(nn.Module):
    """At each tick:
        input  : (prev_token, chord_root, chord_quality, bar_position)
        output : logits over the 59-token vocab
    """

    def __init__(self, n_qualities, hidden=256, n_layers=2, dropout=0.1,
                 d_token=64, d_root=8, d_qual=16, d_bar=8):
        super().__init__()
        self.vocab_size = VOCAB_SIZE
        self.token_embed = nn.Embedding(VOCAB_SIZE, d_token)
        self.root_embed  = nn.Embedding(13, d_root)          # 12 + UNK
        self.qual_embed  = nn.Embedding(n_qualities, d_qual)
        self.bar_embed   = nn.Embedding(TICKS_PER_BAR, d_bar)
        self.lstm = nn.LSTM(
            input_size=d_token + d_root + d_qual + d_bar,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, VOCAB_SIZE)

    def forward(self, prev_token, chord_root, chord_qual, bar_pos, hidden=None):
        x = torch.cat([
            self.token_embed(prev_token),
            self.root_embed(chord_root),
            self.qual_embed(chord_qual),
            self.bar_embed(bar_pos),
        ], dim=-1)
        out, hidden = self.lstm(x, hidden)
        return self.head(out), hidden


# ---------------------------------------------------------------
# Loading
# ---------------------------------------------------------------

def load_model(checkpoint_path, device="cpu", quantize=True, jit=False):
    """Build a JazzLSTM matching a checkpoint, load weights, set eval mode.

    Args:
        checkpoint_path : path to a .pt produced by train.py
        device          : 'cpu' or 'cuda'
        quantize        : if True, replace nn.LSTM + nn.Linear with dynamic
                          int8 versions. ~2-3x faster on ARM (Pi 4B). The
                          model is *much* faster but the same accuracy
                          (tested to within rounding on our checkpoint).
        jit             : if True, additionally compile with TorchScript.
                          Removes Python dispatch overhead per forward.
                          Small extra speedup; stacks with quantize.

    Returns:
        (model, ckpt) — model in eval mode, ckpt dict with hyperparams.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = JazzLSTM(
        n_qualities=ckpt["n_qualities"],
        hidden=ckpt["hidden"],
        n_layers=ckpt["n_layers"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if quantize:
        model = torch.ao.quantization.quantize_dynamic(
            model, {nn.LSTM, nn.Linear}, dtype=torch.qint8,
        )
    if jit:
        model = torch.jit.script(model)

    return model, ckpt
