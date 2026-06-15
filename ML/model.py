"""JazzLSTM next-token prediction model and its loader.

This module defines:
    * the token vocabulary (REST=0, HOLD=1, NOTE_<midi> for midi in [40, 96])
      as constants shared across engine.py, train.py and data_prep.py;
    * the JazzLSTM PyTorch model — a small autoregressive LSTM that
      predicts the next 16th-note token given (prev_token, chord_root,
      chord_quality, bar_position);
    * load_model(), which builds an empty model, loads the checkpoint
      weights, and optionally applies dynamic int8 quantisation and
      TorchScript JIT compilation for fast inference on the Raspberry Pi.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------
# Token vocab — shared across the whole ML/ stack.
# A token represents the action at one 16th-note tick:
#   0          → REST (silence)
#   1          → HOLD (sustain the previous pitch)
#   2 .. 58    → NOTE_<midi> for midi in [PITCH_LOW, PITCH_HIGH]
# ---------------------------------------------------------------
TOK_REST       = 0
TOK_HOLD       = 1
TOK_NOTE_BASE  = 2
PITCH_LOW      = 40                   # MIDI E2 — lowest playable note
PITCH_HIGH     = 96                   # MIDI C7 — highest playable note
N_PITCHES      = PITCH_HIGH - PITCH_LOW + 1
VOCAB_SIZE     = 2 + N_PITCHES        # 59 tokens total

# Rhythmic grid (4/4 time, 16th-note resolution).
TICKS_PER_BEAT = 4
TICKS_PER_BAR  = 16


# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------

class JazzLSTM(nn.Module):
    """Autoregressive next-token predictor for jazz solos.

    At each 16th-note tick the model receives four integer inputs:
        prev_token    — the token at the previous tick
        chord_root    — root pitch class (0..11, or 12 = UNK)
        chord_quality — id into the quality vocabulary
        bar_position  — 0..15, the tick's position within the bar
    Each input is embedded independently, concatenated, fed through a
    stack of LSTM layers, and projected to logits over the 59-token vocab.

    The default configuration (~0.9 M parameters) was picked to fit
    comfortably inside the Raspberry Pi 4B inference budget after int8
    dynamic quantisation.
    """

    def __init__(self, n_qualities, hidden=256, n_layers=2, dropout=0.1,
                 d_token=64, d_root=8, d_qual=16, d_bar=8):
        super().__init__()
        self.vocab_size = VOCAB_SIZE

        # Four parallel input embeddings.
        self.token_embed = nn.Embedding(VOCAB_SIZE, d_token)
        self.root_embed  = nn.Embedding(13, d_root)          # 12 roots + UNK
        self.qual_embed  = nn.Embedding(n_qualities, d_qual)
        self.bar_embed   = nn.Embedding(TICKS_PER_BAR, d_bar)

        # Stacked LSTM core. Concatenated input dimension is the sum of the
        # four embedding sizes; dropout only takes effect when n_layers > 1.
        self.lstm = nn.LSTM(
            input_size=d_token + d_root + d_qual + d_bar,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Linear projection to logits over the token vocabulary.
        self.head = nn.Linear(hidden, VOCAB_SIZE)

    def forward(self, prev_token, chord_root, chord_qual, bar_pos, hidden=None):
        """Forward pass over a (B, T) input batch.

        Returns (logits, hidden), where logits has shape (B, T, VOCAB_SIZE)
        and hidden is the LSTM (h, c) tuple for chaining incremental inference
        calls one tick at a time.
        """
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
    """Construct a JazzLSTM matching a checkpoint and load its weights.

    The architecture hyperparameters (hidden, n_layers, dropout, n_qualities)
    are read directly from the checkpoint dict so a single load call always
    matches the trained model exactly.

    Args:
        checkpoint_path : path to a .pt file produced by train.py.
        device          : torch device string ('cpu' or 'cuda').
        quantize        : if True, replace nn.LSTM and nn.Linear layers with
                          dynamic int8 versions. Yields ~2-3x speedup on ARM
                          (Raspberry Pi 4B) with no measurable accuracy loss.
        jit             : if True, additionally TorchScript-compile the model.
                          Removes Python dispatch overhead per forward; small
                          extra speedup that stacks with quantize.

    Returns:
        (model, ckpt) — the model in eval mode, and the raw checkpoint dict
        so callers can recover the training hyperparameters.
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
