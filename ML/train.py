"""Train the JazzLSTM next-token model on the 16th-note grid dataset.

The training data is consumed as random fixed-length chunks: each step
picks a random solo and a random start offset inside it, then trains on
plain next-token prediction with cross-entropy loss. Adam + cosine LR
schedule with gradient clipping.

Inputs (produced by data_prep.py):
    ML/data/processed/grid_solos.parquet
    ML/data/processed/chord_vocab.json
    ML/data/processed/metadata.json

Output:
    ML/models/<--out>     a checkpoint .pt file, overwritten at the end
                          of every epoch (so an interrupted run still
                          leaves a usable model on disk).

CLI usage (example):
    python ML/train.py --epochs 30 --batch_size 64 --seq_len 256 \
                       --out jazz_lstm.pt
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from chord_utils import build_chord_tables, UNK_ROOT_ID
from model import JazzLSTM


DATA_DIR  = os.path.join("ML", "data", "processed")
MODELS_DIR = os.path.join("ML", "models")
# Default checkpoint filename, written under ML/models/. Override per-run
# with `--out <name>.pt`. To use a freshly trained checkpoint at inference,
# point engine.py's DEFAULT_CHECKPOINT at the matching filename.
DEFAULT_CKPT_NAME = "jazz_lstm_v2.pt"


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------

def load_solos(seq_len):
    """Load the per-tick training data and prepare per-solo numpy arrays.

    Reads grid_solos.parquet, attaches the (chord_root, chord_quality)
    embedding ids from the chord vocab, and groups the data by solo so each
    solo's token / root / quality / bar_pos sequences live in contiguous
    numpy arrays for cache-friendly random sampling later.

    Solos shorter than seq_len + 1 ticks are dropped — they can't yield a
    full-length training chunk.

    Returns:
        solos       : list of dicts (one per solo) with the four arrays.
        meta        : the metadata.json dict written by data_prep.
        n_qualities : size of the chord-quality vocab (includes UNK).
    """
    df = pd.read_parquet(os.path.join(DATA_DIR, "grid_solos.parquet"))
    with open(os.path.join(DATA_DIR, "chord_vocab.json")) as f:
        chord_vocab = json.load(f)
    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        meta = json.load(f)

    chord_to_root, chord_to_qual, n_qualities, _ = build_chord_tables(chord_vocab)
    unk_qid = n_qualities - 1

    df["chord_root"] = df["chord"].map(chord_to_root).fillna(UNK_ROOT_ID).astype(np.int16)
    df["chord_qual"] = df["chord"].map(chord_to_qual).fillna(unk_qid).astype(np.int16)

    solos = []
    for _, g in df.groupby("melid", sort=False):
        if len(g) <= seq_len + 1:
            continue
        solos.append({
            "token": g["token"].to_numpy(dtype=np.int64),
            "root":  g["chord_root"].to_numpy(dtype=np.int64),
            "qual":  g["chord_qual"].to_numpy(dtype=np.int64),
            "bar":   g["bar_pos"].to_numpy(dtype=np.int64),
        })
    return solos, meta, n_qualities


def sample_batch(solos, batch_size, seq_len, rng):
    """Build one training batch of random fixed-length chunks.

    For each batch element, pick a random solo and a random start offset
    inside it, then slice out a contiguous (T = seq_len)-long window.
    The slicing is set up so the model is trained on plain next-token
    prediction:

        At position j ∈ [0, T):
            prev_token = tokens[start + j]            (input)
            chord/bar  = features at tick start+j+1   (input)
            target     = tokens[start + j + 1]        (label)

    Returns five (B, T) long tensors in this order:
        prev_token, chord_root, chord_quality, bar_position, target.
    """
    B, T = batch_size, seq_len
    pt = np.empty((B, T), dtype=np.int64)
    cr = np.empty((B, T), dtype=np.int64)
    cq = np.empty((B, T), dtype=np.int64)
    bp = np.empty((B, T), dtype=np.int64)
    tg = np.empty((B, T), dtype=np.int64)
    n_solos = len(solos)
    for i in range(B):
        s = solos[int(rng.integers(n_solos))]
        start = int(rng.integers(len(s["token"]) - T - 1))
        pt[i] = s["token"][start     : start + T]
        cr[i] = s["root"] [start + 1 : start + 1 + T]
        cq[i] = s["qual"] [start + 1 : start + 1 + T]
        bp[i] = s["bar"]  [start + 1 : start + 1 + T]
        tg[i] = s["token"][start + 1 : start + 1 + T]
    return [torch.from_numpy(a) for a in (pt, cr, cq, bp, tg)]


# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------

def main():
    """CLI entry point: parse args, build the model, run the training loop."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",          type=int,   default=30)
    ap.add_argument("--steps_per_epoch", type=int,   default=200)
    ap.add_argument("--batch_size",      type=int,   default=64)
    ap.add_argument("--seq_len",         type=int,   default=256)
    ap.add_argument("--lr",              type=float, default=2e-3)
    ap.add_argument("--weight_decay",    type=float, default=1e-5)
    ap.add_argument("--grad_clip",       type=float, default=1.0)
    ap.add_argument("--hidden",          type=int,   default=256)
    ap.add_argument("--n_layers",        type=int,   default=2)
    ap.add_argument("--dropout",         type=float, default=0.1)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--device",          default=None)
    ap.add_argument("--out",             default=DEFAULT_CKPT_NAME,
                    help="Output checkpoint filename (saved under ML/models/). "
                         f"Default: {DEFAULT_CKPT_NAME}")
    args = ap.parse_args()
    ckpt_path = os.path.join(MODELS_DIR, args.out)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")
    print("loading data...")
    solos, meta, n_qualities = load_solos(args.seq_len)
    print(f"  solos usable     : {len(solos)}")
    print(f"  ticks total      : {sum(len(s['token']) for s in solos):,}")
    print(f"  token vocab size : {meta['token_vocab_size']}")
    print(f"  chord qualities  : {n_qualities}")
    print(f"  uniform-loss ref : {np.log(meta['token_vocab_size']):.4f}")

    model = JazzLSTM(
        n_qualities=n_qualities,
        hidden=args.hidden, n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)
    print(f"  model params     : {sum(p.numel() for p in model.parameters()):,}\n")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * args.steps_per_epoch,
    )
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for _ in range(args.steps_per_epoch):
            pt, cr, cq, bp, tg = (t.to(device) for t in
                                  sample_batch(solos, args.batch_size, args.seq_len, rng))
            logits, _ = model(pt, cr, cq, bp)
            loss = loss_fn(logits.reshape(-1, model.vocab_size), tg.reshape(-1))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            sched.step()
            total += loss.item()
        avg = total / args.steps_per_epoch
        print(f"epoch {ep+1:>3d}/{args.epochs}  loss={avg:.4f}  "
              f"lr={optim.param_groups[0]['lr']:.5f}  "
              f"elapsed={time.time() - t0:.0f}s")

        torch.save({
            "model_state": model.state_dict(),
            "vocab_size":  meta["token_vocab_size"],
            "n_qualities": n_qualities,
            "hidden":      args.hidden,
            "n_layers":    args.n_layers,
            "dropout":     args.dropout,
            "epoch":       ep + 1,
        }, ckpt_path)

    print(f"\ndone. checkpoint at {ckpt_path}")


if __name__ == "__main__":
    main()
