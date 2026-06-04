"""Train JazzLSTM on the 16th-note grid produced by `data_prep.py`.

Reads:
    ML/data/processed/grid_solos.parquet
    ML/data/processed/chord_vocab.json
    ML/data/processed/metadata.json
Writes:
    ML/checkpoints/jazz_lstm.pt

CLI:
    python ML/train.py --epochs 30 --batch_size 64 --seq_len 256
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
# Where to save the trained checkpoint. Inference reads this same path by
# default (see engine.py's DEFAULT_CHECKPOINT). To train multiple models
# without overwriting, change the filename here, e.g. "jazz_lstm_v2.pt".
CKPT_PATH = os.path.join("ML", "models", "jazz_lstm.pt")


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------

def load_solos(seq_len):
    """Read the parquet, attach chord (root, quality) ids, bundle per-solo
    numpy arrays so random-chunk sampling is cache-friendly.

    Solos shorter than seq_len+1 are dropped (can't sample a chunk).
    Returns (solos, meta, n_qualities).
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
    """Random fixed-length chunks across the whole corpus.

    At chunk position j (j = 0..seq_len-1) the model gets:
        prev_token = tokens[start + j]        (the token at j-1 absolute)
        chord/bar  = features at tick start+j+1 (the tick being predicted)
        target     = tokens[start + j + 1]
    so the network is trained on next-token prediction.
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
    args = ap.parse_args()

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

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
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
        }, CKPT_PATH)

    print(f"\ndone. checkpoint at {CKPT_PATH}")


if __name__ == "__main__":
    main()
