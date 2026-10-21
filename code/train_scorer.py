
"""E6: Train a 1024 -> 64 -> 1 MLP scorer using NumPy."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer


def initialize(rng, hidden):
    return {
        "w1": (rng.standard_normal((1024, hidden)) *
               np.sqrt(2.0 / 1024)).astype(np.float32),
        "b1": np.zeros(hidden, dtype=np.float32),
        "w2": (rng.standard_normal((hidden, 1)) *
               np.sqrt(1.0 / hidden)).astype(np.float32),
        "b2": np.zeros(1, dtype=np.float32),
    }


def forward(x, p):
    h = np.maximum(x @ p["w1"] + p["b1"], 0)
    logits = h @ p["w2"] + p["b2"]
    return h, logits


def loss_value(x, y, p):
    _, logits = forward(x, p)
    logits = logits[:, 0]
    return float(np.mean(np.logaddexp(0, logits) - y * logits))


def train_epochs(x, y, p, epochs, batch_size, lr, wd, seed,
                 x_dev=None, y_dev=None):
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    step = 0
    best_loss = float("inf")
    best_epoch = 1

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y))

        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb, yb = x[idx], y[idx, None]
            n = len(idx)

            h, logits = forward(xb, p)
            prob = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))

            dz = (prob - yb) / n
            grads = {
                "w2": h.T @ dz + wd * p["w2"],
                "b2": dz.sum(axis=0),
            }

            dh = (dz @ p["w2"].T) * (h > 0)
            grads["w1"] = xb.T @ dh + wd * p["w1"]
            grads["b1"] = dh.sum(axis=0)

            step += 1
            for key in p:
                g = grads[key]
                m[key] = 0.9 * m[key] + 0.1 * g
                v[key] = 0.999 * v[key] + 0.001 * (g * g)

                mhat = m[key] / (1 - 0.9 ** step)
                vhat = v[key] / (1 - 0.999 ** step)

                p[key] -= lr * mhat / (np.sqrt(vhat) + 1e-8)

        if x_dev is not None:
            dev_loss = loss_value(x_dev, y_dev, p)
            if dev_loss < best_loss:
                best_loss = dev_loss
                best_epoch = epoch
            print(
                f"Epoch {epoch:2d} | "
                f"train loss {loss_value(x, y, p):.5f} | "
                f"dev loss {dev_loss:.5f}",
                flush=True
            )
        else:
            print(
                f"Final training epoch {epoch}/{epochs}",
                flush=True
            )

    return best_epoch


def main():
    timer = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("runs/e6-mlp.npz"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    bank, arrays = load_inputs("train")
    embeddings = arrays["embeddings"].astype(np.float32)

    with open(DATA / "gsm8k_train.jsonl", encoding="utf-8") as f:
        refs = [json.loads(line) for line in f if line.strip()]

    if len(bank) != len(refs):
        raise ValueError("Reference count mismatch")

    labels = []
    for row, ref in zip(bank, refs):
        if row["id"] != ref["id"] or row["question"] != ref["question"]:
            raise ValueError("Reference alignment mismatch")

        gold = parse_answer(ref["answer"])
        if gold is None:
            raise ValueError("Unparseable reference")

        labels.append([
            float(parse_answer(c["text"]) == gold)
            for c in row["candidates"]
        ])

    labels = np.asarray(labels, dtype=np.float32)
    num_questions = len(bank)

    # Split by QUESTION to avoid candidate leakage.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(num_questions)
    n_dev = max(1, int(num_questions * 0.1))

    dev_ids = order[:n_dev]
    train_ids = order[n_dev:]

    def flatten(ids):
        xx = embeddings[ids].reshape(-1, 1024)
        yy = labels[ids].reshape(-1)
        return xx, yy

    x_train, y_train = flatten(train_ids)
    x_dev, y_dev = flatten(dev_ids)

    # Statistics computed from internal training portion only.
    mean = x_train.mean(axis=0)
    std = np.maximum(x_train.std(axis=0), 1e-6)

    x_train = (x_train - mean) / std
    x_dev = (x_dev - mean) / std

    print(f"Train questions: {len(train_ids)}")
    print(f"Internal dev questions: {len(dev_ids)}")
    print(f"Hidden dimension: {args.hidden}")

    print("\nPhase 1: choose epoch using internal dev")
    p = initialize(np.random.default_rng(args.seed), args.hidden)

    best_epoch = train_epochs(
        x_train, y_train, p,
        args.epochs, args.batch_size, args.lr,
        args.weight_decay, args.seed,
        x_dev, y_dev
    )

    print(f"\nBest epoch: {best_epoch}")

    # Retrain on the full official TRAIN split.
    x_full = embeddings.reshape(-1, 1024)
    y_full = labels.reshape(-1)
    full_mean = x_full.mean(axis=0)
    full_std = np.maximum(x_full.std(axis=0), 1e-6)
    x_full = (x_full - full_mean) / full_std

    print("\nPhase 2: retrain on full TRAIN")
    p_final = initialize(
        np.random.default_rng(args.seed), args.hidden
    )

    train_epochs(
        x_full, y_full, p_final,
        best_epoch, args.batch_size, args.lr,
        args.weight_decay, args.seed
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.output,
        **p_final,
        mean=full_mean.astype(np.float32),
        std=full_std.astype(np.float32),
    )

    elapsed = time.perf_counter() - timer
    print(f"\nCheckpoint: {args.output}")
    print(f"Checkpoint size: {args.output.stat().st_size:,} bytes")
    print(f"Training pipeline seconds: {elapsed:.3f}")


if __name__ == "__main__":
    main()
