
"""E2: Train a logistic-regression scorer on frozen embeddings."""

import argparse
import json
from pathlib import Path

import numpy as np

from common import load_inputs, parse_answer, DATA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/e2-scorer.npz")
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()

    # Official loader: shape (N, 4, 1024)
    bank, arrays = load_inputs("train")
    x = arrays["embeddings"].astype(np.float32)

    with open(DATA / "gsm8k_train.jsonl", encoding="utf-8") as f:
        references = [json.loads(line) for line in f if line.strip()]

    if len(bank) != len(references):
        raise ValueError("Train reference count mismatch")

    labels = []
    for row, ref in zip(bank, references):
        if row["id"] != ref["id"] or row["question"] != ref["question"]:
            raise ValueError("Train reference alignment mismatch")

        gold = parse_answer(ref["answer"])
        if gold is None:
            raise ValueError("Could not parse a train reference answer")

        labels.extend([
            int(parse_answer(c["text"]) == gold)
            for c in row["candidates"]
        ])

    y = np.asarray(labels, dtype=np.float32)
    x = x.reshape(-1, 1024)

    # Standardize using TRAIN statistics only.
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.maximum(std, 1e-6)
    x = (x - mean) / std

    # Logistic regression, trained with Adam in NumPy.
    rng = np.random.default_rng(17)
    w = np.zeros(1024, dtype=np.float32)
    b = np.float32(0.0)

    mw = np.zeros_like(w)
    vw = np.zeros_like(w)
    mb = np.float32(0.0)
    vb = np.float32(0.0)

    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    n = len(y)

    print(f"Train questions: {len(bank)}")
    print(f"Train candidates: {n}")
    print(f"Positive labels: {int(y.sum())}")
    print(f"Positive fraction: {float(y.mean()):.4f}")

    for step in range(1, args.epochs + 1):
        logits = np.clip(x @ w + b, -30, 30)
        probs = 1.0 / (1.0 + np.exp(-logits))

        error = probs - y
        dw = (x.T @ error) / n + args.weight_decay * w
        db = np.float32(error.mean())

        mw = beta1 * mw + (1 - beta1) * dw
        vw = beta2 * vw + (1 - beta2) * (dw * dw)
        mb = beta1 * mb + (1 - beta1) * db
        vb = beta2 * vb + (1 - beta2) * (db * db)

        mw_hat = mw / (1 - beta1 ** step)
        vw_hat = vw / (1 - beta2 ** step)
        mb_hat = mb / (1 - beta1 ** step)
        vb_hat = vb / (1 - beta2 ** step)

        w -= args.lr * mw_hat / (np.sqrt(vw_hat) + eps)
        b -= args.lr * mb_hat / (np.sqrt(vb_hat) + eps)

        if step == 1 or step % 50 == 0 or step == args.epochs:
            logits = x @ w + b
            loss = float(
                np.logaddexp(0, logits).mean()
                - (y * logits).mean()
                + 0.5 * args.weight_decay * (w @ w)
            )
            print(f"Step {step:4d} | train loss {loss:.5f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.output,
        weight=w,
        bias=np.asarray(b, dtype=np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )

    print(f"Checkpoint saved: {args.output}")
    print(f"Checkpoint size: {args.output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
