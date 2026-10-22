
"""E13: Group-aware MLP with multi-positive group loss."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from common import DATA, load_inputs, parse_answer

DIM = 1024
FEATURE_DIM = 2050
EPS = 1e-6


def make_features(bank, embeddings):
    """Original, relative, voting, and similarity features."""
    e = embeddings.astype(np.float32)
    n = len(bank)

    group_mean = e.mean(axis=1, keepdims=True)
    relative = e - group_mean

    norm = e / np.maximum(
        np.linalg.norm(e, axis=2, keepdims=True), EPS
    )
    similarity = norm @ norm.transpose(0, 2, 1)

    # Average similarity to the other three candidates.
    mean_similarity = (
        similarity.sum(axis=2)
        - np.diagonal(similarity, axis1=1, axis2=2)
    ) / 3.0

    votes = np.zeros((n, 4), dtype=np.float32)

    for i, row in enumerate(bank):
        answers = [
            parse_answer(c["text"]) for c in row["candidates"]
        ]
        counts = Counter(a for a in answers if a is not None)

        for j, answer in enumerate(answers):
            votes[i, j] = (
                counts[answer] / 4.0 if answer is not None else 0.0
            )

    return np.concatenate(
        [
            e,
            relative,
            votes[..., None],
            mean_similarity[..., None],
        ],
        axis=2,
    ).astype(np.float32)


def initialize(seed, hidden):
    rng = np.random.default_rng(seed)
    return {
        "w1": (
            rng.standard_normal((FEATURE_DIM, hidden))
            * np.sqrt(2.0 / FEATURE_DIM)
        ).astype(np.float32),
        "b1": np.zeros(hidden, dtype=np.float32),
        "w2": (
            rng.standard_normal((hidden, 1))
            * np.sqrt(1.0 / hidden)
        ).astype(np.float32),
        "b2": np.zeros(1, dtype=np.float32),
    }


def forward(x, p):
    flat = x.reshape(-1, FEATURE_DIM)
    hidden = np.maximum(flat @ p["w1"] + p["b1"], 0)
    scores = (hidden @ p["w2"] + p["b2"]).reshape(-1, 4)
    return flat, hidden, scores


def group_loss_grad(scores, labels):
    """Negative log of total probability assigned to correct candidates.

    Loss = -log(sum softmax(scores)[correct candidates]).
    """
    correct = labels > 0.5

    if np.any(~correct.any(axis=1)):
        raise ValueError("A training question has no correct candidate")

    # Numerically stable softmax over all four candidates.
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_all = np.exp(shifted)
    prob_all = exp_all / exp_all.sum(axis=1, keepdims=True)

    positive_exp = exp_all * correct
    positive_sum = positive_exp.sum(axis=1, keepdims=True)

    prob_positive = positive_exp / positive_sum

    loss_each = (
        np.log(exp_all.sum(axis=1))
        - np.log(positive_sum[:, 0])
    )

    loss = float(loss_each.mean())
    ds = (prob_all - prob_positive) / len(scores)

    return loss, ds.astype(np.float32)


def evaluate(x, y, p):
    _, _, scores = forward(x, p)
    loss, _ = group_loss_grad(scores, y)
    predictions = np.argmax(scores, axis=1)
    accuracy = float(
        y[np.arange(len(y)), predictions].mean()
    )
    return loss, accuracy


def train(
    x, y, p, epochs, batch_size, lr, wd, seed,
    x_dev=None, y_dev=None
):
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}

    step = 0
    best_epoch = 1
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y))

        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb, yb = x[idx], y[idx]

            flat, hidden, scores = forward(xb, p)
            _, ds = group_loss_grad(scores, yb)
            ds = ds.reshape(-1, 1)

            grads = {
                "w2": hidden.T @ ds + wd * p["w2"],
                "b2": ds.sum(axis=0),
            }

            dh = (ds @ p["w2"].T) * (hidden > 0)
            grads["w1"] = flat.T @ dh + wd * p["w1"]
            grads["b1"] = dh.sum(axis=0)

            step += 1

            for key in p:
                g = grads[key]
                m[key] = 0.9 * m[key] + 0.1 * g
                v[key] = 0.999 * v[key] + 0.001 * g * g

                mhat = m[key] / (1 - 0.9 ** step)
                vhat = v[key] / (1 - 0.999 ** step)

                p[key] -= lr * mhat / (
                    np.sqrt(vhat) + 1e-8
                )

        if x_dev is not None:
            tr_loss, tr_acc = evaluate(x, y, p)
            dev_loss, dev_acc = evaluate(x_dev, y_dev, p)

            if dev_loss < best_loss:
                best_loss = dev_loss
                best_epoch = epoch

            print(
                f"Epoch {epoch:2d} | "
                f"train loss {tr_loss:.5f} | "
                f"dev loss {dev_loss:.5f} | "
                f"dev acc {dev_acc:.4f}",
                flush=True
            )
        else:
            print(f"Final training epoch {epoch}/{epochs}", flush=True)

    return best_epoch


def main():
    timer = time.perf_counter()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/e13-group-aware.npz")
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    bank, arrays = load_inputs("train")
    features = make_features(bank, arrays["embeddings"])

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
            raise ValueError("Unparseable training reference")

        labels.append([
            float(parse_answer(c["text"]) == gold)
            for c in row["candidates"]
        ])

    labels = np.asarray(labels, dtype=np.float32)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(bank))
    n_dev = max(1, int(len(bank) * 0.1))

    dev_ids = order[:n_dev]
    train_ids = order[n_dev:]

    x_train, y_train = features[train_ids], labels[train_ids]
    x_dev, y_dev = features[dev_ids], labels[dev_ids]

    # Normalization uses only the internal training portion.
    flat_train = x_train.reshape(-1, FEATURE_DIM)
    mean = flat_train.mean(axis=0)
    std = np.maximum(flat_train.std(axis=0), EPS)

    x_train = (x_train - mean) / std
    x_dev = (x_dev - mean) / std

    print(f"Train questions: {len(train_ids)}")
    print(f"Internal dev questions: {len(dev_ids)}")
    print(f"Feature dimension: {FEATURE_DIM}")
    print(f"Hidden dimension: {args.hidden}")
    print(f"Batch size (questions): {args.batch_size}")

    print("\nPhase 1: internal dev selection")
    p = initialize(args.seed, args.hidden)

    best_epoch = train(
        x_train, y_train, p,
        args.epochs, args.batch_size, args.lr,
        args.weight_decay, args.seed,
        x_dev, y_dev
    )

    print(f"\nBest epoch: {best_epoch}")

    # Reinitialize, then retrain on all official TRAIN data.
    full_flat = features.reshape(-1, FEATURE_DIM)
    full_mean = full_flat.mean(axis=0)
    full_std = np.maximum(full_flat.std(axis=0), EPS)
    full_x = (features - full_mean) / full_std

    print("\nPhase 2: full TRAIN retraining")
    final_p = initialize(args.seed, args.hidden)

    train(
        full_x, labels, final_p,
        best_epoch, args.batch_size, args.lr,
        args.weight_decay, args.seed
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        **final_p,
        mean=full_mean.astype(np.float32),
        std=full_std.astype(np.float32),
    )

    print(f"\nCheckpoint: {args.output}")
    print(f"Checkpoint size: {args.output.stat().st_size:,} bytes")
    print(
        f"Training pipeline seconds: "
        f"{time.perf_counter() - timer:.3f}"
    )


if __name__ == "__main__":
    main()
