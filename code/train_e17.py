
"""E17: Per-question pairwise ranking experiments.

E17-A: equal question weights
E17-B: hard-case question weights

Uses only official TRAIN references and frozen embeddings.
Model: 1024 -> 32 -> 1, NumPy Adam.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from train_e14 import (
    load_training_data,
    initialize,
    question_weights,
)

DIM = 1024
N_CAND = 4
EPS = 1e-6


def forward(x, params):
    n = len(x)
    flat = x.reshape(-1, DIM)

    hidden = np.maximum(
        flat @ params["w1"] + params["b1"], 0.0
    )
    scores = (
        hidden @ params["w2"] + params["b2"]
    ).reshape(n, N_CAND)

    return flat, hidden, scores


def ranking_loss_grad(scores, labels, weights):
    """Per-question average pairwise loss and gradient.

    Fully correct groups contribute neither loss nor denominator.
    """
    positive = labels > 0.5
    negative = ~positive

    mask = (
        positive[:, :, None]
        & negative[:, None, :]
    )

    pair_counts = mask.sum(axis=(1, 2))
    valid = pair_counts > 0

    if not np.any(valid):
        return 0.0, np.zeros_like(scores, dtype=np.float32)

    # diff[q,i,j] = positive score - negative score.
    diff = (
        scores[:, :, None] - scores[:, None, :]
    )

    # Stable softplus(-diff).
    pair_losses = np.logaddexp(0.0, -diff)

    per_question_loss = (
        (pair_losses * mask).sum(axis=(1, 2))
        / np.maximum(pair_counts, 1)
    )

    effective_weights = weights * valid
    denominator = float(effective_weights.sum())

    loss = float(
        (per_question_loss * effective_weights).sum()
        / denominator
    )

    # d softplus(-diff) / d diff = -sigmoid(-diff).
    probabilities = 1.0 / (
        1.0 + np.exp(np.clip(diff, -30.0, 30.0))
    )

    coefficients = (
        probabilities
        * mask
        * (
            effective_weights
            / np.maximum(pair_counts, 1)
            / denominator
        )[:, None, None]
    )

    # Positive score increases; negative score decreases.
    ds = (
        -coefficients.sum(axis=2)
        + coefficients.sum(axis=1)
    )

    return loss, ds.astype(np.float32)


def evaluate(x, y, params):
    _, _, scores = forward(x, params)

    # Common unweighted dev loss for fair epoch selection.
    uniform = np.ones(len(y), dtype=np.float32)
    loss, _ = ranking_loss_grad(scores, y, uniform)

    selected = np.argmax(scores, axis=1)
    accuracy = float(
        y[np.arange(len(y)), selected].mean()
    )

    counts = y.sum(axis=1)
    hard = counts == 1

    hard_acc = (
        float(
            y[np.arange(len(y))[hard], selected[hard]].mean()
        )
        if np.any(hard) else float("nan")
    )

    return loss, accuracy, hard_acc


def train(
    x, y, weights, params,
    epochs, batch_size, lr, weight_decay, seed,
    x_dev=None, y_dev=None,
):
    rng = np.random.default_rng(seed)

    adam_m = {
        key: np.zeros_like(value)
        for key, value in params.items()
    }
    adam_v = {
        key: np.zeros_like(value)
        for key, value in params.items()
    }

    step = 0
    best_epoch = 1
    best_dev_loss = float("inf")

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y))

        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]

            xb = x[idx]
            yb = y[idx]
            wb = weights[idx]

            flat, hidden, scores = forward(xb, params)

            _, ds = ranking_loss_grad(scores, yb, wb)

            if not np.any(ds):
                continue

            ds = ds.reshape(-1, 1)

            grads = {
                "w2": (
                    hidden.T @ ds
                    + weight_decay * params["w2"]
                ),
                "b2": ds.sum(axis=0),
            }

            dh = (
                ds @ params["w2"].T
            ) * (hidden > 0)

            grads["w1"] = (
                flat.T @ dh
                + weight_decay * params["w1"]
            )
            grads["b1"] = dh.sum(axis=0)

            step += 1

            for key in params:
                g = grads[key]

                adam_m[key] = (
                    0.9 * adam_m[key] + 0.1 * g
                )
                adam_v[key] = (
                    0.999 * adam_v[key]
                    + 0.001 * g * g
                )

                m_hat = (
                    adam_m[key] / (1.0 - 0.9 ** step)
                )
                v_hat = (
                    adam_v[key] / (1.0 - 0.999 ** step)
                )

                params[key] -= (
                    lr * m_hat
                    / (np.sqrt(v_hat) + 1e-8)
                )

        if x_dev is not None:
            train_loss, train_acc, train_hard = evaluate(
                x, y, params
            )
            dev_loss, dev_acc, dev_hard = evaluate(
                x_dev, y_dev, params
            )

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_epoch = epoch

            print(
                f"Epoch {epoch:2d} | "
                f"train loss {train_loss:.5f} | "
                f"dev loss {dev_loss:.5f} | "
                f"dev acc {dev_acc:.4f} | "
                f"dev 1-correct acc {dev_hard:.4f}",
                flush=True,
            )
        else:
            print(
                f"Final training epoch {epoch}/{epochs}",
                flush=True,
            )

    return best_epoch, best_dev_loss


def main():
    start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "weighted"],
    )
    parser.add_argument(
        "--output", type=Path, required=True
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--weight-decay", type=float, default=0.01
    )
    parser.add_argument("--seed", type=int, default=17)

    args = parser.parse_args()

    x, y, correct_counts = load_training_data()
    weights = question_weights(correct_counts, args.mode)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(x))

    n_dev = max(1, int(len(x) * 0.1))
    dev_ids = order[:n_dev]
    train_ids = order[n_dev:]

    x_train = x[train_ids]
    y_train = y[train_ids]
    w_train = weights[train_ids]

    x_dev = x[dev_ids]
    y_dev = y[dev_ids]

    # Normalize using the internal training portion only.
    flat = x_train.reshape(-1, DIM)
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), EPS)

    x_train = (x_train - mean) / std
    x_dev = (x_dev - mean) / std

    print(f"\nMode: {args.mode}")
    print(f"Train questions: {len(train_ids)}")
    print(f"Internal dev questions: {len(dev_ids)}")
    print(
        "Dev single-correct questions:",
        int((y_dev.sum(axis=1) == 1).sum()),
    )
    print(f"Hidden: {args.hidden}")
    print(f"Batch size: {args.batch_size}")

    print("\nPhase 1: internal development selection")

    params = initialize(args.seed, args.hidden)

    best_epoch, best_loss = train(
        x_train, y_train, w_train, params,
        args.epochs, args.batch_size, args.lr,
        args.weight_decay, args.seed,
        x_dev, y_dev,
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best internal dev pair loss: {best_loss:.6f}")

    # Retrain on full official TRAIN.
    full_flat = x.reshape(-1, DIM)
    full_mean = full_flat.mean(axis=0)
    full_std = np.maximum(full_flat.std(axis=0), EPS)
    full_x = (x - full_mean) / full_std

    print("\nPhase 2: full TRAIN retraining")

    final_params = initialize(args.seed, args.hidden)

    train(
        full_x, y, weights, final_params,
        best_epoch, args.batch_size, args.lr,
        args.weight_decay, args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.output,
        **final_params,
        mean=full_mean.astype(np.float32),
        std=full_std.astype(np.float32),
    )

    print(f"\nCheckpoint: {args.output}")
    print(
        f"Checkpoint size: "
        f"{args.output.stat().st_size:,} bytes"
    )
    print(
        f"Training pipeline seconds: "
        f"{time.perf_counter() - start:.3f}"
    )


if __name__ == "__main__":
    main()
