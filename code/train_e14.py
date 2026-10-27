
"""E14: Hard-case weighted BCE training.

E14-A: Ordinary BCE baseline.
E14-B: Hard-case weighted BCE.

Both use:
- 1024 -> 32 -> 1 MLP
- Official TRAIN data only
- Same initialization and Adam optimizer
- Same train/internal-dev split
- Epoch selection by ordinary internal-dev BCE
- Full TRAIN retraining after epoch selection

No validation labels are used during training.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer


INPUT_DIM = 1024
CANDIDATES = 4
EPS = 1e-6

HARD_WEIGHTS = {
    1: 2.0,
    2: 1.5,
    3: 1.0,
    4: 0.25,
}


def load_training_data():
    bank, arrays = load_inputs("train")
    x = arrays["embeddings"].astype(np.float32)

    if x.shape != (len(bank), CANDIDATES, INPUT_DIM):
        raise ValueError(f"Unexpected embedding shape: {x.shape}")

    with open(DATA / "gsm8k_train.jsonl", encoding="utf-8") as f:
        references = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if len(bank) != len(references):
        raise ValueError("Training reference count mismatch")

    labels = []

    for row, ref in zip(bank, references):
        if (
            row["id"] != ref["id"]
            or row["question"] != ref["question"]
        ):
            raise ValueError("Training reference alignment mismatch")

        gold = parse_answer(ref["answer"])

        if gold is None:
            raise ValueError(f"Cannot parse gold: {row['id']}")

        row_labels = [
            float(parse_answer(c["text"]) == gold)
            for c in row["candidates"]
        ]

        labels.append(row_labels)

    y = np.asarray(labels, dtype=np.float32)

    if y.shape != (len(bank), CANDIDATES):
        raise ValueError("Unexpected label shape")

    correct_counts = y.sum(axis=1).astype(int)

    if np.any(correct_counts < 1) or np.any(correct_counts > 4):
        raise ValueError("Invalid correct-candidate count")

    print(f"Total TRAIN questions: {len(bank)}")
    for count in range(1, 5):
        print(
            f"  {count} correct candidates: "
            f"{int((correct_counts == count).sum())}"
        )

    return x, y, correct_counts


def question_weights(correct_counts, mode):
    if mode == "baseline":
        return np.ones(
            len(correct_counts),
            dtype=np.float32
        )

    if mode == "weighted":
        return np.asarray(
            [HARD_WEIGHTS[int(k)] for k in correct_counts],
            dtype=np.float32
        )

    raise ValueError(f"Unknown mode: {mode}")


def initialize(seed, hidden):
    rng = np.random.default_rng(seed)

    return {
        "w1": (
            rng.standard_normal((INPUT_DIM, hidden))
            * np.sqrt(2.0 / INPUT_DIM)
        ).astype(np.float32),

        "b1": np.zeros(hidden, dtype=np.float32),

        "w2": (
            rng.standard_normal((hidden, 1))
            * np.sqrt(1.0 / hidden)
        ).astype(np.float32),

        "b2": np.zeros(1, dtype=np.float32),
    }


def forward(x, params):
    n = len(x)
    flat_x = x.reshape(-1, INPUT_DIM)

    hidden = np.maximum(
        flat_x @ params["w1"] + params["b1"],
        0.0
    )

    logits = (
        hidden @ params["w2"] + params["b2"]
    ).reshape(n, CANDIDATES)

    return flat_x, hidden, logits


def bce_loss_and_grad(logits, labels, weights):
    """Weighted BCE normalized by sum of question weights.

    logits, labels: (n_questions, 4)
    weights: (n_questions,)

    The ordinary evaluation loss uses all weights equal to 1.
    """
    weights = weights.astype(np.float32)

    # Stable binary cross entropy with logits.
    element_loss = (
        np.logaddexp(0.0, logits) - labels * logits
    )

    denominator = float(weights.sum()) * CANDIDATES

    loss = float(
        (element_loss * weights[:, None]).sum()
        / denominator
    )

    sigmoid = 1.0 / (
        1.0 + np.exp(-np.clip(logits, -30.0, 30.0))
    )

    grad = (
        (sigmoid - labels)
        * weights[:, None]
        / denominator
    )

    return loss, grad.astype(np.float32)


def evaluate(x, y, params):
    _, _, logits = forward(x, params)

    # Always evaluate with ordinary BCE,
    # so both experiments use the same epoch-selection metric.
    uniform = np.ones(len(y), dtype=np.float32)
    loss, _ = bce_loss_and_grad(logits, y, uniform)

    predictions = np.argmax(logits, axis=1)
    correct = y[np.arange(len(y)), predictions]

    accuracy = float(correct.mean())

    return loss, accuracy


def train(
    x,
    y,
    weights,
    params,
    epochs,
    batch_size,
    lr,
    weight_decay,
    seed,
    x_dev=None,
    y_dev=None,
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
            indices = order[start:start + batch_size]

            xb = x[indices]
            yb = y[indices]
            wb = weights[indices]

            flat_x, hidden, logits = forward(xb, params)

            _, dlogits = bce_loss_and_grad(
                logits, yb, wb
            )

            ds = dlogits.reshape(-1, 1)

            grads = {
                "w2": (
                    hidden.T @ ds
                    + weight_decay * params["w2"]
                ),
                "b2": ds.sum(axis=0),
            }

            dh = (
                ds @ params["w2"].T
            ) * (hidden > 0.0)

            grads["w1"] = (
                flat_x.T @ dh
                + weight_decay * params["w1"]
            )

            grads["b1"] = dh.sum(axis=0)

            step += 1

            for key in params:
                grad = grads[key]

                adam_m[key] = (
                    0.9 * adam_m[key] + 0.1 * grad
                )

                adam_v[key] = (
                    0.999 * adam_v[key]
                    + 0.001 * grad * grad
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
            train_loss, train_acc = evaluate(x, y, params)
            dev_loss, dev_acc = evaluate(
                x_dev, y_dev, params
            )

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_epoch = epoch

            print(
                f"Epoch {epoch:2d} | "
                f"train BCE {train_loss:.5f} | "
                f"dev BCE {dev_loss:.5f} | "
                f"dev acc {dev_acc:.4f}",
                flush=True
            )
        else:
            print(
                f"Final training epoch {epoch}/{epochs}",
                flush=True
            )

    return best_epoch, best_dev_loss


def main():
    timer = time.perf_counter()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["baseline", "weighted"],
        required=True
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True
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

    dev_indices = order[:n_dev]
    train_indices = order[n_dev:]

    x_train = x[train_indices]
    y_train = y[train_indices]
    w_train = weights[train_indices]

    x_dev = x[dev_indices]
    y_dev = y[dev_indices]

    # Fit normalization on internal training portion only.
    flat_train = x_train.reshape(-1, INPUT_DIM)

    mean = flat_train.mean(axis=0)
    std = np.maximum(flat_train.std(axis=0), EPS)

    x_train = (x_train - mean) / std
    x_dev = (x_dev - mean) / std

    print(f"\nMode: {args.mode}")
    print(f"Train questions: {len(train_indices)}")
    print(f"Internal dev questions: {len(dev_indices)}")
    print(f"Hidden dimension: {args.hidden}")
    print(f"Batch size: {args.batch_size}")
    print(f"Weight decay: {args.weight_decay}")

    print("\nPhase 1: choose epoch on internal dev")

    params = initialize(args.seed, args.hidden)

    best_epoch, best_dev_loss = train(
        x_train,
        y_train,
        w_train,
        params,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.seed,
        x_dev,
        y_dev,
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best internal dev BCE: {best_dev_loss:.6f}")

    # Refit normalization on the full official TRAIN set.
    flat_full = x.reshape(-1, INPUT_DIM)
    full_mean = flat_full.mean(axis=0)
    full_std = np.maximum(
        flat_full.std(axis=0), EPS
    )

    x_full = (x - full_mean) / full_std

    print("\nPhase 2: retrain on full TRAIN")

    final_params = initialize(args.seed, args.hidden)

    train(
        x_full,
        y,
        weights,
        final_params,
        best_epoch,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.seed,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        args.output,
        **final_params,
        mean=full_mean.astype(np.float32),
        std=full_std.astype(np.float32),
    )

    elapsed = time.perf_counter() - timer

    print(f"\nCheckpoint: {args.output}")
    print(
        f"Checkpoint size: "
        f"{args.output.stat().st_size:,} bytes"
    )
    print(f"Training pipeline seconds: {elapsed:.3f}")


if __name__ == "__main__":
    main()
