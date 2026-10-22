
"""E10: Bradley-Terry Pairwise Ranking MLP.

Train a 1024 -> 32 -> 1 scorer with pairwise ranking loss.
Uses only official TRAIN references and frozen embeddings.

NumPy implementation, CPU compatible.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer


INPUT_DIM = 1024
CANDIDATES = 4


def initialize(seed, hidden):
    """Initialize MLP parameters."""
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
    """Forward pass.

    x: (batch_questions, 4, 1024)

    Returns:
        hidden: (batch_questions * 4, hidden_dim)
        scores: (batch_questions, 4)
    """
    batch_questions = x.shape[0]

    flat_x = x.reshape(-1, INPUT_DIM)

    hidden = np.maximum(
        flat_x @ params["w1"] + params["b1"],
        0.0
    )

    flat_scores = (
        hidden @ params["w2"] + params["b2"]
    )

    scores = flat_scores.reshape(batch_questions, CANDIDATES)

    return hidden, scores


def pair_loss_and_grad(scores, labels):
    """Compute Bradley-Terry loss and gradients.

    Compare every correct candidate against every incorrect
    candidate within the same question.

    scores: (batch_questions, 4)
    labels: (batch_questions, 4)

    Returns:
        loss: average pairwise loss
        ds: derivative with respect to scores
        n_pairs: number of valid positive-negative pairs
    """
    positive = labels > 0.5
    negative = ~positive

    # mask[b, i, j] is True when:
    # candidate i is correct and candidate j is incorrect.
    mask = (
        positive[:, :, None]
        & negative[:, None, :]
    )

    n_pairs = int(mask.sum())

    if n_pairs == 0:
        return (
            0.0,
            np.zeros_like(scores, dtype=np.float32),
            0
        )

    # diff[b, i, j] = score_i - score_j
    diff = (
        scores[:, :, None]
        - scores[:, None, :]
    )

    active_diff = diff[mask]

    # Softplus(-diff), numerically stable.
    loss = float(
        np.logaddexp(0.0, -active_diff).mean()
    )

    # d loss / d diff = -sigmoid(-diff)
    sigmoid_negative = (
        1.0 / (1.0 + np.exp(np.clip(diff, -30, 30)))
    )

    contributions = np.where(
        mask,
        sigmoid_negative,
        0.0
    )

    # Positive candidate gradients: negative (raise score).
    # Negative candidate gradients: positive (lower score).
    ds = (
        -contributions.sum(axis=2)
        + contributions.sum(axis=1)
    ) / n_pairs

    return loss, ds.astype(np.float32), n_pairs


def evaluate_loss(x, y, params):
    """Evaluate average pairwise loss without updating weights."""
    _, scores = forward(x, params)

    loss, _, n_pairs = pair_loss_and_grad(scores, y)

    if n_pairs == 0:
        raise ValueError(
            "No positive-negative pairs in evaluation data"
        )

    return loss


def train(
    x,
    y,
    params,
    epochs,
    batch_size,
    lr,
    weight_decay,
    seed,
    x_dev=None,
    y_dev=None,
):
    """Train MLP with Adam and pairwise ranking loss."""
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

    hidden_dim = params["w1"].shape[1]

    for epoch in range(1, epochs + 1):

        # Shuffle questions, not individual candidates.
        order = rng.permutation(len(y))

        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]

            xb = x[indices]
            yb = y[indices]

            # Forward:
            # hidden: (batch_questions * 4, hidden_dim)
            # scores: (batch_questions, 4)
            hidden, scores = forward(xb, params)

            _, ds, n_pairs = pair_loss_and_grad(
                scores, yb
            )

            if n_pairs == 0:
                continue

            # FIX: reshape the score gradients into candidate rows.
            ds = ds.reshape(-1, 1)

            flat_x = xb.reshape(-1, INPUT_DIM)

            # Shapes:
            # hidden.T: (hidden_dim, batch_questions * 4)
            # ds:       (batch_questions * 4, 1)
            # w2 grad:  (hidden_dim, 1)
            grads = {
                "w2": (
                    hidden.T @ ds
                    + weight_decay * params["w2"]
                ),
                "b2": ds.sum(axis=0),
            }

            # Backpropagate through ReLU.
            dh = (
                (ds @ params["w2"].T)
                * (hidden > 0)
            )

            grads["w1"] = (
                flat_x.T @ dh
                + weight_decay * params["w1"]
            )

            grads["b1"] = dh.sum(axis=0)

            # Adam update.
            step += 1

            for key in params:
                grad = grads[key]

                adam_m[key] = (
                    0.9 * adam_m[key]
                    + 0.1 * grad
                )

                adam_v[key] = (
                    0.999 * adam_v[key]
                    + 0.001 * (grad * grad)
                )

                m_hat = (
                    adam_m[key]
                    / (1.0 - 0.9 ** step)
                )

                v_hat = (
                    adam_v[key]
                    / (1.0 - 0.999 ** step)
                )

                params[key] -= (
                    lr * m_hat
                    / (np.sqrt(v_hat) + 1e-8)
                )

        # Evaluate on internal development set.
        if x_dev is not None:
            train_loss = evaluate_loss(x, y, params)
            dev_loss = evaluate_loss(
                x_dev, y_dev, params
            )

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_epoch = epoch

            print(
                f"Epoch {epoch:2d} | "
                f"train pair loss {train_loss:.5f} | "
                f"dev pair loss {dev_loss:.5f}",
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

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/e10-pairwise.npz")
    )

    parser.add_argument(
        "--epochs", type=int, default=30
    )

    parser.add_argument(
        "--hidden", type=int, default=32
    )

    parser.add_argument(
        "--batch-size", type=int, default=64
    )

    parser.add_argument(
        "--lr", type=float, default=0.001
    )

    parser.add_argument(
        "--weight-decay", type=float, default=0.01
    )

    parser.add_argument(
        "--seed", type=int, default=17
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # 1. Load official training data.
    # --------------------------------------------------
    bank, arrays = load_inputs("train")

    embeddings = arrays["embeddings"].astype(
        np.float32
    )

    if embeddings.shape != (
        len(bank), CANDIDATES, INPUT_DIM
    ):
        raise ValueError(
            f"Unexpected embeddings shape: "
            f"{embeddings.shape}"
        )

    with open(
        DATA / "gsm8k_train.jsonl",
        encoding="utf-8"
    ) as file:
        references = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if len(bank) != len(references):
        raise ValueError(
            "Training bank and reference count mismatch"
        )

    # --------------------------------------------------
    # 2. Construct correctness labels using TRAIN only.
    # --------------------------------------------------
    labels = []

    for row, ref in zip(bank, references):

        if (
            row["id"] != ref["id"]
            or row["question"] != ref["question"]
        ):
            raise ValueError(
                "Training reference alignment mismatch"
            )

        gold_answer = parse_answer(ref["answer"])

        if gold_answer is None:
            raise ValueError(
                f"Unparseable gold answer: {row['id']}"
            )

        row_labels = [
            float(
                parse_answer(candidate["text"])
                == gold_answer
            )
            for candidate in row["candidates"]
        ]

        labels.append(row_labels)

    labels = np.asarray(
        labels, dtype=np.float32
    )

    if labels.shape != (
        len(bank), CANDIDATES
    ):
        raise ValueError("Incorrect labels shape")

    # --------------------------------------------------
    # 3. Internal 90/10 split by question.
    # --------------------------------------------------
    rng = np.random.default_rng(args.seed)

    order = rng.permutation(len(bank))

    n_dev = max(
        1, int(len(bank) * 0.1)
    )

    dev_indices = order[:n_dev]
    train_indices = order[n_dev:]

    x_train = embeddings[train_indices]
    y_train = labels[train_indices]

    x_dev = embeddings[dev_indices]
    y_dev = labels[dev_indices]

    # Normalization statistics based only on
    # the internal training portion.
    flat_train = x_train.reshape(-1, INPUT_DIM)

    mean = flat_train.mean(axis=0)

    std = np.maximum(
        flat_train.std(axis=0), 1e-6
    )

    x_train = (x_train - mean) / std
    x_dev = (x_dev - mean) / std

    print(f"Train questions: {len(train_indices)}")
    print(f"Internal dev questions: {len(dev_indices)}")
    print(f"Hidden dimension: {args.hidden}")
    print(
        f"Batch size (questions): {args.batch_size}"
    )

    # --------------------------------------------------
    # 4. Phase 1: find best epoch using internal dev.
    # --------------------------------------------------
    print(
        "\nPhase 1: choose epoch using internal dev"
    )

    params = initialize(
        args.seed, args.hidden
    )

    best_epoch = train(
        x_train,
        y_train,
        params,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.seed,
        x_dev,
        y_dev
    )

    print(f"\nBest epoch: {best_epoch}")

    # --------------------------------------------------
    # 5. Phase 2: retrain from scratch on full TRAIN.
    # --------------------------------------------------
    full_x = embeddings
    full_y = labels

    flat_full = full_x.reshape(-1, INPUT_DIM)

    full_mean = flat_full.mean(axis=0)

    full_std = np.maximum(
        flat_full.std(axis=0), 1e-6
    )

    full_x = (full_x - full_mean) / full_std

    print("\nPhase 2: retrain on full TRAIN")

    final_params = initialize(
        args.seed, args.hidden
    )

    train(
        full_x,
        full_y,
        final_params,
        best_epoch,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.seed
    )

    # --------------------------------------------------
    # 6. Save inference checkpoint.
    # --------------------------------------------------
    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )

    np.savez(
        args.output,
        **final_params,
        mean=full_mean.astype(np.float32),
        std=full_std.astype(np.float32)
    )

    elapsed = time.perf_counter() - timer

    print(f"\nCheckpoint: {args.output}")
    print(
        f"Checkpoint size: "
        f"{args.output.stat().st_size:,} bytes"
    )
    print(
        f"Training pipeline seconds: {elapsed:.3f}"
    )


if __name__ == "__main__":
    main()
