
"""E21: Full-TRAIN Weighted BCE + Pairwise Ranking.

Model: 1024 -> 32 -> 1 (ReLU)
Seed: 17
Epochs: 10
Loss: Weighted BCE + 0.3 * Pairwise Ranking
Data: All official TRAIN questions

Checkpoint: runs/e21-bce-ranking.npz
"""

import time
from pathlib import Path

import numpy as np

from train_e14 import load_training_data, question_weights


SEED = 17
HIDDEN = 32
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 0.01
RANK_WEIGHT = 0.3
EPS = 1e-6

OUTPUT = Path("runs/e21-bce-ranking.npz")


def init_params(rng, dimension):
    return {
        "w1": (
            rng.standard_normal((dimension, HIDDEN))
            * np.sqrt(2.0 / dimension)
        ).astype(np.float32),
        "b1": np.zeros(HIDDEN, dtype=np.float32),
        "w2": (
            rng.standard_normal((HIDDEN, 1))
            * np.sqrt(2.0 / HIDDEN)
        ).astype(np.float32),
        "b2": np.zeros(1, dtype=np.float32),
    }


def forward(x, params):
    flat = x.reshape(-1, x.shape[-1])

    pre = flat @ params["w1"] + params["b1"]
    hidden = np.maximum(pre, 0.0)

    logits = (
        hidden @ params["w2"] + params["b2"]
    ).reshape(-1, 4)

    return logits, flat, pre, hidden


def loss_and_grads(x, y, weights, params, rank_weight):
    logits, flat, pre, hidden = forward(x, params)

    # ----- Weighted BCE -----
    bce_element = np.logaddexp(0.0, logits) - y * logits
    per_question_bce = bce_element.mean(axis=1)

    denom = max(float(weights.sum()), EPS)

    bce_loss = float(
        (per_question_bce * weights).sum() / denom
    )

    sigmoid = 1.0 / (
        1.0 + np.exp(-np.clip(logits, -40.0, 40.0))
    )

    grad_logits = (
        (sigmoid - y)
        * (weights[:, None] / (4.0 * denom))
    ).astype(np.float32)

    # ----- Pairwise Ranking Loss -----
    rank_grad = np.zeros_like(grad_logits)
    rank_loss_sum = 0.0
    rank_weight_sum = 0.0

    for q in range(len(x)):
        good = np.flatnonzero(y[q] > 0.5)
        bad = np.flatnonzero(y[q] < 0.5)

        if not len(good) or not len(bad):
            continue

        pairs = len(good) * len(bad)
        pair_weight = float(weights[q]) / pairs

        for i in good:
            for j in bad:
                diff = float(logits[q, i] - logits[q, j])

                rank_loss_sum += pair_weight * float(
                    np.logaddexp(0.0, -diff)
                )

                p = 1.0 / (
                    1.0 + np.exp(np.clip(diff, -40.0, 40.0))
                )

                rank_grad[q, i] -= pair_weight * p
                rank_grad[q, j] += pair_weight * p

        rank_weight_sum += float(weights[q])

    if rank_weight_sum > 0:
        rank_loss = rank_loss_sum / rank_weight_sum
        rank_grad /= rank_weight_sum
    else:
        rank_loss = 0.0

    grad_logits += rank_weight * rank_grad

    # ----- Backpropagation -----
    grad_out = grad_logits.reshape(-1, 1)

    dw2 = hidden.T @ grad_out
    db2 = grad_out.sum(axis=0)

    dh = grad_out @ params["w2"].T
    dpre = dh * (pre > 0.0)

    dw1 = flat.T @ dpre
    db1 = dpre.sum(axis=0)

    grads = {
        "w1": dw1.astype(np.float32),
        "b1": db1.astype(np.float32),
        "w2": dw2.astype(np.float32),
        "b2": db2.astype(np.float32),
    }

    return bce_loss, rank_loss, grads


def train_model(x, y, weights):
    rng = np.random.default_rng(SEED)
    params = init_params(rng, x.shape[-1])

    m = {
        k: np.zeros_like(v)
        for k, v in params.items()
    }
    v = {
        k: np.zeros_like(v)
        for k, v in params.items()
    }

    step = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(len(x))
        epoch_bce = []
        epoch_rank = []

        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]

            bce, rank, grads = loss_and_grads(
                x[ids],
                y[ids],
                weights[ids],
                params,
                RANK_WEIGHT,
            )

            epoch_bce.append(bce)
            epoch_rank.append(rank)
            step += 1

            # AdamW optimizer
            for key in params:
                grad = grads[key]

                m[key] = (
                    0.9 * m[key] + 0.1 * grad
                )
                v[key] = (
                    0.999 * v[key]
                    + 0.001 * (grad * grad)
                )

                m_hat = m[key] / (1.0 - 0.9 ** step)
                v_hat = v[key] / (1.0 - 0.999 ** step)

                if key.startswith("w"):
                    params[key] *= (
                        1.0 - LR * WEIGHT_DECAY
                    )

                params[key] -= (
                    LR * m_hat
                    / (np.sqrt(v_hat) + 1e-8)
                ).astype(np.float32)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"BCE={np.mean(epoch_bce):.4f} | "
            f"Ranking={np.mean(epoch_rank):.4f}"
        )

    return params


def main():
    started = time.perf_counter()

    # Load all official TRAIN data
    x, y, correct_counts = load_training_data()

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if x.ndim != 3 or x.shape[1:] != (4, 1024):
        raise ValueError(
            f"Unexpected embeddings shape: {x.shape}"
        )

    if y.shape != (len(x), 4):
        raise ValueError(
            f"Unexpected labels shape: {y.shape}"
        )

    weights = np.asarray(
        question_weights(
            correct_counts,
            "weighted",
        ),
        dtype=np.float32,
    )

    print("\n=== E21: Full TRAIN ===")
    print(f"Training questions: {len(x)}")
    print(f"Seed: {SEED}")
    print(f"Epochs: {EPOCHS}")
    print(f"Hidden dimension: {HIDDEN}")
    print(f"Learning rate: {LR}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Ranking weight: {RANK_WEIGHT}")

    # Normalize using FULL TRAIN only
    flat = x.reshape(-1, 1024)

    mean = flat.mean(axis=0)
    std = np.maximum(
        flat.std(axis=0),
        EPS,
    )

    x_normalized = (
        x - mean
    ) / std

    # Train final E21 model
    params = train_model(
        x_normalized,
        y,
        weights,
    )

    # Save checkpoint
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        OUTPUT,
        **params,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        rank_weight=np.array(
            [RANK_WEIGHT],
            dtype=np.float32,
        ),
    )

    elapsed = time.perf_counter() - started

    print("\n=== E21 Training Complete ===")
    print(f"Checkpoint: {OUTPUT}")
    print(
        f"Checkpoint size: "
        f"{OUTPUT.stat().st_size:,} bytes"
    )
    print(f"Training seconds: {elapsed:.4f}")


if __name__ == "__main__":
    main()
