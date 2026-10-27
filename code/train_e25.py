
"""E25: Multi-Positive Group-Normalized Candidate Selection.

Official TRAIN only.
Internal dev comparison + full TRAIN checkpoint.

Model: 1024 -> 32 -> 1 (ReLU)
Seed: 17
Epochs: 10
Output: runs/e25-group-selection.npz
"""

import json
import time
from pathlib import Path

import numpy as np

from common import load_inputs, parse_answer
from train_e14 import load_training_data, question_weights

SEED = 17
HIDDEN = 32
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 0.01
EPS = 1e-6

CHECKPOINT = Path("runs/e25-group-selection.npz")
REPORT = Path("runs/e25-internal-dev.json")


def detect_disagreement(bank):
    flags = []

    for row in bank:
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]
        valid = [a for a in answers if a is not None]
        flags.append(len(set(valid)) >= 2)

    return np.asarray(flags, dtype=bool)


def initialize(dimension):
    rng = np.random.default_rng(SEED)

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

    scores = (
        hidden @ params["w2"] + params["b2"]
    ).reshape(-1, 4)

    return scores, flat, pre, hidden


def loss_gradient(scores, targets, weights, objective):
    """Return scalar loss and gradient wrt candidate logits."""

    if objective == "bce":
        element = (
            np.logaddexp(0.0, scores)
            - targets * scores
        )
        question_loss = element.mean(axis=1)

        denom = max(float(weights.sum()), EPS)

        loss = float(
            (question_loss * weights).sum() / denom
        )

        sigmoid = 1.0 / (
            1.0 + np.exp(
                -np.clip(scores, -40.0, 40.0)
            )
        )

        gradient = (
            (sigmoid - targets)
            * weights[:, None] / (4.0 * denom)
        )

    elif objective == "group":
        # Stable softmax probabilities over four candidates.
        z = scores.astype(np.float64)
        z -= z.max(axis=1, keepdims=True)

        exp_z = np.exp(z)
        probabilities = (
            exp_z / exp_z.sum(axis=1, keepdims=True)
        )

        positives = targets > 0.5
        correct_count = positives.sum(axis=1)

        # All-correct groups convey no candidate-selection
        # information. Exclude these from the loss.
        informative = (
            (correct_count >= 1)
            & (correct_count < 4)
        )

        positive_mass = (
            probabilities * positives
        ).sum(axis=1)

        safe_mass = np.maximum(positive_mass, 1e-12)

        question_loss = -np.log(safe_mass)

        effective_weights = (
            weights * informative
        )

        denom = max(
            float(effective_weights.sum()), EPS
        )

        loss = float(
            (question_loss * effective_weights).sum()
            / denom
        )

        # Gradient:
        # p_i - 1[y_i=1] * p_i / P(correct group)
        conditional_positive = (
            probabilities * positives
            / safe_mass[:, None]
        )

        gradient = (
            probabilities - conditional_positive
        ) * (effective_weights[:, None] / denom)

    else:
        raise ValueError(f"Unknown objective: {objective}")

    return loss, gradient.astype(np.float32)


def train_model(x, y, weights, objective):
    params = initialize(x.shape[-1])

    first = {
        k: np.zeros_like(v)
        for k, v in params.items()
    }
    second = {
        k: np.zeros_like(v)
        for k, v in params.items()
    }

    rng = np.random.default_rng(SEED)
    step = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(len(x))
        losses = []

        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]

            scores, flat, pre, hidden = forward(
                x[ids], params
            )

            loss, grad_scores = loss_gradient(
                scores,
                y[ids],
                weights[ids],
                objective,
            )
            losses.append(loss)

            grad_out = grad_scores.reshape(-1, 1)

            dw2 = hidden.T @ grad_out
            db2 = grad_out.sum(axis=0)

            dhidden = grad_out @ params["w2"].T
            dpre = dhidden * (pre > 0.0)

            dw1 = flat.T @ dpre
            db1 = dpre.sum(axis=0)

            grads = {
                "w1": dw1,
                "b1": db1,
                "w2": dw2,
                "b2": db2,
            }

            step += 1

            # AdamW
            for key in params:
                g = np.asarray(
                    grads[key], dtype=np.float32
                )

                first[key] = (
                    0.9 * first[key] + 0.1 * g
                )
                second[key] = (
                    0.999 * second[key]
                    + 0.001 * (g * g)
                )

                m_hat = first[key] / (1.0 - 0.9 ** step)
                v_hat = second[key] / (1.0 - 0.999 ** step)

                if key in ("w1", "w2"):
                    params[key] *= (
                        1.0 - LR * WEIGHT_DECAY
                    )

                params[key] -= (
                    LR * m_hat
                    / (np.sqrt(v_hat) + 1e-8)
                ).astype(np.float32)

        print(
            f"{objective.upper():5s} "
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"Loss={np.mean(losses):.5f}"
        )

    return params


def evaluate(name, x, y, disagreement, params):
    scores, _, _, _ = forward(x, params)

    selected = np.argmax(scores, axis=1)
    rows = np.arange(len(y))

    correct = y[rows, selected] > 0.5
    counts = y.sum(axis=1)

    masks = {
        "overall": np.ones(len(y), dtype=bool),
        "disagreement": disagreement,
        "single_correct": counts == 1,
        "disagreement_single_correct": (
            disagreement & (counts == 1)
        ),
        "non_all_correct": counts < 4,
    }

    print(f"\n=== {name} ===")
    results = {}

    for metric, mask in masks.items():
        total = int(mask.sum())
        hits = int((correct & mask).sum())

        accuracy = (
            100.0 * hits / total
            if total else None
        )

        results[metric] = {
            "correct": hits,
            "total": total,
            "accuracy_percent": accuracy,
        }

        if total:
            print(
                f"{metric}: {hits}/{total} "
                f"= {accuracy:.4f}%"
            )

    return results


def normalize(x_train, x_dev=None):
    flat = x_train.reshape(-1, x_train.shape[-1])

    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), EPS)

    normalized_train = (x_train - mean) / std

    if x_dev is None:
        return normalized_train, mean, std

    normalized_dev = (x_dev - mean) / std

    return normalized_train, normalized_dev, mean, std


def main():
    started = time.perf_counter()

    bank, _ = load_inputs("train")
    x, y, correct_counts = load_training_data()

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if x.shape != (len(bank), 4, 1024):
        raise ValueError(f"Invalid input shape: {x.shape}")

    if y.shape != (len(bank), 4):
        raise ValueError(f"Invalid label shape: {y.shape}")

    disagreement = detect_disagreement(bank)

    weights = np.asarray(
        question_weights(correct_counts, "weighted"),
        dtype=np.float32,
    )

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(x))

    n_dev = max(1, int(len(x) * 0.1))
    dev_ids = order[:n_dev]
    train_ids = order[n_dev:]

    print("=== E25 Internal Development ===")
    print(f"Total TRAIN: {len(x)}")
    print(f"Internal train: {len(train_ids)}")
    print(f"Internal dev: {len(dev_ids)}")
    print(
        "Dev disagreement:",
        int(disagreement[dev_ids].sum()),
    )

    x_train, x_dev, mean, std = normalize(
        x[train_ids],
        x[dev_ids],
    )

    y_train = y[train_ids]
    y_dev = y[dev_ids]
    train_weights = weights[train_ids]
    dev_disagreement = disagreement[dev_ids]

    print("\n=== Phase 1: Weighted BCE Baseline ===")

    baseline = train_model(
        x_train,
        y_train,
        train_weights,
        objective="bce",
    )

    baseline_results = evaluate(
        "Weighted BCE baseline",
        x_dev,
        y_dev,
        dev_disagreement,
        baseline,
    )

    print("\n=== Phase 2: E25 Group-Normalized ===")

    group_model = train_model(
        x_train,
        y_train,
        train_weights,
        objective="group",
    )

    group_results = evaluate(
        "E25 internal dev",
        x_dev,
        y_dev,
        dev_disagreement,
        group_model,
    )

    print("\n=== Internal Comparison ===")

    for metric in baseline_results:
        a = baseline_results[metric]
        b = group_results[metric]

        print(
            f"{metric}: "
            f"baseline={a['correct']}/{a['total']} "
            f"E25={b['correct']}/{b['total']} "
            f"delta={b['correct'] - a['correct']:+d}"
        )

    print("\n=== Phase 3: Full TRAIN E25 ===")

    x_full, full_mean, full_std = normalize(x)

    full_model = train_model(
        x_full,
        y,
        weights,
        objective="group",
    )

    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        CHECKPOINT,
        **full_model,
        mean=full_mean.astype(np.float32),
        std=full_std.astype(np.float32),
    )

    elapsed = time.perf_counter() - started

    report = {
        "experiment": "E25",
        "objective": "multi-positive group-normalized loss",
        "seed": SEED,
        "epochs": EPOCHS,
        "internal_train": len(train_ids),
        "internal_dev": len(dev_ids),
        "baseline": baseline_results,
        "e25": group_results,
        "checkpoint": str(CHECKPOINT),
        "training_and_analysis_seconds": elapsed,
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== E25 Complete ===")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Checkpoint size: {CHECKPOINT.stat().st_size:,} bytes")
    print(f"Report: {REPORT}")
    print(f"Total seconds: {elapsed:.4f}")


if __name__ == "__main__":
    main()
