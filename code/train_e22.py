
"""E22: Disagreement-Focused Weighted BCE MLP.

Train data only.

Architecture:
    1024 -> 32 -> 1, ReLU

Training:
    Seed = 17
    Epochs = 10
    Learning rate = 0.001
    Weight decay = 0.01

E22 weights:
    Original E14-B question weights
    multiplied by 0.25 when there is no answer disagreement.

Checkpoint:
    runs/e22-disagreement-mlp.npz
"""

import json
import time
from pathlib import Path

import numpy as np

from common import load_inputs, parse_answer
from train_e14 import (
    load_training_data,
    initialize,
    train,
    question_weights,
)


SEED = 17
HIDDEN = 32
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 0.01
ALPHA = 0.25
EPS = 1e-6

CHECKPOINT = Path("runs/e22-disagreement-mlp.npz")
REPORT = Path("runs/e22-internal-dev.json")


def detect_disagreement(bank):
    """At least two different parsed final answers."""

    result = np.zeros(len(bank), dtype=bool)

    for i, row in enumerate(bank):
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]

        parsed = [a for a in answers if a is not None]
        result[i] = len(set(parsed)) >= 2

    return result


def predict_scores(x, params):
    flat = x.reshape(-1, 1024)

    hidden = np.maximum(
        flat @ params["w1"] + params["b1"],
        0.0,
    )

    scores = (
        hidden @ params["w2"] + params["b2"]
    ).reshape(-1, 4)

    return scores


def evaluate(name, x, y, disagreement, params):
    scores = predict_scores(x, params)
    selected = np.argmax(scores, axis=1)

    rows = np.arange(len(y))
    ok = y[rows, selected] > 0.5

    counts = y.sum(axis=1)

    masks = {
        "overall": np.ones(len(y), dtype=bool),
        "disagreement": disagreement,
        "single_correct": counts == 1,
        "non_all_correct": counts < 4,
        "disagreement_single_correct": (
            disagreement & (counts == 1)
        ),
    }

    print(f"\n=== {name} ===")

    results = {}

    for label, mask in masks.items():
        total = int(mask.sum())
        correct = int((ok & mask).sum())

        accuracy = (
            100.0 * correct / total
            if total else None
        )

        results[label] = {
            "correct": correct,
            "total": total,
            "accuracy_percent": accuracy,
        }

        if total:
            print(
                f"{label}: {correct}/{total} "
                f"= {accuracy:.4f}%"
            )

    return results


def normalize(x_train, x_other=None):
    flat = x_train.reshape(-1, 1024)

    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), EPS)

    normalized_train = (x_train - mean) / std

    if x_other is None:
        return normalized_train, mean, std

    normalized_other = (x_other - mean) / std

    return normalized_train, normalized_other, mean, std


def fit_model(x, y, weights):
    params = initialize(SEED, HIDDEN)

    train(
        x,
        y,
        weights,
        params,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        seed=SEED,
    )

    return params


def main():
    started = time.perf_counter()

    bank, _ = load_inputs("train")
    x, y, correct_counts = load_training_data()

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if len(bank) != len(x):
        raise ValueError("TRAIN bank size mismatch")

    if x.shape[1:] != (4, 1024):
        raise ValueError(f"Unexpected shape: {x.shape}")

    disagreement = detect_disagreement(bank)

    base_weights = np.asarray(
        question_weights(correct_counts, "weighted"),
        dtype=np.float32,
    )

    e22_weights = base_weights * np.where(
        disagreement, 1.0, ALPHA
    ).astype(np.float32)

    print("=== E22 Dataset ===")
    print(f"Total TRAIN questions: {len(x)}")
    print(f"Disagreement: {int(disagreement.sum())}")
    print(f"No disagreement: {int((~disagreement).sum())}")
    print(f"Alpha: {ALPHA}")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(x))

    n_dev = max(1, int(len(x) * 0.1))
    dev_ids = order[:n_dev]
    train_ids = order[n_dev:]

    x_train, x_dev, mean, std = normalize(
        x[train_ids],
        x[dev_ids],
    )

    y_train = y[train_ids]
    y_dev = y[dev_ids]
    dev_disagreement = disagreement[dev_ids]

    print("\n=== Internal split ===")
    print(f"Train: {len(train_ids)}")
    print(f"Dev: {len(dev_ids)}")
    print(
        "Dev disagreement questions:",
        int(dev_disagreement.sum()),
    )

    print("\n=== Phase 1: Original Weighted BCE ===")

    baseline = fit_model(
        x_train,
        y_train,
        base_weights[train_ids],
    )

    baseline_results = evaluate(
        "E14-B-style baseline",
        x_dev,
        y_dev,
        dev_disagreement,
        baseline,
    )

    print("\n=== Phase 2: Disagreement-Focused BCE ===")

    focused = fit_model(
        x_train,
        y_train,
        e22_weights[train_ids],
    )

    focused_results = evaluate(
        "E22 internal dev",
        x_dev,
        y_dev,
        dev_disagreement,
        focused,
    )

    print("\n=== Internal Dev Comparison ===")

    for metric in (
        "overall",
        "disagreement",
        "single_correct",
        "non_all_correct",
        "disagreement_single_correct",
    ):
        a = baseline_results[metric]
        b = focused_results[metric]

        difference = b["correct"] - a["correct"]

        print(
            f"{metric}: "
            f"baseline={a['correct']}/{a['total']}, "
            f"E22={b['correct']}/{b['total']}, "
            f"delta={difference:+d}"
        )

    # Full TRAIN model: same fixed recipe.
    print("\n=== Phase 3: Full TRAIN E22 ===")

    x_full, full_mean, full_std = normalize(x)

    full_params = fit_model(
        x_full,
        y,
        e22_weights,
    )

    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        CHECKPOINT,
        **full_params,
        mean=full_mean.astype(np.float32),
        std=full_std.astype(np.float32),
    )

    elapsed = time.perf_counter() - started

    report = {
        "experiment": "E22",
        "seed": SEED,
        "epochs": EPOCHS,
        "alpha": ALPHA,
        "internal_train": len(train_ids),
        "internal_dev": len(dev_ids),
        "train_disagreement": int(disagreement.sum()),
        "baseline": baseline_results,
        "e22": focused_results,
        "checkpoint": str(CHECKPOINT),
        "total_seconds": elapsed,
    }

    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== E22 Complete ===")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Checkpoint size: {CHECKPOINT.stat().st_size:,} bytes")
    print(f"Report: {REPORT}")
    print(f"Total training and analysis seconds: {elapsed:.4f}")


if __name__ == "__main__":
    main()
