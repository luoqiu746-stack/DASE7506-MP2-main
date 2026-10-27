
"""E24: Question-Relative Embedding Scorer.

Original input: 1024-dimensional candidate embedding.
New input: [candidate embedding, candidate - question mean].
Architecture: 2048 -> 32 -> 1 (ReLU).

Uses TRAIN only for fitting and internal model comparison.
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

CHECKPOINT = Path("runs/e24-relative-mlp.npz")
REPORT = Path("runs/e24-internal-dev.json")


def make_features(x, relative):
    x = np.asarray(x, dtype=np.float32)

    if not relative:
        return x.copy()

    question_mean = x.mean(axis=1, keepdims=True)
    difference = x - question_mean

    return np.concatenate(
        [x, difference], axis=-1
    ).astype(np.float32)


def normalize_train(x):
    dimension = x.shape[-1]
    flat = x.reshape(-1, dimension)

    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), EPS)

    return (x - mean) / std, mean, std


def normalize_other(x, mean, std):
    return (x - mean) / std


def init_model(dimension, seed):
    rng = np.random.default_rng(seed)

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


def train_model(x, y, weights, seed=SEED):
    rng = np.random.default_rng(seed)
    params = init_model(x.shape[-1], seed)

    first = {k: np.zeros_like(v) for k, v in params.items()}
    second = {k: np.zeros_like(v) for k, v in params.items()}

    update_step = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(len(x))
        losses = []

        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]

            logits, flat, pre, hidden = forward(
                x[ids], params
            )

            target = y[ids]
            question_weights_batch = weights[ids]

            denominator = max(
                float(question_weights_batch.sum()), EPS
            )

            bce = (
                np.logaddexp(0.0, logits)
                - target * logits
            ).mean(axis=1)

            batch_loss = float(
                (bce * question_weights_batch).sum()
                / denominator
            )
            losses.append(batch_loss)

            sigmoid = 1.0 / (
                1.0 + np.exp(
                    -np.clip(logits, -40.0, 40.0)
                )
            )

            grad_logits = (
                (sigmoid - target)
                * (
                    question_weights_batch[:, None]
                    / (4.0 * denominator)
                )
            )

            grad_out = grad_logits.reshape(-1, 1)

            dw2 = hidden.T @ grad_out
            db2 = grad_out.sum(axis=0)

            dh = grad_out @ params["w2"].T
            dpre = dh * (pre > 0.0)

            dw1 = flat.T @ dpre
            db1 = dpre.sum(axis=0)

            grads = {
                "w1": dw1,
                "b1": db1,
                "w2": dw2,
                "b2": db2,
            }

            update_step += 1

            for key in params:
                grad = np.asarray(
                    grads[key], dtype=np.float32
                )

                first[key] = (
                    0.9 * first[key] + 0.1 * grad
                )
                second[key] = (
                    0.999 * second[key]
                    + 0.001 * grad * grad
                )

                m_hat = (
                    first[key]
                    / (1.0 - 0.9 ** update_step)
                )
                v_hat = (
                    second[key]
                    / (1.0 - 0.999 ** update_step)
                )

                if key in ("w1", "w2"):
                    params[key] *= (
                        1.0 - LR * WEIGHT_DECAY
                    )

                params[key] -= (
                    LR * m_hat
                    / (np.sqrt(v_hat) + 1e-8)
                ).astype(np.float32)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"BCE={np.mean(losses):.5f}"
        )

    return params


def detect_disagreement(bank):
    flags = []

    for row in bank:
        parsed = [
            parse_answer(c["text"])
            for c in row["candidates"]
        ]
        parsed = [a for a in parsed if a is not None]
        flags.append(len(set(parsed)) >= 2)

    return np.asarray(flags, dtype=bool)


def evaluate(name, x, labels, disagreement, params):
    scores, _, _, _ = forward(x, params)

    choices = scores.argmax(axis=1)
    indices = np.arange(len(labels))

    correct = labels[indices, choices] > 0.5
    correct_counts = labels.sum(axis=1)

    masks = {
        "overall": np.ones(len(labels), dtype=bool),
        "disagreement": disagreement,
        "single_correct": correct_counts == 1,
        "disagreement_single_correct": (
            disagreement & (correct_counts == 1)
        ),
        "non_all_correct": correct_counts < 4,
    }

    print(f"\n=== {name} ===")

    results = {}

    for label, mask in masks.items():
        total = int(mask.sum())
        hits = int((correct & mask).sum())

        accuracy = (
            100.0 * hits / total if total else None
        )

        results[label] = {
            "correct": hits,
            "total": total,
            "accuracy_percent": accuracy,
        }

        if total:
            print(
                f"{label}: {hits}/{total} "
                f"= {accuracy:.4f}%"
            )

    return results


def main():
    start = time.perf_counter()

    bank, _ = load_inputs("train")
    x, y, correct_counts = load_training_data()

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if x.shape != (len(bank), 4, 1024):
        raise ValueError(f"Unexpected x shape: {x.shape}")

    weights = np.asarray(
        question_weights(correct_counts, "weighted"),
        dtype=np.float32,
    )

    disagreement = detect_disagreement(bank)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(x))

    n_dev = max(1, int(len(x) * 0.1))
    dev_ids = order[:n_dev]
    train_ids = order[n_dev:]

    print("=== E24 Internal Comparison ===")
    print(f"Total TRAIN questions: {len(x)}")
    print(f"Internal train: {len(train_ids)}")
    print(f"Internal dev: {len(dev_ids)}")
    print(f"Dev disagreement: {disagreement[dev_ids].sum()}")

    results = {}

    for name, use_relative in (
        ("baseline", False),
        ("E24", True),
    ):
        print(f"\n=== Train {name} ===")

        features = make_features(x, use_relative)

        x_train, mean, std = normalize_train(
            features[train_ids]
        )
        x_dev = normalize_other(
            features[dev_ids], mean, std
        )

        params = train_model(
            x_train,
            y[train_ids],
            weights[train_ids],
        )

        results[name] = evaluate(
            name,
            x_dev,
            y[dev_ids],
            disagreement[dev_ids],
            params,
        )

    print("\n=== Internal Dev Comparison ===")

    for metric in results["baseline"]:
        a = results["baseline"][metric]
        b = results["E24"][metric]

        print(
            f"{metric}: "
            f"baseline={a['correct']}/{a['total']} "
            f"E24={b['correct']}/{b['total']} "
            f"delta={b['correct'] - a['correct']:+d}"
        )

    print("\n=== Full TRAIN E24 ===")

    full_features = make_features(x, True)

    full_x, mean, std = normalize_train(
        full_features
    )

    full_params = train_model(
        full_x, y, weights
    )

    CHECKPOINT.parent.mkdir(
        parents=True, exist_ok=True
    )

    np.savez(
        CHECKPOINT,
        **full_params,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )

    elapsed = time.perf_counter() - start

    report = {
        "experiment": "E24",
        "model": "Question-Relative Embedding MLP",
        "seed": SEED,
        "epochs": EPOCHS,
        "input_dimensions": 2048,
        "internal_train_questions": len(train_ids),
        "internal_dev_questions": len(dev_ids),
        "results": results,
        "checkpoint": str(CHECKPOINT),
        "training_and_analysis_seconds": elapsed,
    }

    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== E24 Complete ===")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Checkpoint size: {CHECKPOINT.stat().st_size:,} bytes")
    print(f"Report: {REPORT}")
    print(f"Total seconds: {elapsed:.4f}")


if __name__ == "__main__":
    main()
