
"""E23: Dual-Scorer Fusion.

Train-only model selection.
Base models:
  E14-B: weighted BCE
  E22: disagreement-focused weighted BCE

The final checkpoint packages both full-TRAIN models.
"""

import json
import time
from collections import defaultdict
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
TEMPERATURE = 1.0
EPS = 1e-6

BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]

E14_CHECKPOINT = Path("runs/e14b-weighted-mlp.npz")
E22_CHECKPOINT = Path("runs/e22-disagreement-mlp.npz")
OUTPUT = Path("runs/e23-dual-scorer.npz")
REPORT = Path("runs/e23-internal-dev.json")

PARAM_KEYS = ("w1", "b1", "w2", "b2", "mean", "std")


def get_answers(bank):
    return [
        [parse_answer(c["text"]) for c in row["candidates"]]
        for row in bank
    ]


def disagreement_flags(answers):
    flags = []

    for row in answers:
        parsed = [a for a in row if a is not None]
        flags.append(len(set(parsed)) >= 2)

    return np.asarray(flags, dtype=bool)


def fit(x, y, weights):
    params = initialize(SEED, HIDDEN)

    train(
        x, y, weights, params,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        seed=SEED,
    )

    return params


def score(x, params):
    flat = x.reshape(-1, 1024)
    hidden = np.maximum(
        flat @ params["w1"] + params["b1"],
        0.0,
    )
    return (hidden @ params["w2"] + params["b2"]).reshape(-1, 4)


def softmax_rows(scores):
    z = scores.astype(np.float64) / TEMPERATURE
    z -= z.max(axis=1, keepdims=True)

    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def aggregate_choice(probs, answers, raw_scores):
    selected = []

    for i, row in enumerate(answers):
        groups = defaultdict(list)

        for j, answer in enumerate(row):
            key = (
                ("answer", answer)
                if answer is not None
                else ("unparsed", j)
            )
            groups[key].append(j)

        best_group = max(
            groups.values(),
            key=lambda indices: (
                sum(float(probs[i, j]) for j in indices),
                max(float(raw_scores[i, j]) for j in indices),
            ),
        )

        chosen = max(
            best_group,
            key=lambda j: (
                float(probs[i, j]),
                float(raw_scores[i, j]),
            ),
        )
        selected.append(chosen)

    return np.asarray(selected, dtype=int)


def evaluate(choices, labels, disagreement):
    rows = np.arange(len(labels))
    success = labels[rows, choices] > 0.5
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

    result = {}

    for name, mask in masks.items():
        hits = int((success & mask).sum())
        n = int(mask.sum())
        result[name] = {
            "correct": hits,
            "total": n,
            "accuracy_percent": 100 * hits / n if n else None,
        }

    return result


def show(label, metrics):
    d = metrics["disagreement"]
    s = metrics["single_correct"]
    o = metrics["overall"]

    print(
        f"{label:<22} "
        f"Disagreement {d['correct']}/{d['total']} "
        f"({d['accuracy_percent']:.2f}%) | "
        f"Single {s['correct']}/{s['total']} "
        f"({s['accuracy_percent']:.2f}%) | "
        f"Overall {o['correct']}/{o['total']} "
        f"({o['accuracy_percent']:.2f}%)"
    )


def package_models(beta):
    package = {}

    for prefix, path in [
        ("e14", E14_CHECKPOINT),
        ("e22", E22_CHECKPOINT),
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as model:
            for key in PARAM_KEYS:
                package[f"{prefix}_{key}"] = model[key].copy()

    package["beta"] = np.asarray([beta], dtype=np.float32)
    package["temperature"] = np.asarray(
        [TEMPERATURE], dtype=np.float32
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT, **package)

    print(f"\nFinal checkpoint: {OUTPUT}")
    print(f"Checkpoint size: {OUTPUT.stat().st_size:,} bytes")


def main():
    start = time.perf_counter()

    bank, _ = load_inputs("train")
    x, y, correct_counts = load_training_data()

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if len(bank) != len(x):
        raise ValueError("TRAIN data alignment mismatch")

    answers = get_answers(bank)
    disagreement = disagreement_flags(answers)

    base_weights = np.asarray(
        question_weights(correct_counts, "weighted"),
        dtype=np.float32,
    )
    focus_weights = base_weights * np.where(
        disagreement, 1.0, ALPHA
    ).astype(np.float32)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(x))

    n_dev = max(1, int(len(x) * 0.1))
    dev_ids = order[:n_dev]
    train_ids = order[n_dev:]

    x_train = x[train_ids]
    x_dev = x[dev_ids]
    y_train = y[train_ids]
    y_dev = y[dev_ids]

    dev_answers = [answers[int(i)] for i in dev_ids]
    dev_disagreement = disagreement[dev_ids]

    flat = x_train.reshape(-1, 1024)
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), EPS)

    x_train = (x_train - mean) / std
    x_dev = (x_dev - mean) / std

    print("=== E23 Internal Dev ===")
    print(f"Internal train: {len(train_ids)}")
    print(f"Internal dev: {len(dev_ids)}")
    print(f"Dev disagreement: {int(dev_disagreement.sum())}")

    print("\n=== Train E14-style scorer ===")
    p14 = fit(
        x_train, y_train, base_weights[train_ids]
    )

    print("\n=== Train E22-style scorer ===")
    p22 = fit(
        x_train, y_train, focus_weights[train_ids]
    )

    s14 = score(x_dev, p14)
    s22 = score(x_dev, p22)

    prob14 = softmax_rows(s14)
    prob22 = softmax_rows(s22)

    print("\n=== Internal Dev Comparison ===")

    results = {}

    raw14 = evaluate(
        np.argmax(s14, axis=1),
        y_dev, dev_disagreement
    )
    raw22 = evaluate(
        np.argmax(s22, axis=1),
        y_dev, dev_disagreement
    )

    show("E14 raw scorer", raw14)
    show("E22 raw scorer", raw22)

    results["E14_raw"] = raw14
    results["E22_raw"] = raw22

    for beta in BETAS:
        probs = (1 - beta) * prob14 + beta * prob22

        # Aggregate equal final answers.
        choices = aggregate_choice(
            probs, dev_answers,
            (1 - beta) * s14 + beta * s22,
        )

        metrics = evaluate(
            choices, y_dev, dev_disagreement
        )

        show(f"Fusion beta={beta}", metrics)
        results[f"beta_{beta}"] = metrics

    # Primary: disagreement correct count.
    # Secondary: single-correct correct count.
    # Then overall correct count; prefer smaller beta in ties.
    chosen_beta = max(
        BETAS,
        key=lambda beta: (
            results[f"beta_{beta}"]["disagreement"]["correct"],
            results[f"beta_{beta}"]["single_correct"]["correct"],
            results[f"beta_{beta}"]["overall"]["correct"],
            -beta,
        ),
    )

    print(f"\nSelected beta: {chosen_beta}")

    baseline_hits = results["beta_0.0"]["disagreement"]["correct"]
    chosen_hits = results[
        f"beta_{chosen_beta}"
    ]["disagreement"]["correct"]

    if chosen_hits <= baseline_hits:
        print(
            "WARNING: Fusion did not improve disagreement "
            "accuracy over E19-style aggregation internally."
        )

    package_models(chosen_beta)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps({
            "experiment": "E23",
            "split": "TRAIN internal dev",
            "seed": SEED,
            "epochs": EPOCHS,
            "alpha": ALPHA,
            "temperature": TEMPERATURE,
            "selected_beta": chosen_beta,
            "results": results,
            "checkpoint": str(OUTPUT),
            "total_seconds": time.perf_counter() - start,
        }, indent=2),
        encoding="utf-8",
    )

    print(f"Report: {REPORT}")
    print(f"Total seconds: {time.perf_counter() - start:.4f}")


if __name__ == "__main__":
    main()
