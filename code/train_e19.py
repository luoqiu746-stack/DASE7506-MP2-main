
"""E19-A: Train-only temperature selection for answer aggregation.

Uses official TRAIN only for choosing temperature.
Final checkpoint packages existing E14-B weights and temperature.
"""

import argparse
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


TEMPERATURES = [0.25, 0.5, 1.0, 2.0, 4.0]
VOTE_WEIGHT = 0.5
EPS = 1e-6


def score_mlp(x, params):
    flat = x.reshape(-1, 1024)
    hidden = np.maximum(
        flat @ params["w1"] + params["b1"], 0.0
    )
    return (hidden @ params["w2"] + params["b2"]).reshape(-1, 4)


def get_answers(bank):
    return [
        [parse_answer(c["text"]) for c in row["candidates"]]
        for row in bank
    ]


def vote_scores(answers):
    from collections import Counter

    result = np.zeros((len(answers), 4), dtype=np.float32)

    for i, row in enumerate(answers):
        counts = Counter(a for a in row if a is not None)
        for j, answer in enumerate(row):
            if answer is not None:
                result[i, j] = counts[answer] / 4.0

    return result


def aggregate_choices(scores, answers, temperature):
    """Sum softmax probabilities for candidates sharing an answer."""
    selected = []

    for logits, row_answers in zip(scores, answers):
        z = logits.astype(np.float64) / temperature
        z -= z.max()

        probabilities = np.exp(z)
        probabilities /= probabilities.sum()

        groups = defaultdict(list)

        for j, answer in enumerate(row_answers):
            # An unparsed answer is treated as its own candidate,
            # not grouped with other unparsed candidates.
            key = ("answer", answer) if answer is not None else (
                "unparsed", j
            )
            groups[key].append(j)

        best_group = max(
            groups.values(),
            key=lambda idx: (
                sum(probabilities[j] for j in idx),
                max(logits[j] for j in idx),
            ),
        )

        best_candidate = max(
            best_group,
            key=lambda j: float(logits[j]),
        )
        selected.append(best_candidate)

    return np.asarray(selected, dtype=int)


def accuracy(labels, choices, mask=None):
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)

    indices = np.flatnonzero(mask)

    if len(indices) == 0:
        return float("nan"), 0, 0

    hits = int(labels[indices, choices[indices]].sum())

    return hits / len(indices), hits, len(indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("runs/e14b-weighted-mlp.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/e19-answer-aggregation.npz"),
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    bank, _ = load_inputs("train")
    x, labels, correct_counts = load_training_data()

    if len(bank) != len(x):
        raise ValueError("Train bank alignment error")

    answers = get_answers(bank)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(x))

    n_dev = max(1, int(len(x) * 0.1))
    dev_ids = order[:n_dev]
    train_ids = order[n_dev:]

    x_train = x[train_ids]
    x_dev = x[dev_ids]

    y_train = labels[train_ids]
    y_dev = labels[dev_ids]

    weights = question_weights(correct_counts, "weighted")
    w_train = weights[train_ids]

    flat = x_train.reshape(-1, 1024)
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), EPS)

    x_train = (x_train - mean) / std
    x_dev = (x_dev - mean) / std

    print("\n=== Phase 1: held-out TRAIN model ===")
    print(f"Internal train questions: {len(train_ids)}")
    print(f"Internal dev questions: {len(dev_ids)}")

    params = initialize(args.seed, 32)

    # Match the previously selected E14-B training recipe.
    train(
        x_train,
        y_train,
        w_train,
        params,
        epochs=10,
        batch_size=64,
        lr=0.001,
        weight_decay=0.01,
        seed=args.seed,
    )

    scores = score_mlp(x_dev, params)
    dev_answers = [answers[int(i)] for i in dev_ids]
    hard_mask = y_dev.sum(axis=1) == 1

    print("\n=== Phase 2: internal dev comparison ===")

    raw_choices = np.argmax(scores, axis=1)
    raw_acc, raw_hits, _ = accuracy(y_dev, raw_choices)

    votes = vote_scores(dev_answers)
    fixed_choices = np.argmax(
        scores + VOTE_WEIGHT * votes, axis=1
    )
    fixed_acc, fixed_hits, _ = accuracy(y_dev, fixed_choices)

    print(
        f"Raw scorer: {raw_hits}/{len(y_dev)} "
        f"= {100 * raw_acc:.4f}%"
    )
    print(
        f"Fixed voting (0.5): {fixed_hits}/{len(y_dev)} "
        f"= {100 * fixed_acc:.4f}%"
    )

    print("\nTemperature | Correct | Accuracy | 1-correct accuracy")

    results = []

    for temperature in TEMPERATURES:
        choices = aggregate_choices(
            scores, dev_answers, temperature
        )
        acc, hits, total = accuracy(y_dev, choices)
        hard_acc, hard_hits, hard_total = accuracy(
            y_dev, choices, hard_mask
        )

        print(
            f"T={temperature:<4} | "
            f"{hits}/{total} | "
            f"{100 * acc:.4f}% | "
            f"{hard_hits}/{hard_total} "
            f"= {100 * hard_acc:.2f}%"
        )

        results.append((temperature, hits, hard_hits))

    # Select using overall internal dev correct count.
    # If tied, prefer temperature closest to 1.
    best_t, best_hits, _ = max(
        results,
        key=lambda item: (
            item[1],
            -abs(np.log(item[0])),
        ),
    )

    print(f"\nSelected temperature: {best_t}")
    print(f"Best aggregation: {best_hits}/{len(y_dev)}")
    print(f"Fixed voting: {fixed_hits}/{len(y_dev)}")

    if best_hits <= fixed_hits:
        print(
            "\nWARNING: Aggregation did not outperform fixed "
            "voting on internal dev. Treat E19 as exploratory; "
            "there is no strong reason to replace E16-A."
        )

    print("\n=== Phase 3: package full TRAIN checkpoint ===")

    if not args.base_checkpoint.exists():
        raise FileNotFoundError(args.base_checkpoint)

    package = {}

    with np.load(
        args.base_checkpoint,
        allow_pickle=False,
    ) as model:
        for key in ("w1", "b1", "w2", "b2", "mean", "std"):
            package[key] = model[key].copy()

    package["temperature"] = np.asarray(
        [best_t], dtype=np.float32
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **package)

    print(f"Saved: {args.output}")
    print(
        f"Checkpoint size: "
        f"{args.output.stat().st_size:,} bytes"
    )


if __name__ == "__main__":
    main()
