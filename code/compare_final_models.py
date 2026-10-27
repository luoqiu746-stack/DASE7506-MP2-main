
"""E20: Fair internal-dev comparison of E16-A and E19-B.

Uses only official TRAIN data.
Both methods share the same held-out Weighted BCE scorer.
No official validation or test labels are accessed.
No submitted checkpoint is modified.
"""

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import load_inputs, parse_answer
from train_e14 import (
    load_training_data,
    initialize,
    question_weights,
    train,
)

SEED = 17
EPS = 1e-6
VOTE_WEIGHT = 0.5
CONFIDENCE_THRESHOLD = 1.0
TEMPERATURE = 1.0
OUTPUT = Path("runs/e20-final-model-comparison.json")


def get_scores(x, params):
    flat = x.reshape(-1, 1024)
    hidden = np.maximum(
        flat @ params["w1"] + params["b1"], 0.0
    )
    return (hidden @ params["w2"] + params["b2"]).reshape(-1, 4)


def get_answers(bank, indices):
    return [
        [
            parse_answer(candidate["text"])
            for candidate in bank[int(i)]["candidates"]
        ]
        for i in indices
    ]


def get_votes(answers):
    votes = np.zeros((len(answers), 4), dtype=np.float32)

    for i, row in enumerate(answers):
        counts = Counter(a for a in row if a is not None)

        for j, answer in enumerate(row):
            if answer is not None:
                votes[i, j] = counts[answer] / 4.0

    return votes


def predict_e16a(scores, answers, votes):
    """Exact E16-A agreement-gate decision rule."""
    result = []

    for i, row in enumerate(answers):
        s = scores[i]
        raw_choice = int(np.argmax(s))

        sorted_scores = np.sort(s)
        gap = float(sorted_scores[-1] - sorted_scores[-2])
        scale = max(float(s.std()), EPS)
        normalized_gap = gap / scale

        counts = Counter(a for a in row if a is not None)
        top_answer = row[raw_choice]

        top_votes = (
            counts[top_answer] if top_answer is not None else 0
        )
        max_votes = max(counts.values(), default=0)

        confident_minority = (
            top_votes < max_votes
            and normalized_gap >= CONFIDENCE_THRESHOLD
        )

        weight = 0.0 if confident_minority else VOTE_WEIGHT
        result.append(int(np.argmax(s + weight * votes[i])))

    return np.asarray(result, dtype=int)


def predict_e19b(scores, answers):
    """Exact E19-B answer-level probability aggregation."""
    result = []

    for logits, row in zip(scores, answers):
        z = logits.astype(np.float64) / TEMPERATURE
        z -= z.max()

        probabilities = np.exp(z)
        probabilities /= probabilities.sum()

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
                sum(probabilities[j] for j in indices),
                max(float(logits[j]) for j in indices),
            ),
        )

        chosen = max(
            best_group,
            key=lambda j: float(logits[j]),
        )

        result.append(int(chosen))

    return np.asarray(result, dtype=int)


def evaluate(name, choices, labels):
    rows = np.arange(len(labels))
    ok = labels[rows, choices] > 0.5
    counts = labels.sum(axis=1).astype(int)

    overall = int(ok.sum())
    print(
        f"\n{name}: {overall}/{len(labels)} "
        f"= {100 * ok.mean():.4f}%"
    )

    groups = {}

    for k in range(1, 5):
        mask = counts == k
        n = int(mask.sum())
        hits = int((mask & ok).sum())

        groups[str(k)] = {
            "correct": hits,
            "total": n,
        }

        print(
            f"  {k} correct candidates: "
            f"{hits}/{n} = {100 * hits / n:.2f}%"
            if n else f"  {k} correct candidates: 0 questions"
        )

    return ok, {
        "correct": overall,
        "total": len(labels),
        "accuracy_percent": 100 * overall / len(labels),
        "by_correct_count": groups,
    }


def main():
    start = time.perf_counter()

    bank, _ = load_inputs("train")
    x, y, correct_counts = load_training_data()

    if len(bank) != len(x):
        raise ValueError("Train bank size mismatch")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(x))
    n_dev = max(1, int(len(x) * 0.1))

    dev_ids = order[:n_dev]
    train_ids = order[n_dev:]

    x_train = x[train_ids]
    y_train = y[train_ids]

    x_dev = x[dev_ids]
    y_dev = y[dev_ids]

    weights = question_weights(correct_counts, "weighted")
    w_train = weights[train_ids]

    flat_train = x_train.reshape(-1, 1024)
    mean = flat_train.mean(axis=0)
    std = np.maximum(flat_train.std(axis=0), EPS)

    x_train = (x_train - mean) / std
    x_dev = (x_dev - mean) / std

    print("=== Phase 1. Train held-out scorer ===")
    print(f"Internal train: {len(train_ids)}")
    print(f"Internal dev: {len(dev_ids)}")
    print("Seed: 17; epochs: 10")

    params = initialize(SEED, 32)

    train(
        x_train,
        y_train,
        w_train,
        params,
        epochs=10,
        batch_size=64,
        lr=0.001,
        weight_decay=0.01,
        seed=SEED,
    )

    print("\n=== Phase 2. Compare methods ===")

    scores = get_scores(x_dev, params)
    answers = get_answers(bank, dev_ids)
    votes = get_votes(answers)

    raw = np.argmax(scores, axis=1)
    fixed = np.argmax(scores + VOTE_WEIGHT * votes, axis=1)

    e16 = predict_e16a(scores, answers, votes)
    e19 = predict_e19b(scores, answers)

    summary = {}

    for name, choices in [
        ("Raw scorer", raw),
        ("Fixed voting", fixed),
        ("E16-A", e16),
        ("E19-B", e19),
    ]:
        _, summary[name] = evaluate(name, choices, y_dev)

    rows = np.arange(len(y_dev))
    e16_ok = y_dev[rows, e16] > 0.5
    e19_ok = y_dev[rows, e19] > 0.5

    print("\n=== Phase 3. Pairwise comparison ===")

    both_correct = int((e16_ok & e19_ok).sum())
    e16_only = int((e16_ok & ~e19_ok).sum())
    e19_only = int((~e16_ok & e19_ok).sum())
    both_wrong = int((~e16_ok & ~e19_ok).sum())

    print("Both correct:", both_correct)
    print("E16-A only correct:", e16_only)
    print("E19-B only correct:", e19_only)
    print("Both wrong:", both_wrong)
    print("Different selected candidate:", int((e16 != e19).sum()))

    print("\n=== Phase 4. Save summary ===")

    report = {
        "split": "TRAIN internal dev",
        "seed": SEED,
        "internal_train_questions": len(train_ids),
        "internal_dev_questions": len(dev_ids),
        "scorer_epochs": 10,
        "e16_threshold": CONFIDENCE_THRESHOLD,
        "e19_temperature": TEMPERATURE,
        "results": summary,
        "comparison": {
            "both_correct": both_correct,
            "e16_only_correct": e16_only,
            "e19_only_correct": e19_only,
            "both_wrong": both_wrong,
            "different_selected_candidate": int(
                (e16 != e19).sum()
            ),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {OUTPUT}")
    print(
        f"Total analysis seconds: "
        f"{time.perf_counter() - start:.3f}"
    )


if __name__ == "__main__":
    main()
