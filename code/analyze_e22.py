
"""E22-A: Hard Validation Audit.

Compare existing models on answer-disagreement questions.

Models:
    E14-B: Weighted BCE
    E16-A: Agreement-Gated Voting
    E19-B: Answer-Level Aggregation
    E21: BCE + Ranking

Read-only experiment. Does not train or modify checkpoints.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer


PREDICTION_FILES = {
    "E14-B": "runs/e14b-validation-predictions.json",
    "E16-A": "runs/e16a-validation-predictions.json",
    "E19-B": "runs/e19-validation-predictions.json",
    "E21": "runs/e21-validation-predictions.json",
    "E22": "runs/e22-validation-predictions.json",
    "E24": "runs/e24-validation-predictions.json",
    "E25": "runs/e25-validation-predictions.json",
}

OUTPUT = Path("runs/e22-hard-validation-analysis.json")


def load_predictions(path, bank):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if obj.get("split") != "validation":
        raise ValueError(f"Incorrect split: {path}")

    expected_ids = [row["id"] for row in bank]

    if obj.get("ids") != expected_ids:
        raise ValueError(f"Question IDs do not match: {path}")

    selected = np.asarray(obj["selected"], dtype=int)

    if selected.shape != (len(bank),):
        raise ValueError(
            f"Incorrect prediction shape: {path}"
        )

    if np.any((selected < 0) | (selected > 3)):
        raise ValueError(f"Invalid candidate index: {path}")

    return selected


def load_references(bank):
    with (DATA / "gsm8k_validation.jsonl").open(
        "r", encoding="utf-8"
    ) as f:
        refs = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if len(bank) != len(refs):
        raise ValueError("Reference count mismatch")

    return refs


def answer_pattern(answers):
    """Return a vote pattern such as 3-1 or 2-1-1.

    Only parsed answers are grouped. Each unparseable
    candidate is its own group, not a shared None group.
    """
    keys = []

    for j, answer in enumerate(answers):
        if answer is None:
            keys.append(("unparsed", j))
        else:
            keys.append(("answer", answer))

    counts = Counter(keys)
    sizes = sorted(counts.values(), reverse=True)

    return "-".join(str(v) for v in sizes)


def get_metrics(ok, mask):
    total = int(mask.sum())

    if total == 0:
        return {
            "correct": 0,
            "total": 0,
            "accuracy_percent": None,
        }

    hits = int((ok & mask).sum())

    return {
        "correct": hits,
        "total": total,
        "accuracy_percent": 100.0 * hits / total,
    }


def show_metrics(label, data):
    if data["total"] == 0:
        print(f"  {label}: N/A")
    else:
        print(
            f"  {label}: "
            f"{data['correct']}/{data['total']} "
            f"= {data['accuracy_percent']:.2f}%"
        )


def main():
    print("=== E22-A: Hard Validation Audit ===")

    bank, _ = load_inputs("validation")
    refs = load_references(bank)

    n = len(bank)

    correct = np.zeros((n, 4), dtype=bool)
    disagreement = np.zeros(n, dtype=bool)
    patterns = []
    unparsed_counts = []

    for i, (row, ref) in enumerate(zip(bank, refs)):
        if (
            row["id"] != ref["id"]
            or row["question"] != ref["question"]
        ):
            raise ValueError(
                f"Reference alignment error at row {i}"
            )

        gold = parse_answer(ref["answer"])

        if gold is None:
            raise ValueError(
                f"Cannot parse gold answer at row {i}"
            )

        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]

        for j, answer in enumerate(answers):
            correct[i, j] = (
                answer is not None and answer == gold
            )

        parsed_answers = [
            a for a in answers if a is not None
        ]

        # At least two DISTINCT parsed final answers.
        disagreement[i] = len(set(parsed_answers)) >= 2

        patterns.append(answer_pattern(answers))
        unparsed_counts.append(
            sum(a is None for a in answers)
        )

    correct_counts = correct.sum(axis=1)

    if np.any(correct_counts == 0):
        raise ValueError(
            "Found a question without a correct candidate"
        )

    single_correct = correct_counts == 1
    non_all_correct = correct_counts < 4

    print("\n=== 1. Dataset composition ===")
    print(f"Total questions: {n}")
    print(f"Disagreement questions: {int(disagreement.sum())}")
    print(f"Single-correct questions: {int(single_correct.sum())}")
    print(f"Non-all-correct questions: {int(non_all_correct.sum())}")
    print(
        "Questions containing unparseable candidates:",
        int(np.sum(np.asarray(unparsed_counts) > 0)),
    )

    print("\nAnswer vote patterns:")
    for pattern, count in Counter(patterns).most_common():
        print(f"  {pattern}: {count}")

    rows = np.arange(n)

    predictions = {}
    success = {}
    results = {}

    print("\n=== 2. Model accuracy comparison ===")

    for name, filename in PREDICTION_FILES.items():
        selected = load_predictions(filename, bank)
        ok = correct[rows, selected]

        predictions[name] = selected
        success[name] = ok

        metrics = {
            "overall": get_metrics(
                ok, np.ones(n, dtype=bool)
            ),
            "disagreement": get_metrics(ok, disagreement),
            "single_correct": get_metrics(ok, single_correct),
            "non_all_correct": get_metrics(ok, non_all_correct),
            "disagreement_single_correct": get_metrics(
                ok, disagreement & single_correct
            ),
        }

        results[name] = metrics

        print(f"\n{name}")
        for label, data in metrics.items():
            show_metrics(label, data)

    print("\n=== 3. Accuracy by vote pattern ===")

    pattern_results = defaultdict(dict)

    unique_patterns = sorted(
        set(patterns),
        key=lambda p: (
            -Counter(patterns)[p],
            p,
        ),
    )

    patterns_array = np.asarray(patterns)

    for pattern in unique_patterns:
        mask = patterns_array == pattern
        total = int(mask.sum())

        print(f"\nPattern {pattern} ({total} questions)")

        for name, ok in success.items():
            metrics = get_metrics(ok, mask)
            pattern_results[pattern][name] = metrics
            show_metrics(name, metrics)

    print("\n=== 4. Disagreement-only pairwise comparisons ===")

    pairwise_results = {}

    model_names = list(PREDICTION_FILES)

    for i, name_a in enumerate(model_names):
        for name_b in model_names[i + 1:]:
            a = success[name_a] & disagreement
            b = success[name_b] & disagreement

            a_only = int((a & ~b).sum())
            b_only = int((b & ~a).sum())
            both_correct = int((a & b).sum())
            both_wrong = int(
                (disagreement & ~a & ~b).sum()
            )

            key = f"{name_a} vs {name_b}"
            pairwise_results[key] = {
                "both_correct": both_correct,
                f"{name_a}_only": a_only,
                f"{name_b}_only": b_only,
                "both_wrong": both_wrong,
                "oracle": both_correct + a_only + b_only,
            }

            print(f"\n{key}")
            print(f"  Both correct: {both_correct}")
            print(f"  {name_a} only correct: {a_only}")
            print(f"  {name_b} only correct: {b_only}")
            print(f"  Both wrong: {both_wrong}")
            print(
                "  Pairwise oracle:",
                both_correct + a_only + b_only,
                "/", int(disagreement.sum()),
            )

    print("\n=== 5. Best model by difficult subset ===")

    for subset in (
        "disagreement",
        "single_correct",
        "non_all_correct",
        "disagreement_single_correct",
    ):
        ranked = sorted(
            model_names,
            key=lambda name: (
                results[name][subset]["correct"],
                name,
            ),
            reverse=True,
        )

        print(f"\n{subset}")
        for name in ranked:
            data = results[name][subset]
            show_metrics(name, data)

    summary = {
        "experiment": "E22-A",
        "split": "validation",
        "n_questions": n,
        "disagreement_questions": int(disagreement.sum()),
        "single_correct_questions": int(single_correct.sum()),
        "non_all_correct_questions": int(non_all_correct.sum()),
        "vote_pattern_counts": dict(Counter(patterns)),
        "models": results,
        "by_vote_pattern": dict(pattern_results),
        "disagreement_pairwise": pairwise_results,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
