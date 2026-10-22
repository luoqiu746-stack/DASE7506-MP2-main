
"""E11-D: Diagnostic analysis of candidate-selection errors.

Uses validation references for offline analysis ONLY.
Never import this file into submission.py.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer


def load_predictions(path, bank):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))

    if obj["split"] != "validation":
        raise ValueError(f"Not validation: {path}")

    if obj["ids"] != [row["id"] for row in bank]:
        raise ValueError(f"IDs do not align: {path}")

    selected = np.asarray(obj["selected"], dtype=int)

    if selected.shape != (len(bank),):
        raise ValueError(f"Unexpected prediction shape: {path}")

    if np.any((selected < 0) | (selected > 3)):
        raise ValueError(f"Invalid candidate index: {path}")

    return selected


def majority_choice(answers):
    counts = Counter(a for a in answers if a is not None)

    if not counts:
        return 0

    # Deterministic first-occurrence tie breaker.
    return max(
        range(4),
        key=lambda i: counts.get(answers[i], 0)
        if answers[i] is not None else 0
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--e11",
        default="runs/e11-validation-predictions.json"
    )
    parser.add_argument(
        "--e10",
        default="runs/e10-validation-predictions.json"
    )
    parser.add_argument(
        "--e6d",
        default=None,
        help="Path to saved E6-D prediction JSON"
    )
    parser.add_argument(
        "--output",
        default="runs/e11-error-diagnosis.json"
    )

    args = parser.parse_args()

    bank, _ = load_inputs("validation")

    references = [
        json.loads(line)
        for line in (
            DATA / "gsm8k_validation.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(bank) != len(references):
        raise ValueError("Reference count mismatch")

    correct = []
    answers_all = []
    majority = []

    for row, ref in zip(bank, references):

        if (
            row["id"] != ref["id"]
            or row["question"] != ref["question"]
        ):
            raise ValueError("Reference alignment mismatch")

        gold = parse_answer(ref["answer"])

        if gold is None:
            raise ValueError("Unparseable gold answer")

        answers = [
            parse_answer(c["text"])
            for c in row["candidates"]
        ]

        flags = [a == gold for a in answers]

        correct.append(flags)
        answers_all.append(answers)
        majority.append(majority_choice(answers))

    correct = np.asarray(correct, dtype=bool)
    majority = np.asarray(majority, dtype=int)

    n = len(bank)
    row_idx = np.arange(n)

    predictions = {
        "E10": load_predictions(args.e10, bank),
        "E11": load_predictions(args.e11, bank),
        "Voting_proxy": majority,
    }

    if args.e6d is not None:
        predictions["E6-D"] = load_predictions(args.e6d, bank)

    results = {}

    print("\n=== 1. Candidate availability ===")

    correct_counts = correct.sum(axis=1)
    zero_correct = int(np.sum(correct_counts == 0))

    print(f"Total questions: {n}")
    print(f"Questions with zero correct candidates: {zero_correct}")

    if zero_correct:
        print("WARNING: Unexpected zero-correct groups!")

    print("\n=== 2. Accuracy by number of correct candidates ===")

    for k in range(5):
        mask = correct_counts == k
        total = int(mask.sum())

        print(f"\n{k} correct candidates: {total} questions")

        if total == 0:
            continue

        for name, selected in predictions.items():
            wins = int(correct[row_idx, selected][mask].sum())
            accuracy = 100 * wins / total

            print(
                f"  {name:12s}: "
                f"{wins}/{total} = {accuracy:.2f}%"
            )

    print("\n=== 3. Overall accuracy ===")

    correctness = {}

    for name, selected in predictions.items():
        flags = correct[row_idx, selected]
        correctness[name] = flags

        wins = int(flags.sum())
        print(f"{name:12s}: {wins}/{n} = {wins/n*100:.4f}%")

    print("\n=== 4. E10 vs E11 ===")

    a = correctness["E10"]
    b = correctness["E11"]

    for description, count in [
        ("Both correct", np.sum(a & b)),
        ("E10 only correct", np.sum(a & ~b)),
        ("E11 only correct", np.sum(~a & b)),
        ("Both wrong", np.sum(~a & ~b)),
    ]:
        print(f"{description}: {int(count)}")

    print("\n=== 5. E11 vs Voting ===")

    a = correctness["E11"]
    b = correctness["Voting_proxy"]

    for description, count in [
        ("Both correct", np.sum(a & b)),
        ("E11 only correct", np.sum(a & ~b)),
        ("Voting only correct", np.sum(~a & b)),
        ("Both wrong", np.sum(~a & ~b)),
    ]:
        print(f"{description}: {int(count)}")

    print("\n=== 6. E10/Voting disagreements ===")

    e10 = predictions["E10"]
    vote = predictions["Voting_proxy"]

    disagree = np.array([
        answers_all[i][e10[i]] != answers_all[i][vote[i]]
        for i in range(n)
    ])

    print(f"Different final answers: {int(disagree.sum())}")
    print(
        "E10 correct in disagreements:",
        int(correctness["E10"][disagree].sum())
    )
    print(
        "Voting correct in disagreements:",
        int(correctness["Voting_proxy"][disagree].sum())
    )

    if "E6-D" in predictions:
        print("\n=== 7. E6-D vs E11 ===")

        a = correctness["E6-D"]
        b = correctness["E11"]

        for description, count in [
            ("Both correct", np.sum(a & b)),
            ("E6-D only correct", np.sum(a & ~b)),
            ("E11 only correct", np.sum(~a & b)),
            ("Both wrong", np.sum(~a & ~b)),
        ]:
            print(f"{description}: {int(count)}")

        oracle = int(np.sum(a | b))
        print(
            f"Two-method oracle: "
            f"{oracle}/{n} = {100*oracle/n:.4f}%"
        )

    # Aggregate diagnostic results; do not store gold answers
    # or per-question selection lookup tables.
    results["total"] = n
    results["zero_correct_candidates"] = zero_correct

    results["correct_candidate_distribution"] = {
        str(k): int(np.sum(correct_counts == k))
        for k in range(5)
    }

    results["overall_correct"] = {
        name: int(flags.sum())
        for name, flags in correctness.items()
    }

    results["e10_voting_disagreement_count"] = int(
        disagree.sum()
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    print(f"\nDiagnostic summary saved: {output}")


if __name__ == "__main__":
    main()
