
"""E15: Error analysis for E14-C.

Compare E14-B, E14-C, E6-D, E11, and E1 Voting.
Read-only diagnostic: no training or checkpoint changes.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer


PREDICTION_FILES = {
    "E1": "runs/e1-validation-predictions.json",
    "E6-D": "runs/e6d-validation-predictions.json",
    "E11": "runs/e11-validation-predictions.json",
    "E14-B": "runs/e14b-validation-predictions.json",
    "E14-C": "runs/e14c-validation-predictions.json",
}


def load_predictions(path, bank):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    obj = json.loads(path.read_text(encoding="utf-8"))

    if obj["split"] != "validation":
        raise ValueError(f"Wrong split: {path}")

    expected_ids = [row["id"] for row in bank]

    if obj["ids"] != expected_ids:
        raise ValueError(f"ID alignment mismatch: {path}")

    selected = np.asarray(obj["selected"], dtype=int)

    if selected.shape != (len(bank),):
        raise ValueError(f"Wrong prediction shape: {path}")

    if np.any((selected < 0) | (selected > 3)):
        raise ValueError(f"Invalid candidate index: {path}")

    return selected


def print_overlap(name_a, name_b, predictions, correct):
    a = predictions[name_a]
    b = predictions[name_b]

    rows = np.arange(len(correct))

    a_ok = correct[rows, a]
    b_ok = correct[rows, b]

    both_correct = int((a_ok & b_ok).sum())
    a_only = int((a_ok & ~b_ok).sum())
    b_only = int((~a_ok & b_ok).sum())
    both_wrong = int((~a_ok & ~b_ok).sum())

    print(f"\n{name_a} vs {name_b}")
    print(f"  Both correct:        {both_correct}")
    print(f"  {name_a} only correct: {a_only}")
    print(f"  {name_b} only correct: {b_only}")
    print(f"  Both wrong:          {both_wrong}")
    print(
        f"  Two-method oracle:   "
        f"{both_correct + a_only + b_only}/{len(correct)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="runs/e15-error-analysis.json",
    )
    args = parser.parse_args()

    bank, _ = load_inputs("validation")

    refs_path = DATA / "gsm8k_validation.jsonl"

    with open(refs_path, encoding="utf-8") as f:
        references = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if len(bank) != len(references):
        raise ValueError("Reference count mismatch")

    correct_rows = []
    answers_all = []

    for row, ref in zip(bank, references):
        if (
            row["id"] != ref["id"]
            or row["question"] != ref["question"]
        ):
            raise ValueError("Reference alignment mismatch")

        gold = parse_answer(ref["answer"])

        if gold is None:
            raise ValueError(f"Cannot parse gold: {row['id']}")

        answers = [
            parse_answer(c["text"])
            for c in row["candidates"]
        ]

        flags = [answer == gold for answer in answers]

        correct_rows.append(flags)
        answers_all.append(answers)

    correct = np.asarray(correct_rows, dtype=bool)
    n = len(bank)
    rows = np.arange(n)

    correct_counts = correct.sum(axis=1)

    predictions = {
        name: load_predictions(path, bank)
        for name, path in PREDICTION_FILES.items()
    }

    success = {
        name: correct[rows, selected]
        for name, selected in predictions.items()
    }

    print("=== 1. Overall accuracy ===")

    overall = {}

    for name, ok in success.items():
        hits = int(ok.sum())
        accuracy = 100.0 * hits / n
        overall[name] = {
            "correct": hits,
            "n": n,
            "accuracy_percent": accuracy,
        }
        print(f"{name:6s}: {hits}/{n} = {accuracy:.4f}%")

    print("\n=== 2. Accuracy by correct candidate count ===")

    by_count = {}

    for k in range(1, 5):
        mask = correct_counts == k
        total = int(mask.sum())

        print(f"\n{k} correct candidates: {total} questions")

        by_count[str(k)] = {}

        for name, ok in success.items():
            hits = int(ok[mask].sum())
            accuracy = 100.0 * hits / total if total else 0.0

            by_count[str(k)][name] = {
                "correct": hits,
                "total": total,
                "accuracy_percent": accuracy,
            }

            print(
                f"  {name:6s}: "
                f"{hits}/{total} = {accuracy:.2f}%"
            )

    print("\n=== 3. E14-C vs earlier best models ===")

    print_overlap("E14-C", "E6-D", predictions, correct)
    print_overlap("E14-C", "E11", predictions, correct)

    print("\n=== 4. Effect of adding Voting to E14-B ===")

    b = success["E14-B"]
    c = success["E14-C"]

    both_correct = int((b & c).sum())
    voting_helped = int((~b & c).sum())
    voting_hurt = int((b & ~c).sum())
    both_wrong = int((~b & ~c).sum())

    print(f"Both correct: {both_correct}")
    print(f"Voting helped (B wrong, C correct): {voting_helped}")
    print(f"Voting hurt   (B correct, C wrong): {voting_hurt}")
    print(f"Both wrong: {both_wrong}")
    print(f"Net improvement: {voting_helped - voting_hurt:+d}")

    print("\n=== 5. Voting effects by difficulty ===")

    voting_by_count = {}

    for k in range(1, 5):
        mask = correct_counts == k

        helped = int((mask & ~b & c).sum())
        hurt = int((mask & b & ~c).sum())

        voting_by_count[str(k)] = {
            "helped": helped,
            "hurt": hurt,
            "net": helped - hurt,
        }

        print(
            f"{k} correct: helped={helped}, "
            f"hurt={hurt}, net={helped - hurt:+d}"
        )

    print("\n=== 6. E14-C errors: answer voting patterns ===")

    pattern_counts = Counter()

    for i in range(n):
        if success["E14-C"][i]:
            continue

        answers = [
            a for a in answers_all[i]
            if a is not None
        ]

        counts = Counter(answers)
        pattern = tuple(
            sorted(counts.values(), reverse=True)
        )

        if len(answers) < 4:
            pattern = pattern + ("unparsed", 4 - len(answers))

        pattern_counts[str(pattern)] += 1

    for pattern, count in pattern_counts.most_common():
        print(f"{pattern}: {count}")

    print("\n=== 7. Three-model correctness overlap ===")

    a = success["E6-D"]
    d = success["E11"]
    c = success["E14-C"]

    triple_overlap = {}

    for a_ok in (False, True):
        for d_ok in (False, True):
            for c_ok in (False, True):
                count = int(
                    (
                        (a == a_ok)
                        & (d == d_ok)
                        & (c == c_ok)
                    ).sum()
                )

                key = (
                    f"E6D={int(a_ok)},"
                    f"E11={int(d_ok)},"
                    f"E14C={int(c_ok)}"
                )

                triple_overlap[key] = count
                print(f"{key}: {count}")

    any_correct = a | d | c
    print(
        f"\nThree-model oracle: "
        f"{int(any_correct.sum())}/{n}"
    )

    summary = {
        "n_questions": n,
        "zero_correct_candidates": int(
            (correct_counts == 0).sum()
        ),
        "overall": overall,
        "by_correct_candidate_count": by_count,
        "e14b_to_e14c": {
            "both_correct": both_correct,
            "voting_helped": voting_helped,
            "voting_hurt": voting_hurt,
            "both_wrong": both_wrong,
        },
        "voting_by_correct_count": voting_by_count,
        "e14c_wrong_answer_patterns": dict(pattern_counts),
        "three_model_overlap": triple_overlap,
        "three_model_oracle": int(any_correct.sum()),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved diagnostic summary: {output}")


if __name__ == "__main__":
    main()
