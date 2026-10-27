
"""E19-C: Compare E19-B answer aggregation with E16-A.

Read-only analysis using official validation references.
No model training or prediction changes.
"""

import json
from pathlib import Path

import numpy as np
from common import DATA, load_inputs, parse_answer


FILES = {
    "E14-B": "runs/e14b-validation-predictions.json",
    "E16-A": "runs/e16a-validation-predictions.json",
    "E19-B": "runs/e19-validation-predictions.json",
}


def load_predictions(path, bank):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))

    if obj["split"] != "validation":
        raise ValueError(f"Wrong split: {path}")

    if obj["ids"] != [row["id"] for row in bank]:
        raise ValueError(f"ID mismatch: {path}")

    selected = np.asarray(obj["selected"], dtype=int)

    if selected.shape != (len(bank),):
        raise ValueError(f"Wrong shape: {path}")

    if np.any((selected < 0) | (selected > 3)):
        raise ValueError(f"Invalid candidate index: {path}")

    return selected


def main():
    bank, _ = load_inputs("validation")

    with open(DATA / "gsm8k_validation.jsonl", encoding="utf-8") as f:
        references = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if len(bank) != len(references):
        raise ValueError("Reference count mismatch")

    correct = []

    for row, ref in zip(bank, references):
        if (
            row["id"] != ref["id"]
            or row["question"] != ref["question"]
        ):
            raise ValueError("Reference alignment mismatch")

        gold = parse_answer(ref["answer"])

        if gold is None:
            raise ValueError(f"Unparseable gold: {row['id']}")

        correct.append([
            parse_answer(candidate["text"]) == gold
            for candidate in row["candidates"]
        ])

    correct = np.asarray(correct, dtype=bool)
    n = len(bank)
    rows = np.arange(n)
    correct_counts = correct.sum(axis=1)

    predictions = {
        name: load_predictions(path, bank)
        for name, path in FILES.items()
    }

    success = {
        name: correct[rows, selected]
        for name, selected in predictions.items()
    }

    print("=== 1. Overall accuracy ===")

    for name, ok in success.items():
        hits = int(ok.sum())
        print(f"{name}: {hits}/{n} = {100 * hits / n:.4f}%")

    print("\n=== 2. Accuracy by number of correct candidates ===")

    by_count = {}

    for k in range(1, 5):
        mask = correct_counts == k
        total = int(mask.sum())

        by_count[str(k)] = {}

        print(f"\n{k} correct candidates: {total} questions")

        for name, ok in success.items():
            hits = int((mask & ok).sum())

            by_count[str(k)][name] = hits

            print(
                f"  {name}: {hits}/{total} "
                f"= {100 * hits / total:.2f}%"
            )

    print("\n=== 3. E19-B vs E16-A ===")

    old = success["E16-A"]
    new = success["E19-B"]

    both_correct = old & new
    e19_only = ~old & new
    e16_only = old & ~new
    both_wrong = ~old & ~new

    print("Both correct:", int(both_correct.sum()))
    print("E19-B only correct:", int(e19_only.sum()))
    print("E16-A only correct:", int(e16_only.sum()))
    print("Both wrong:", int(both_wrong.sum()))
    print(
        "Two-model oracle:",
        int((old | new).sum()),
        "/",
        n,
    )

    changed = predictions["E16-A"] != predictions["E19-B"]

    print("Different selected candidate:", int(changed.sum()))

    print("\n=== 4. Improvements and regressions by difficulty ===")

    changes_by_count = {}

    for k in range(1, 5):
        mask = correct_counts == k

        gained = int((mask & e19_only).sum())
        lost = int((mask & e16_only).sum())

        changes_by_count[str(k)] = {
            "gained": gained,
            "lost": lost,
            "net": gained - lost,
        }

        print(
            f"{k} correct candidates: "
            f"gained={gained}, lost={lost}, "
            f"net={gained - lost:+d}"
        )

    print("\n=== 5. Single-correct candidate performance ===")

    hard = correct_counts == 1

    for name, ok in success.items():
        hits = int((hard & ok).sum())
        print(f"{name}: {hits}/{int(hard.sum())}")

    print("\n=== 6. E19-B errors still remaining ===")

    wrong = ~new

    print("Total wrong:", int(wrong.sum()))

    for k in range(1, 5):
        print(
            f"{k} correct candidates:",
            int((wrong & (correct_counts == k)).sum()),
        )

    summary = {
        "n": n,
        "overall": {
            name: int(ok.sum()) for name, ok in success.items()
        },
        "by_correct_count": by_count,
        "both_correct": int(both_correct.sum()),
        "e19_only_correct": int(e19_only.sum()),
        "e16_only_correct": int(e16_only.sum()),
        "both_wrong": int(both_wrong.sum()),
        "different_selected_candidate": int(changed.sum()),
        "two_model_oracle": int((old | new).sum()),
        "changes_by_correct_count": changes_by_count,
    }

    output = Path("runs/e19-error-analysis.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
