
"""E16-D: Diagnose agreement-gated voting.

Compare E14-B, E14-C and E16-A using official
validation references. Read-only diagnostic.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from common import DATA, load_inputs, parse_answer


FILES = {
    "E14-B": "runs/e14b-validation-predictions.json",
    "E14-C": "runs/e14c-validation-predictions.json",
    "E16-A": "runs/e16a-validation-predictions.json",
}


def load_predictions(path, bank):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))

    if obj["split"] != "validation":
        raise ValueError(f"Wrong split: {path}")

    if obj["ids"] != [r["id"] for r in bank]:
        raise ValueError(f"ID mismatch: {path}")

    selected = np.asarray(obj["selected"], dtype=int)

    if selected.shape != (len(bank),):
        raise ValueError(f"Wrong shape: {path}")

    if np.any((selected < 0) | (selected > 3)):
        raise ValueError(f"Invalid selected index: {path}")

    return selected


def main():
    bank, _ = load_inputs("validation")

    with open(
        DATA / "gsm8k_validation.jsonl",
        encoding="utf-8",
    ) as f:
        refs = [json.loads(line) for line in f if line.strip()]

    if len(bank) != len(refs):
        raise ValueError("Reference count mismatch")

    correct = []
    vote_patterns = []

    for row, ref in zip(bank, refs):
        if (
            row["id"] != ref["id"]
            or row["question"] != ref["question"]
        ):
            raise ValueError("Reference alignment mismatch")

        gold = parse_answer(ref["answer"])

        if gold is None:
            raise ValueError(f"Unparseable gold: {row['id']}")

        answers = [
            parse_answer(c["text"])
            for c in row["candidates"]
        ]

        correct.append([a == gold for a in answers])

        counts = Counter(
            a for a in answers if a is not None
        )
        vote_patterns.append(
            tuple(sorted(counts.values(), reverse=True))
        )

    correct = np.asarray(correct, dtype=bool)
    n = len(bank)
    rows = np.arange(n)
    counts_correct = correct.sum(axis=1)

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
        print(
            f"{name:6s}: {hits}/{n} "
            f"= {100 * hits / n:.4f}%"
        )

    print("\n=== 2. E14-C vs E16-A ===")

    old = success["E14-C"]
    new = success["E16-A"]

    repaired = ~old & new
    damaged = old & ~new
    both_correct = old & new
    both_wrong = ~old & ~new

    changed_index = (
        predictions["E14-C"] != predictions["E16-A"]
    )

    print(f"Different selected index: {int(changed_index.sum())}")
    print(f"Both correct: {int(both_correct.sum())}")
    print(f"E16-A repaired: {int(repaired.sum())}")
    print(f"E16-A damaged: {int(damaged.sum())}")
    print(f"Both wrong: {int(both_wrong.sum())}")
    print(
        f"Net improvement: "
        f"{int(repaired.sum()) - int(damaged.sum()):+d}"
    )

    print("\n=== 3. Changes by number of correct candidates ===")

    by_count = {}

    for k in range(1, 5):
        mask = counts_correct == k

        values = {
            "total": int(mask.sum()),
            "e14c_correct": int((mask & old).sum()),
            "e16a_correct": int((mask & new).sum()),
            "changed": int((mask & changed_index).sum()),
            "repaired": int((mask & repaired).sum()),
            "damaged": int((mask & damaged).sum()),
        }

        by_count[str(k)] = values

        print(
            f"{k} correct candidates: "
            f"E14-C={values['e14c_correct']}, "
            f"E16-A={values['e16a_correct']}, "
            f"changed={values['changed']}, "
            f"repaired={values['repaired']}, "
            f"damaged={values['damaged']}"
        )

    print("\n=== 4. Previously harmed by Voting ===")

    baseline = success["E14-B"]

    old_voting_hurt = baseline & ~old
    old_voting_helped = ~baseline & old

    print(
        "E14-C voting harmed:",
        int(old_voting_hurt.sum())
    )
    print(
        "Of those, E16-A repaired:",
        int((old_voting_hurt & new).sum())
    )
    print(
        "E14-C voting helped:",
        int(old_voting_helped.sum())
    )
    print(
        "Of those, E16-A lost:",
        int((old_voting_helped & ~new).sum())
    )

    print("\n=== 5. Answer patterns for changed predictions ===")

    pattern_summary = Counter(
        str(vote_patterns[i])
        for i in range(n)
        if changed_index[i]
    )

    for pattern, total in pattern_summary.most_common():
        print(f"{pattern}: {total}")

    print("\n=== 6. Cases still wrong in E16-A ===")

    wrong = ~new
    print(f"Total E16-A errors: {int(wrong.sum())}")

    for k in range(1, 5):
        print(
            f"{k} correct candidates: "
            f"{int((wrong & (counts_correct == k)).sum())}"
        )

    print("\n=== 7. E16-A vs E14-B ===")

    b = success["E14-B"]

    print("Both correct:", int((b & new).sum()))
    print("E16-A only correct:", int((~b & new).sum()))
    print("E14-B only correct:", int((b & ~new).sum()))
    print("Both wrong:", int((~b & ~new).sum()))

    summary = {
        "total": n,
        "overall": {
            name: int(ok.sum())
            for name, ok in success.items()
        },
        "different_selected_index": int(
            changed_index.sum()
        ),
        "repaired": int(repaired.sum()),
        "damaged": int(damaged.sum()),
        "both_correct": int(both_correct.sum()),
        "both_wrong": int(both_wrong.sum()),
        "by_correct_count": by_count,
        "old_voting_harmed": int(
            old_voting_hurt.sum()
        ),
        "old_voting_harmed_repaired": int(
            (old_voting_hurt & new).sum()
        ),
        "old_voting_helped": int(
            old_voting_helped.sum()
        ),
        "old_voting_helped_lost": int(
            (old_voting_helped & ~new).sum()
        ),
        "changed_answer_patterns": dict(
            pattern_summary
        ),
    }

    output = Path("runs/e16-error-analysis.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
