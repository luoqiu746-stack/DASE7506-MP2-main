
"""E18: Conservative arithmetic reliability analysis.

Read-only analysis on official validation data.
No changes to submission.py or model checkpoints.
"""

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer


# Match explicit elementary equations, e.g. 12 * 3 = 36.
# We deliberately avoid nested or multi-operation expressions.
NUMBER = r"-?\d+(?:\.\d+)?"
EQUATION = re.compile(
    rf"(?<![\w.\d])"
    rf"({NUMBER})\s*([+×*÷/])\s*"
    rf"({NUMBER})\s*=\s*({NUMBER})"
    rf"(?![\w.\d])"
)

TOLERANCE = Decimal("0.000001")


def check_equation(match):
    try:
        a = Decimal(match.group(1))
        op = match.group(2)
        b = Decimal(match.group(3))
        result = Decimal(match.group(4))

        if op == "+":
            expected = a + b
        elif op in ("*", "×"):
            expected = a * b
        elif op in ("/", "÷"):
            if b == 0:
                return None
            expected = a / b
        else:
            return None

        return abs(expected - result) <= TOLERANCE * max(
            Decimal(1), abs(expected)
        )

    except (InvalidOperation, ZeroDivisionError):
        return None


def inspect_text(text):
    valid = 0
    invalid = 0

    for match in EQUATION.finditer(text):
        result = check_equation(match)

        if result is True:
            valid += 1
        elif result is False:
            invalid += 1

    return valid, invalid


def load_predictions(path, bank):
    obj = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    if obj["split"] != "validation":
        raise ValueError("Expected validation predictions")

    if obj["ids"] != [row["id"] for row in bank]:
        raise ValueError("Prediction IDs do not align")

    selected = np.asarray(obj["selected"], dtype=int)

    if selected.shape != (len(bank),):
        raise ValueError("Wrong prediction shape")

    if np.any((selected < 0) | (selected > 3)):
        raise ValueError("Invalid selected index")

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="runs/e16a-validation-predictions.json",
    )
    parser.add_argument(
        "--output",
        default="runs/e18-arithmetic-analysis.json",
    )
    args = parser.parse_args()

    bank, _ = load_inputs("validation")

    with open(
        DATA / "gsm8k_validation.jsonl",
        encoding="utf-8",
    ) as f:
        refs = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if len(bank) != len(refs):
        raise ValueError("Reference count mismatch")

    selected = load_predictions(args.predictions, bank)

    correct = np.zeros(
        (len(bank), 4), dtype=bool
    )
    valid_eq = np.zeros(
        (len(bank), 4), dtype=int
    )
    invalid_eq = np.zeros(
        (len(bank), 4), dtype=int
    )

    for i, (row, ref) in enumerate(zip(bank, refs)):
        if (
            row["id"] != ref["id"]
            or row["question"] != ref["question"]
        ):
            raise ValueError("Reference alignment mismatch")

        gold = parse_answer(ref["answer"])

        if gold is None:
            raise ValueError("Unparseable gold answer")

        for j, candidate in enumerate(row["candidates"]):
            correct[i, j] = (
                parse_answer(candidate["text"]) == gold
            )

            valid_eq[i, j], invalid_eq[i, j] = (
                inspect_text(candidate["text"])
            )

    rows = np.arange(len(bank))
    selected_correct = correct[rows, selected]
    selected_invalid = invalid_eq[rows, selected] > 0

    any_invalid = invalid_eq > 0
    any_checked = (valid_eq + invalid_eq) > 0

    print("=== 1. Candidate-level coverage ===")
    print(f"Total candidates: {correct.size}")
    print(
        "Candidates with checked equations:",
        int(any_checked.sum())
    )
    print(
        "Candidates with flagged arithmetic errors:",
        int(any_invalid.sum())
    )

    print("\n=== 2. Flags vs final-answer correctness ===")

    for name, mask in [
        ("Correct final answer", correct),
        ("Wrong final answer", ~correct),
    ]:
        total = int(mask.sum())
        flagged = int((mask & any_invalid).sum())
        covered = int((mask & any_checked).sum())

        print(f"{name}:")
        print(f"  Candidates: {total}")
        print(f"  Checked at least one equation: {covered}")
        print(f"  Flagged: {flagged}")
        print(
            f"  Flag rate: "
            f"{100 * flagged / total:.2f}%"
        )

    print("\n=== 3. Current model's selected candidates ===")

    print(
        "Correct selected:",
        int(selected_correct.sum())
    )
    print(
        "Wrong selected:",
        int((~selected_correct).sum())
    )
    print(
        "Correct selected but flagged:",
        int((selected_correct & selected_invalid).sum())
    )
    print(
        "Wrong selected and flagged:",
        int((~selected_correct & selected_invalid).sum())
    )

    print("\n=== 4. Potentially actionable questions ===")

    # Current choice contains a flagged equation AND
    # an alternative candidate has no flagged equations.
    alternative_unflagged = (
        (~any_invalid).sum(axis=1) >= 1
    )

    actionable = (
        selected_invalid & alternative_unflagged
    )

    print(
        "Questions where chosen candidate is flagged "
        "and an unflagged alternative exists:",
        int(actionable.sum())
    )
    print(
        "Of these, currently wrong:",
        int((actionable & ~selected_correct).sum())
    )
    print(
        "Of these, currently correct:",
        int((actionable & selected_correct).sum())
    )

    # Diagnostic only: inspect potential upside
    # without applying any selection rule.
    actionable_and_correct_alt = np.zeros(
        len(bank), dtype=bool
    )

    for i in np.flatnonzero(actionable):
        possible = (
            ~any_invalid[i]
            & correct[i]
        )
        actionable_and_correct_alt[i] = possible.any()

    print(
        "Actionable cases with at least one correct "
        "unflagged alternative (oracle diagnostic):",
        int(actionable_and_correct_alt.sum())
    )

    summary = {
        "n_questions": len(bank),
        "n_candidates": int(correct.size),
        "candidate_equation_coverage": int(any_checked.sum()),
        "candidate_arithmetic_flags": int(any_invalid.sum()),
        "flags_on_correct_candidates": int(
            (correct & any_invalid).sum()
        ),
        "flags_on_wrong_candidates": int(
            ((~correct) & any_invalid).sum()
        ),
        "selected_correct": int(selected_correct.sum()),
        "selected_wrong": int((~selected_correct).sum()),
        "selected_correct_flagged": int(
            (selected_correct & selected_invalid).sum()
        ),
        "selected_wrong_flagged": int(
            ((~selected_correct) & selected_invalid).sum()
        ),
        "actionable_questions": int(actionable.sum()),
        "actionable_currently_wrong": int(
            (actionable & ~selected_correct).sum()
        ),
        "actionable_currently_correct": int(
            (actionable & selected_correct).sum()
        ),
        "actionable_with_correct_unflagged_alternative": int(
            actionable_and_correct_alt.sum()
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
