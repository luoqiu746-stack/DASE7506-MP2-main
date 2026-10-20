
"""Local MP2 E1 error analysis. No model training."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

# Allow importing the official parser from the parent code directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import parse_answer


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]



def read_predictions(path):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)

    # Official MP2 prediction format
    if isinstance(obj, dict) and "selected" in obj:
        predictions = obj["selected"]
    elif isinstance(obj, list):
        predictions = obj
    else:
        raise ValueError("Unrecognized prediction JSON structure.")

    return [int(index) for index in predictions]





def voting_pattern(answers):
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return "0 valid"

    pattern = "-".join(
        str(n) for n in sorted(counts.values(), reverse=True)
    )
    missing = sum(a is None for a in answers)

    if missing:
        pattern += f" + {missing} missing"

    return pattern


def analyze(bank, references, predictions, output_dir):
    if not (len(bank) == len(references) == len(predictions)):
        raise ValueError(
            "Bank, reference and prediction row counts differ: "
            f"{len(bank)}, {len(references)}, {len(predictions)}"
        )

    # Match reference answers by question ID, not assumed row order.
    reference_by_id = {
        str(row["id"]): row for row in references
        if "id" in row
    }

    summary = {}
    error_cases = []

    for row_number, (row, selected) in enumerate(
        zip(bank, predictions), start=1
    ):
        candidates = row["candidates"]
        answers = [parse_answer(c["text"]) for c in candidates]

        if not 0 <= selected < len(candidates):
            raise ValueError(f"Invalid choice on row {row_number}")

        if reference_by_id and "id" in row:
            ref = reference_by_id[str(row["id"])]
        else:
            ref = references[row_number - 1]

        gold = parse_answer(ref["answer"])

        if gold is None:
            raise ValueError(f"Unparsable gold answer, row {row_number}")

        counts = Counter(a for a in answers if a is not None)
        pattern = voting_pattern(answers)
        max_votes = max(counts.values(), default=0)

        winners = [
            a for a, votes in counts.items()
            if votes == max_votes
        ]
        tied = len(winners) > 1
        parse_failures = sum(a is None for a in answers)

        chosen_answer = answers[selected]
        correct = chosen_answer == gold

        key = (pattern, "yes" if tied else "no")
        if key not in summary:
            summary[key] = {
                "total": 0,
                "correct": 0,
                "errors": 0
            }

        summary[key]["total"] += 1
        summary[key]["correct"] += int(correct)
        summary[key]["errors"] += int(not correct)

        if not correct:
            if parse_failures:
                category = "Contains parse failure"
            elif tied:
                category = "Tie-breaking / disagreement"
            else:
                category = "Majority answer incorrect"

            correct_indices = [
                i for i, a in enumerate(answers) if a == gold
            ]

            error_cases.append({
                "row": row_number,
                "question_id": row.get("id", ""),
                "question": row.get("question", ""),
                "pattern": pattern,
                "tie": tied,
                "parse_failures": parse_failures,
                "selected_index": selected,
                "selected_answer": chosen_answer,
                "gold_answer": gold,
                "correct_indices": ",".join(map(str, correct_indices)),
                "error_category": category,
                "answer_0": answers[0],
                "answer_1": answers[1],
                "answer_2": answers[2],
                "answer_3": answers[3],
                "trace_0": candidates[0]["text"],
                "trace_1": candidates[1]["text"],
                "trace_2": candidates[2]["text"],
                "trace_3": candidates[3]["text"],
            })

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "voting_pattern_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Voting Pattern", "Tie", "Total",
            "Correct", "Errors", "Accuracy (%)"
        ])

        for (pattern, tied), stats in sorted(summary.items()):
            accuracy = 100 * stats["correct"] / stats["total"]
            writer.writerow([
                pattern, tied, stats["total"],
                stats["correct"], stats["errors"],
                round(accuracy, 4)
            ])

    errors_path = output_dir / "e1_error_cases.csv"
    if error_cases:
        with open(errors_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(error_cases[0].keys())
            )
            writer.writeheader()
            writer.writerows(error_cases)

    total = len(bank)
    wrong = len(error_cases)

    print(f"Total questions: {total}")
    print(f"Correct: {total - wrong}")
    print(f"Errors: {wrong}")
    print(f"Accuracy: {100 * (total - wrong) / total:.4f}%")
    print(f"Summary: {summary_path}")
    print(f"Error cases: {errors_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/e1-analysis")
    )

    args = parser.parse_args()

    analyze(
        read_jsonl(args.bank),
        read_jsonl(args.references),
        read_predictions(args.predictions),
        args.output_dir
    )
