
"""MP2 - E1 Baseline: Majority Voting."""

from collections import Counter
from common import parse_answer


def predict(bank, arrays, checkpoint):
    """Select one candidate per question using majority voting."""

    choices = []

    for row in bank:
        # Extract final numeric answers from all four candidates
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]

        # Count only successfully parsed answers
        valid_answers = [a for a in answers if a is not None]

        if not valid_answers:
            choices.append(0)
            continue

        # Select the most frequent numeric answer
        counts = Counter(valid_answers)
        max_votes = max(counts.values())

        # Tied answer groups: prefer the group containing
        # the lowest candidate index
        for index, answer in enumerate(answers):
            if answer is not None and counts[answer] == max_votes:
                choices.append(index)
                break

    return choices
