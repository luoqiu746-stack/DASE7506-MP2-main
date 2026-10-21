
"""MP2 E5: Embedding Scorer + Voting + Arithmetic Verifier."""

import re
from collections import Counter

import numpy as np

from common import parse_answer


VOTE_WEIGHT = 0.5
ERROR_PENALTY = 0.75

# Match simple arithmetic equations such as:
# 12 + 8 = 22
# 6 * 4 = 24
# 15 / 3 = 5
# 2.5 - 1 = 1.5
NUMBER = r"-?\d+(?:\.\d+)?"

EQUATION = re.compile(
    rf"(?<![\w.])"
    rf"(?P<a>{NUMBER})\s*"
    rf"(?P<op>[+\-*/×÷])\s*"
    rf"(?P<b>{NUMBER})\s*=\s*"
    rf"(?P<c>{NUMBER})"
    rf"(?![\w.%+\-*/×÷=])"
)


def arithmetic_errors(text):
    """Count clear mistakes in simple numeric equations.

    Unsupported equations are ignored.
    No eval(), gold answers, or external model calls.
    """
    errors = 0

    for match in EQUATION.finditer(text):
        a = float(match.group("a"))
        b = float(match.group("b"))
        c = float(match.group("c"))
        operator = match.group("op")

        if operator == "+":
            expected = a + b
        elif operator == "-":
            expected = a - b
        elif operator in ("*", "×"):
            expected = a * b
        elif operator in ("/", "÷"):
            if b == 0:
                continue
            expected = a / b
        else:
            continue

        # Allow small rounding differences.
        tolerance = max(1e-6, abs(expected) * 1e-5)

        if abs(expected - c) > tolerance:
            errors += 1

    return errors


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E5 requires --checkpoint")

    # Reuse the existing E2 Logistic Regression Scorer.
    with np.load(checkpoint, allow_pickle=False) as model:
        w = model["weight"]
        b = float(model["bias"])
        mean = model["mean"]
        std = model["std"]

    x = arrays["embeddings"].astype(np.float32)
    x = (x - mean) / std

    # Shape: (N, 4)
    scorer_scores = x @ w + b

    choices = []

    for row, scores in zip(bank, scorer_scores):
        candidates = row["candidates"]

        answers = [
            parse_answer(candidate["text"])
            for candidate in candidates
        ]

        counts = Counter(
            a for a in answers if a is not None
        )

        vote_scores = np.array([
            counts[a] / 4.0 if a is not None else 0.0
            for a in answers
        ], dtype=np.float32)

        # Arithmetic error signal:
        # 1 if at least one clear error is found, otherwise 0.
        error_flags = np.array([
            float(arithmetic_errors(candidate["text"]) > 0)
            for candidate in candidates
        ], dtype=np.float32)

        # E4 score with an additional arithmetic-error penalty.
        final_scores = (
            scores
            + VOTE_WEIGHT * vote_scores
            - ERROR_PENALTY * error_flags
        )

        choices.append(int(np.argmax(final_scores)))

    return choices
