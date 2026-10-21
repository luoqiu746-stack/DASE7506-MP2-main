
"""MP2 - E4: Embedding Scorer + Voting Score Fusion."""

from collections import Counter

import numpy as np

from common import parse_answer


# First predefined fusion experiment.
# Lambda = 0 reproduces E2 exactly.
VOTE_WEIGHT = 0.5


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E4 requires --checkpoint")

    # Load the existing E2 scorer.
    with np.load(checkpoint, allow_pickle=False) as model:
        w = model["weight"]
        b = float(model["bias"])
        mean = model["mean"]
        std = model["std"]

    x = arrays["embeddings"].astype(np.float32)
    x = (x - mean) / std

    # Logistic-regression logits, shape (N, 4).
    scorer_scores = x @ w + b

    choices = []

    for row, scores in zip(bank, scorer_scores):
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]

        # Vote only for successfully parsed numeric answers.
        counts = Counter(
            answer for answer in answers
            if answer is not None
        )

        vote_scores = np.array([
            counts[answer] / 4.0 if answer is not None else 0.0
            for answer in answers
        ], dtype=np.float32)

        # Scorer dominates; voting provides a bonus.
        final_scores = scores + VOTE_WEIGHT * vote_scores

        selected = int(np.argmax(final_scores))
        choices.append(selected)

    return choices
