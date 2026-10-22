
"""MP2 E11: Pairwise Ranking Scorer + Majority Voting."""

from collections import Counter

import numpy as np

from common import parse_answer


VOTE_WEIGHT = 0.5


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E11 requires --checkpoint")

    # Load the E10 pairwise-ranking MLP checkpoint.
    with np.load(checkpoint, allow_pickle=False) as model:
        w1 = model["w1"]
        b1 = model["b1"]
        w2 = model["w2"]
        b2 = model["b2"]
        mean = model["mean"]
        std = model["std"]

    n = len(bank)

    # Calculate scorer scores for all candidates.
    x = arrays["embeddings"].astype(np.float32)
    x = x.reshape(-1, 1024)
    x = (x - mean) / std

    hidden = np.maximum(x @ w1 + b1, 0)
    scorer_scores = (hidden @ w2 + b2).reshape(n, 4)

    choices = []

    for row, scores in zip(bank, scorer_scores):

        # Parse the final answer of each candidate.
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]

        # Count votes for each valid final answer.
        counts = Counter(
            answer for answer in answers
            if answer is not None
        )

        # Same vote definition as E6-D:
        # frequency of each answer divided by four.
        vote_scores = np.array(
            [
                counts[answer] / 4.0
                if answer is not None else 0.0
                for answer in answers
            ],
            dtype=np.float32
        )

        # Combine pairwise scorer scores and voting scores.
        final_scores = (
            scores + VOTE_WEIGHT * vote_scores
        )

        selected = int(np.argmax(final_scores))
        choices.append(selected)

    return choices
