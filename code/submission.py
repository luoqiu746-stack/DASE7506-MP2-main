
"""MP2 E6-D: MLP Scorer + Voting Fusion."""

from collections import Counter

import numpy as np

from common import parse_answer


VOTE_WEIGHT = 0.5


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E6-D requires --checkpoint")

    with np.load(checkpoint, allow_pickle=False) as model:
        w1 = model["w1"]
        b1 = model["b1"]
        w2 = model["w2"]
        b2 = model["b2"]
        mean = model["mean"]
        std = model["std"]

    x = arrays["embeddings"].astype(np.float32)
    n = len(bank)

    x = x.reshape(-1, 1024)
    x = (x - mean) / std

    hidden = np.maximum(x @ w1 + b1, 0)
    mlp_scores = (hidden @ w2 + b2).reshape(n, 4)

    choices = []

    for row, scores in zip(bank, mlp_scores):
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]

        counts = Counter(
            answer for answer in answers
            if answer is not None
        )

        vote_scores = np.array([
            counts[answer] / 4.0 if answer is not None else 0.0
            for answer in answers
        ], dtype=np.float32)

        final_scores = scores + VOTE_WEIGHT * vote_scores

        choices.append(int(np.argmax(final_scores)))

    return choices
