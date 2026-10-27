
"""E14-C: Hard-case weighted BCE MLP + Voting."""

from collections import Counter

import numpy as np
from common import parse_answer


INPUT_DIM = 1024
CANDIDATES = 4
VOTE_WEIGHT = 0.5


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E14-C requires a checkpoint")

    embeddings = arrays["embeddings"].astype(np.float32)
    n = len(bank)

    with np.load(checkpoint, allow_pickle=False) as model:
        mean = model["mean"]
        std = model["std"]
        w1 = model["w1"]
        b1 = model["b1"]
        w2 = model["w2"]
        b2 = model["b2"]

    flat = embeddings.reshape(-1, INPUT_DIM)
    flat = (flat - mean) / std

    hidden = np.maximum(flat @ w1 + b1, 0.0)
    scores = (hidden @ w2 + b2).reshape(n, CANDIDATES)

    choices = []

    for i, row in enumerate(bank):
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]

        counts = Counter(
            answer for answer in answers
            if answer is not None
        )

        vote_scores = np.asarray(
            [
                counts[answer] / 4.0
                if answer is not None else 0.0
                for answer in answers
            ],
            dtype=np.float32
        )

        combined_scores = (
            scores[i] + VOTE_WEIGHT * vote_scores
        )

        choices.append(int(np.argmax(combined_scores)))

    return choices
