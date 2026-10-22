
"""MP2 E13: Group-aware candidate scorer."""

from collections import Counter

import numpy as np
from common import parse_answer


EPS = 1e-6


def make_features(bank, embeddings):
    e = embeddings.astype(np.float32)
    n = len(bank)

    group_mean = e.mean(axis=1, keepdims=True)
    relative = e - group_mean

    norm = e / np.maximum(
        np.linalg.norm(e, axis=2, keepdims=True), EPS
    )
    similarity = norm @ norm.transpose(0, 2, 1)

    mean_similarity = (
        similarity.sum(axis=2)
        - np.diagonal(similarity, axis1=1, axis2=2)
    ) / 3.0

    votes = np.zeros((n, 4), dtype=np.float32)

    for i, row in enumerate(bank):
        answers = [
            parse_answer(c["text"]) for c in row["candidates"]
        ]
        counts = Counter(a for a in answers if a is not None)

        for j, answer in enumerate(answers):
            votes[i, j] = (
                counts[answer] / 4.0 if answer is not None else 0.0
            )

    return np.concatenate(
        [
            e,
            relative,
            votes[..., None],
            mean_similarity[..., None],
        ],
        axis=2,
    ).astype(np.float32)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E13 requires --checkpoint")

    x = make_features(bank, arrays["embeddings"])
    n = len(bank)

    with np.load(checkpoint, allow_pickle=False) as model:
        mean = model["mean"]
        std = model["std"]
        w1 = model["w1"]
        b1 = model["b1"]
        w2 = model["w2"]
        b2 = model["b2"]

    flat = x.reshape(-1, 2050)
    flat = (flat - mean) / std

    hidden = np.maximum(flat @ w1 + b1, 0)
    scores = (hidden @ w2 + b2).reshape(n, 4)

    return [int(i) for i in np.argmax(scores, axis=1)]
