
"""MP2 E12: BCE + Pairwise MLP ensemble with voting."""

from collections import Counter

import numpy as np

from common import parse_answer


BCE_WEIGHT = 0.5
PAIR_WEIGHT = 0.5
VOTE_WEIGHT = 0.5

EPS = 1e-6


def mlp_scores(x, model, prefix):
    """Compute four candidate scores per question."""

    w1 = model[f"{prefix}_w1"]
    b1 = model[f"{prefix}_b1"]
    w2 = model[f"{prefix}_w2"]
    b2 = model[f"{prefix}_b2"]
    mean = model[f"{prefix}_mean"]
    std = model[f"{prefix}_std"]

    n = x.shape[0]

    flat = x.reshape(-1, 1024)
    flat = (flat - mean) / std

    hidden = np.maximum(flat @ w1 + b1, 0)
    scores = hidden @ w2 + b2

    return scores.reshape(n, 4)


def normalize_scores(scores):
    """Standardize scores within each question."""

    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)

    return (scores - mean) / np.maximum(std, EPS)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E12 requires --checkpoint")

    x = arrays["embeddings"].astype(np.float32)

    with np.load(checkpoint, allow_pickle=False) as model:
        bce_scores = mlp_scores(x, model, "bce")
        pair_scores = mlp_scores(x, model, "pair")

    # Normalize each model's scores within each question.
    bce_norm = normalize_scores(bce_scores)
    pair_norm = normalize_scores(pair_scores)

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
            BCE_WEIGHT * bce_norm[i]
            + PAIR_WEIGHT * pair_norm[i]
            + VOTE_WEIGHT * vote_scores
        )

        choices.append(int(np.argmax(combined_scores)))

    return choices
