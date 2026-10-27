
"""E16-C: OOF-trained logistic voting gate.

Base scorer: E14-B hard-case weighted MLP.
Gate: learned from TRAIN-only OOF disagreements.
"""

from collections import Counter

import numpy as np

from common import parse_answer


INPUT_DIM = 1024
CANDIDATES = 4
VOTE_WEIGHT = 0.5
EPS = 1e-6


def make_vote_scores(bank):
    votes = np.zeros((len(bank), CANDIDATES), dtype=np.float32)

    for i, row in enumerate(bank):
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]

        counts = Counter(
            answer for answer in answers
            if answer is not None
        )

        for j, answer in enumerate(answers):
            votes[i, j] = (
                counts[answer] / 4.0
                if answer is not None else 0.0
            )

    return votes


def make_gate_features(scores, votes, raw_choice, voted_choice):
    """Must match gate_features in train_e16c.py exactly."""
    n = len(scores)
    rows = np.arange(n)

    scale = np.maximum(scores.std(axis=1), EPS)

    sorted_scores = np.sort(scores, axis=1)
    top_gap = (
        sorted_scores[:, -1] - sorted_scores[:, -2]
    ) / scale

    raw_advantage = (
        scores[rows, raw_choice]
        - scores[rows, voted_choice]
    ) / scale

    vote_advantage = (
        votes[rows, voted_choice]
        - votes[rows, raw_choice]
    )

    max_votes = votes.max(axis=1)
    raw_support = votes[rows, raw_choice]

    return np.stack(
        [
            raw_advantage,
            top_gap,
            vote_advantage,
            max_votes,
            raw_support,
        ],
        axis=1,
    ).astype(np.float32)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E16-C requires a checkpoint")

    embeddings = arrays["embeddings"].astype(np.float32)
    n = len(bank)

    if embeddings.shape != (n, CANDIDATES, INPUT_DIM):
        raise ValueError(
            f"Unexpected embeddings shape: {embeddings.shape}"
        )

    with np.load(checkpoint, allow_pickle=False) as model:
        mean = model["base_mean"]
        std = model["base_std"]
        w1 = model["base_w1"]
        b1 = model["base_b1"]
        w2 = model["base_w2"]
        b2 = model["base_b2"]

        gate_w = model["gate_w"]
        gate_b = model["gate_b"]
        gate_mean = model["gate_mean"]
        gate_std = model["gate_std"]

    flat = embeddings.reshape(-1, INPUT_DIM)
    flat = (flat - mean) / std

    hidden = np.maximum(flat @ w1 + b1, 0.0)

    scores = (
        hidden @ w2 + b2
    ).reshape(n, CANDIDATES)

    votes = make_vote_scores(bank)

    raw_choice = np.argmax(scores, axis=1)
    voted_choice = np.argmax(
        scores + VOTE_WEIGHT * votes,
        axis=1,
    )

    # When both methods agree, no gate is needed.
    selected = voted_choice.copy()
    disagreement = raw_choice != voted_choice

    if np.any(disagreement):
        features = make_gate_features(
            scores, votes, raw_choice, voted_choice
        )

        standardized = (
            features[disagreement] - gate_mean
        ) / gate_std

        logits = (
            standardized @ gate_w
            + float(gate_b.reshape(-1)[0])
        )

        probability_trust_raw = (
            1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        )

        selected[disagreement] = np.where(
            probability_trust_raw >= 0.5,
            raw_choice[disagreement],
            voted_choice[disagreement],
        )

    return selected.astype(int).tolist()
