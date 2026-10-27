
"""E16-A: Agreement-gated voting with weighted BCE MLP.

Base checkpoint: E14-B weighted MLP.
No additional training.
"""

from collections import Counter
import numpy as np

from common import parse_answer


INPUT_DIM = 1024
CANDIDATES = 4
VOTE_WEIGHT = 0.5

# Fixed before official validation evaluation.
CONFIDENCE_THRESHOLD = 1.0
EPS = 1e-6


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E16-A requires a checkpoint")

    embeddings = arrays["embeddings"].astype(np.float32)
    n = len(bank)

    if embeddings.shape != (n, CANDIDATES, INPUT_DIM):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

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
            parse_answer(c["text"])
            for c in row["candidates"]
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
            dtype=np.float32,
        )

        raw_scores = scores[i]
        top_index = int(np.argmax(raw_scores))

        # Scorer's confidence in its top candidate.
        sorted_scores = np.sort(raw_scores)
        gap = float(sorted_scores[-1] - sorted_scores[-2])
        scale = max(float(raw_scores.std()), EPS)
        normalized_gap = gap / scale

        top_answer = answers[top_index]
        top_votes = (
            counts[top_answer]
            if top_answer is not None else 0
        )
        max_votes = max(counts.values(), default=0)

        # Protect a confident minority candidate from voting.
        confident_minority = (
            top_votes < max_votes
            and normalized_gap >= CONFIDENCE_THRESHOLD
        )

        effective_weight = (
            0.0 if confident_minority else VOTE_WEIGHT
        )

        combined = (
            raw_scores + effective_weight * vote_scores
        )

        choices.append(int(np.argmax(combined)))

    return choices
