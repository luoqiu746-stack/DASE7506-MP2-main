
"""MP2 E8: Confidence-Aware Adaptive Voting Fusion."""

from collections import Counter

import numpy as np

from common import parse_answer


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E8 requires --checkpoint")

    # Reuse the trained E6-C MLP.
    with np.load(checkpoint, allow_pickle=False) as model:
        w1 = model["w1"]
        b1 = model["b1"]
        w2 = model["w2"]
        b2 = model["b2"]
        mean = model["mean"]
        std = model["std"]

    n = len(bank)
    x = arrays["embeddings"].astype(np.float32)
    x = x.reshape(-1, 1024)
    x = (x - mean) / std

    hidden = np.maximum(x @ w1 + b1, 0)
    mlp_scores = (hidden @ w2 + b2).reshape(n, 4)

    choices = []

    for row, scores in zip(bank, mlp_scores):

        # 1. Calculate the gap between top two MLP scores.
        sorted_scores = np.sort(scores)
        gap = max(
            0.0,
            float(sorted_scores[-1] - sorted_scores[-2])
        )

        # 2. Adaptive voting weight.
        vote_weight = 1.0 / (1.0 + gap)

        # 3. Parse the four final answers.
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

        # 4. Fuse MLP scores with adaptive voting.
        final_scores = scores + vote_weight * vote_scores

        choices.append(int(np.argmax(final_scores)))

    return choices
