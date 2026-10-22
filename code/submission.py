
"""MP2 E10: Bradley-Terry Pairwise Ranking MLP."""

import numpy as np


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E10 requires --checkpoint")

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
    scores = (hidden @ w2 + b2).reshape(n, 4)

    selected = np.argmax(scores, axis=1)

    return [int(i) for i in selected]
