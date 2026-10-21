
"""MP2 E6: MLP Embedding Scorer."""

import numpy as np


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E6 requires --checkpoint")

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

    # MLP forward pass.
    hidden = np.maximum(x @ w1 + b1, 0)
    scores = hidden @ w2 + b2

    # Four candidate scores for each question.
    scores = scores.reshape(n, 4)

    choices = np.argmax(scores, axis=1)
    return [int(i) for i in choices]
