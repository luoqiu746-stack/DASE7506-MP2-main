
"""DASE7506 MP2 - E25 Group-Normalized Candidate Scorer."""

import numpy as np


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("Checkpoint is required")

    x = np.asarray(
        arrays["embeddings"],
        dtype=np.float32,
    )

    n = len(bank)

    if x.shape != (n, 4, 1024):
        raise ValueError(f"Invalid embeddings shape: {x.shape}")

    with np.load(checkpoint, allow_pickle=False) as model:
        mean = model["mean"]
        std = model["std"]
        w1 = model["w1"]
        b1 = model["b1"]
        w2 = model["w2"]
        b2 = model["b2"]

    flat = x.reshape(-1, 1024)
    flat = (flat - mean) / std

    hidden = np.maximum(
        flat @ w1 + b1,
        0.0,
    )

    scores = (
        hidden @ w2 + b2
    ).reshape(n, 4)

    return np.argmax(
        scores, axis=1
    ).astype(int).tolist()
