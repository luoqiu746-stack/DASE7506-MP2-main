
"""DASE7506 MP2 - E24 Question-Relative Embedding Scorer."""

import numpy as np


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("Checkpoint required")

    x = np.asarray(
        arrays["embeddings"], dtype=np.float32
    )

    n = len(bank)

    if x.shape != (n, 4, 1024):
        raise ValueError(
            f"Invalid input shape: {x.shape}"
        )

    question_mean = x.mean(axis=1, keepdims=True)
    relative = x - question_mean

    features = np.concatenate(
        [x, relative], axis=-1
    ).astype(np.float32)

    with np.load(checkpoint, allow_pickle=False) as model:
        mean = model["mean"]
        std = model["std"]
        w1 = model["w1"]
        b1 = model["b1"]
        w2 = model["w2"]
        b2 = model["b2"]

    flat = features.reshape(-1, 2048)
    flat = (flat - mean) / std

    hidden = np.maximum(
        flat @ w1 + b1, 0.0
    )

    scores = (
        hidden @ w2 + b2
    ).reshape(n, 4)

    return np.argmax(scores, axis=1).astype(int).tolist()
