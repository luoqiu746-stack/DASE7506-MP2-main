
"""E14: MLP candidate selection.

Supports both:
E14-A ordinary BCE
E14-B hard-case weighted BCE

Uses only frozen embeddings and saved checkpoint.
"""

import numpy as np


INPUT_DIM = 1024
CANDIDATES = 4


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E14 requires a checkpoint")

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

    hidden = np.maximum(
        flat @ w1 + b1,
        0.0
    )

    logits = (
        hidden @ w2 + b2
    ).reshape(n, CANDIDATES)

    selected = np.argmax(logits, axis=1)

    return selected.astype(int).tolist()
