
"""E17: Per-question pairwise ranking scorer.

Supports E17-A and E17-B checkpoints.
No voting; selects candidate with the highest MLP score.
"""

import numpy as np


INPUT_DIM = 1024
CANDIDATES = 4


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E17 requires a checkpoint")

    embeddings = arrays["embeddings"].astype(np.float32)
    n = len(bank)

    if embeddings.shape != (n, CANDIDATES, INPUT_DIM):
        raise ValueError(
            f"Unexpected embeddings shape: {embeddings.shape}"
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

    return np.argmax(scores, axis=1).astype(int).tolist()
