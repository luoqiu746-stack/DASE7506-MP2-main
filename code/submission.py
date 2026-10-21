
"""E2: Select the candidate with the highest learned score."""

from pathlib import Path
import numpy as np


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E2 requires --checkpoint")

    checkpoint = Path(checkpoint)

    with np.load(checkpoint, allow_pickle=False) as model:
        w = model["weight"]
        b = float(model["bias"])
        mean = model["mean"]
        std = model["std"]

    # Shape: (number of questions, 4, 1024)
    x = arrays["embeddings"].astype(np.float32)

    # Use the same train-only normalization.
    x = (x - mean) / std

    # Score each of the four candidates.
    scores = x @ w + b  # Shape: (N, 4)

    # Highest score wins; ties favor lowest candidate index.
    choices = np.argmax(scores, axis=1)

    # Official interface requires ordinary Python integers.
    return [int(i) for i in choices]
