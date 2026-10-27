
"""E19-B: Answer-level probability aggregation.

Uses E14-B weighted BCE MLP and a temperature
selected using official TRAIN internal-dev data.
"""

from collections import defaultdict

import numpy as np

from common import parse_answer


INPUT_DIM = 1024
CANDIDATES = 4


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E19 requires a checkpoint")

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
        temperature = float(model["temperature"][0])

    flat = embeddings.reshape(-1, INPUT_DIM)
    flat = (flat - mean) / std

    hidden = np.maximum(
        flat @ w1 + b1, 0.0
    )

    scores = (
        hidden @ w2 + b2
    ).reshape(n, CANDIDATES)

    choices = []

    for i, row in enumerate(bank):
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]

        logits = scores[i]

        z = logits.astype(np.float64) / temperature
        z -= z.max()

        probabilities = np.exp(z)
        probabilities /= probabilities.sum()

        groups = defaultdict(list)

        for j, answer in enumerate(answers):
            if answer is None:
                key = ("unparsed", j)
            else:
                key = ("answer", answer)

            groups[key].append(j)

        best_group = max(
            groups.values(),
            key=lambda indices: (
                sum(probabilities[j] for j in indices),
                max(float(logits[j]) for j in indices),
            ),
        )

        chosen = max(
            best_group,
            key=lambda j: float(logits[j]),
        )

        choices.append(int(chosen))

    return choices
