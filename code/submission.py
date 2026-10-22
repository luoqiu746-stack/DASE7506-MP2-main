
"""MP2 E9: Confidence-Weighted Answer Voting."""

from collections import defaultdict

import numpy as np

from common import parse_answer


def scorer_logits(arrays, checkpoint):
    """Return candidate scores for either E2 Logistic or E6-C MLP."""

    x = arrays["embeddings"].astype(np.float32)
    n = x.shape[0]
    x = x.reshape(-1, 1024)

    with np.load(checkpoint, allow_pickle=False) as model:
        mean = model["mean"]
        std = model["std"]
        x = (x - mean) / std

        if "weight" in model.files:
            # E2: Logistic Regression
            scores = x @ model["weight"] + model["bias"]

        elif "w1" in model.files:
            # E6-C: MLP
            hidden = np.maximum(
                x @ model["w1"] + model["b1"], 0
            )
            scores = hidden @ model["w2"] + model["b2"]

        else:
            raise ValueError("Unknown checkpoint format")

    return scores.reshape(n, 4)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E9 requires --checkpoint")

    scores_all = scorer_logits(arrays, checkpoint)
    choices = []

    for row, scores in zip(bank, scores_all):

        # Convert raw scorer logits to values between 0 and 1.
        clipped = np.clip(scores, -30, 30)
        confidence = 1.0 / (1.0 + np.exp(-clipped))

        # Group candidates by their normalized final answer.
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]

        answer_scores = defaultdict(float)

        for answer, probability in zip(answers, confidence):
            if answer is not None:
                answer_scores[answer] += float(probability)

        # If no candidate has a parseable answer, use scorer only.
        if not answer_scores:
            choices.append(int(np.argmax(scores)))
            continue

        # Pick the answer with the largest total confidence.
        # On ties, prefer the best individual scorer value.
        def answer_key(answer):
            best = max(
                float(scores[i])
                for i, a in enumerate(answers)
                if a == answer
            )
            return (answer_scores[answer], best)

        winning_answer = max(answer_scores, key=answer_key)

        # Choose the highest-scoring candidate supporting that answer.
        eligible = [
            i for i, answer in enumerate(answers)
            if answer == winning_answer
        ]

        selected = max(
            eligible,
            key=lambda i: float(scores[i])
        )

        choices.append(int(selected))

    return choices
