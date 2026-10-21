
"""MP2 - E3: Majority Voting + Scorer Tie-Breaking."""

from collections import Counter

import numpy as np

from common import parse_answer


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("E3 requires --checkpoint")

    # Load the Logistic Regression Scorer trained in E2.
    with np.load(checkpoint, allow_pickle=False) as model:
        w = model["weight"]
        b = float(model["bias"])
        mean = model["mean"]
        std = model["std"]

    x = arrays["embeddings"].astype(np.float32)
    x = (x - mean) / std

    # Scores for every candidate, shape (N, 4).
    scores = x @ w + b

    choices = []

    for row, candidate_scores in zip(bank, scores):
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]

        counts = Counter(
            answer for answer in answers
            if answer is not None
        )

        # No extractable answer: choose highest Scorer score.
        if not counts:
            choices.append(int(np.argmax(candidate_scores)))
            continue

        max_votes = max(counts.values())

        winning_answers = [
            answer for answer, count in counts.items()
            if count == max_votes
        ]

        if len(winning_answers) == 1:
            # Unique voting winner: keep the E1 choice.
            selected_answer = winning_answers[0]
            selected = answers.index(selected_answer)

        else:
            # Tie: compare mean Scorer scores per answer group.
            group_scores = []

            for answer in winning_answers:
                indices = [
                    i for i, a in enumerate(answers)
                    if a == answer
                ]
                avg_score = float(
                    np.mean(candidate_scores[indices])
                )
                group_scores.append(avg_score)

            best_group = int(np.argmax(group_scores))
            selected_answer = winning_answers[best_group]

            # Choose the highest-scoring candidate in that group.
            eligible = [
                i for i, a in enumerate(answers)
                if a == selected_answer
            ]

            selected = max(
                eligible,
                key=lambda i: (float(candidate_scores[i]), -i)
            )

        choices.append(int(selected))

    return choices
