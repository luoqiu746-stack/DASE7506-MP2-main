
"""E16-C: Train-only learned gate for weighted MLP voting.

Generate out-of-fold (OOF) predictions on official TRAIN.
Train a small logistic gate only on cases where plain
scoring and fixed voting choose different candidates.

Does not use official validation references.
Does not change the final E14-B checkpoint.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from common import load_inputs, parse_answer
from train_e14 import (
    load_training_data,
    question_weights,
    initialize,
    forward,
    train,
)


VOTE_WEIGHT = 0.5
EPS = 1e-6
N_FOLDS = 5
INPUT_DIM = 1024


def get_vote_scores(bank):
    votes = np.zeros((len(bank), 4), dtype=np.float32)

    for i, row in enumerate(bank):
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]
        counts = Counter(
            a for a in answers if a is not None
        )

        for j, answer in enumerate(answers):
            votes[i, j] = (
                counts[answer] / 4.0
                if answer is not None else 0.0
            )

    return votes


def gate_features(scores, votes, raw_choice, voted_choice):
    """Features available at prediction time."""
    n = len(scores)
    r = np.arange(n)

    scale = np.maximum(scores.std(axis=1), EPS)

    sorted_scores = np.sort(scores, axis=1)
    top_gap = (
        sorted_scores[:, -1] - sorted_scores[:, -2]
    ) / scale

    # Positive when raw scorer prefers its own choice.
    raw_advantage = (
        scores[r, raw_choice]
        - scores[r, voted_choice]
    ) / scale

    # Positive when voting prefers the voted choice.
    vote_advantage = (
        votes[r, voted_choice]
        - votes[r, raw_choice]
    )

    max_votes = votes.max(axis=1)

    raw_support = votes[r, raw_choice]

    return np.stack(
        [
            raw_advantage,
            top_gap,
            vote_advantage,
            max_votes,
            raw_support,
        ],
        axis=1,
    ).astype(np.float32)


def score_model(x, params):
    _, _, scores = forward(x, params)
    return scores.astype(np.float32)


def sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_gate(x, labels, regularization=0.1):
    """Small L2-regularized logistic regression in NumPy."""
    feature_mean = x.mean(axis=0)
    feature_std = np.maximum(x.std(axis=0), EPS)

    z = (x - feature_mean) / feature_std

    w = np.zeros(z.shape[1], dtype=np.float64)

    positive_rate = float(labels.mean())
    positive_rate = np.clip(
        positive_rate, 1e-4, 1.0 - 1e-4
    )

    b = float(np.log(
        positive_rate / (1.0 - positive_rate)
    ))

    # Fixed optimizer settings, no validation-label tuning.
    for _ in range(1500):
        p = sigmoid(z @ w + b)
        error = p - labels

        grad_w = (
            z.T @ error / len(labels)
            + regularization * w
        )
        grad_b = float(error.mean())

        w -= 0.05 * grad_w
        b -= 0.05 * grad_b

    return {
        "gate_w": w.astype(np.float32),
        "gate_b": np.asarray([b], dtype=np.float32),
        "gate_mean": feature_mean.astype(np.float32),
        "gate_std": feature_std.astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/e14b-weighted-mlp.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/e16c-gated-mlp.npz"),
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    bank, _ = load_inputs("train")
    x, labels, correct_counts = load_training_data()

    if len(bank) != len(x):
        raise ValueError("Training data mismatch")

    weights = question_weights(
        correct_counts, "weighted"
    )
    votes = get_vote_scores(bank)

    rng = np.random.default_rng(args.seed)
    folds = np.array_split(
        rng.permutation(len(x)),
        N_FOLDS
    )

    oof_scores = np.zeros(
        (len(x), 4), dtype=np.float32
    )

    print("\n=== Phase 1: OOF weighted MLP scoring ===")

    for fold_id, held_out in enumerate(folds, start=1):
        training = np.concatenate(
            [
                folds[j]
                for j in range(N_FOLDS)
                if j != fold_id - 1
            ]
        )

        x_train = x[training]
        y_train = labels[training]
        w_train = weights[training]

        flat = x_train.reshape(-1, INPUT_DIM)
        mean = flat.mean(axis=0)
        std = np.maximum(flat.std(axis=0), EPS)

        x_train = (x_train - mean) / std
        x_held = (x[held_out] - mean) / std

        params = initialize(args.seed, 32)

        print(
            f"\nFold {fold_id}/{N_FOLDS}: "
            f"train={len(training)}, "
            f"held-out={len(held_out)}",
            flush=True,
        )

        # Fixed at E14-B's previously selected 10 epochs.
        # Do not tune using official validation labels.
        train(
            x_train,
            y_train,
            w_train,
            params,
            epochs=10,
            batch_size=64,
            lr=0.001,
            weight_decay=0.01,
            seed=args.seed,
        )

        oof_scores[held_out] = score_model(
            x_held, params
        )

    print("\n=== Phase 2: identify disagreement cases ===")

    raw_choice = np.argmax(oof_scores, axis=1)

    voted_choice = np.argmax(
        oof_scores + VOTE_WEIGHT * votes,
        axis=1,
    )

    disagreement = raw_choice != voted_choice
    indices = np.flatnonzero(disagreement)

    print(f"TRAIN questions: {len(x)}")
    print(f"Disagreement questions: {len(indices)}")

    if len(indices) < 30:
        print(
            "\nWARNING: Fewer than 30 disagreement examples. "
            "The learned gate may be too unstable."
        )

    raw_correct = labels[
        indices, raw_choice[indices]
    ] > 0.5

    voted_correct = labels[
        indices, voted_choice[indices]
    ] > 0.5

    raw_only = raw_correct & ~voted_correct
    voted_only = ~raw_correct & voted_correct

    print(f"Raw scorer only correct: {raw_only.sum()}")
    print(f"Voting only correct: {voted_only.sum()}")
    print(
        "Same correctness (uninformative):",
        int(
            (~(raw_only | voted_only)).sum()
        ),
    )

    informative = raw_only | voted_only

    if informative.sum() < 20:
        print(
            "\nInsufficient informative disagreements. "
            "No gate checkpoint created."
        )
        return

    features = gate_features(
        oof_scores, votes, raw_choice, voted_choice
    )

    gate_x = features[indices][informative]
    gate_y = raw_only[informative].astype(np.float64)

    print("\n=== Phase 3: train logistic gate ===")
    print(f"Informative examples: {len(gate_y)}")
    print(
        f"Trust-raw proportion: {gate_y.mean():.4f}"
    )

    gate = fit_gate(gate_x, gate_y)

    # Package the already trained full-TRAIN E14-B model
    # together with the newly trained gate.
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    package = {}

    with np.load(
        args.checkpoint,
        allow_pickle=False,
    ) as base:
        for key in (
            "w1", "b1", "w2", "b2", "mean", "std"
        ):
            package[f"base_{key}"] = base[key].copy()

    package.update(gate)

    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )

    np.savez(args.output, **package)

    print(f"\nSaved checkpoint: {args.output}")
    print(
        f"Checkpoint size: "
        f"{args.output.stat().st_size:,} bytes"
    )


if __name__ == "__main__":
    main()
