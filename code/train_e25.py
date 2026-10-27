
"""E25: Multi-Positive Group-Normalized Candidate Selection.

Official TRAIN only.
Internal dev comparison + full TRAIN checkpoint.

Model: 1024 -> 32 -> 1 (ReLU)
Seed: 17
Epochs: 10
Output: runs/e25-group-selection.npz
"""

import json
import time
from pathlib import Path

import numpy as np

from common import load_inputs, parse_answer
from train_e14 import load_training_data, question_weights

SEED = 17
HIDDEN = 32
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 0.01
EPS = 1e-6

CHECKPOINT = Path("runs/e25-group-selection.npz")
REPORT = Path("runs/e25-internal-dev.json")


def detect_disagreement(bank):
    flags = []

    for row in bank:
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]
        valid = [a for a in answers if a is not None]
        flags.append(len(set(valid)) >= 2)

    return np.asarray(flags, dtype=bool)


def initialize(dimension):
    rng = np.random.default_rng(SEED)

    return {
        "w1": (
            rng.standard_normal((dimension, HIDDEN))
            * np.sqrt(2.0 / dimension)
        ).astype(np.float32),
        "b1": np.zeros(HIDDEN, dtype=np.float32),
        "w2": (
            rng.standard_normal((HIDDEN, 1))
            * np.sqrt(2.0 / HIDDEN)
        ).astype(np.float32),
        "b2": np.zeros(1, dtype=np.float32),
    }


def forward(x, params):
    flat = x.reshape(-1, x.shape[-1])

    pre = flat @ params["w1"] + params["b1"]
    hidden = np.maximum(pre, 0.0)

    scores = (
        hidden @ params["w2"] + params["b2"]
    ).reshape(-1, 4)

    return scores, flat, pre, hidden


def loss_gradient(scores, targets, weights, objective):
    """Return scalar loss and gradient wrt candidate logits."""

    if objective == "bce":
        element = (
            np.logaddexp(0.0, scores)
            - targets * scores
        )
        question_loss = element.mean(axis=1)

        denom = max(float(weights.sum()), EPS)

        loss = float(
            (question_loss * weights).sum() / denom
        )

        sigmoid = 1.0 / (
            1.0 + np.exp(
                -np.clip(scores, -40.0, 40.0)
            )
        )

        gradient = (
            (sigmoid - targets)
            * weights[:, None] / (4.0 * denom)
        )

    elif objective == "group":
        # Stable softmax probabilities over four candidates.
        z = scores.astype(np.float64)
        z -= z.max(axis=1, keepdims=True)

        exp_z = np.exp(z)
        probabilities = (
            exp_z / exp_z.sum(axis=1, keepdims=True)
        )

        positives = targets > 0.5
        correct_count = positives.sum(axis=1)

        # All-correct groups convey no candidate-selection
        # information. Exclude these from the loss.
        informative = (
            (correct_count >= 1)
            & (correct_count < 4)
        )

        positive_mass = (
            probabilities * positives
        ).sum(axis=1)

        safe_mass = np.maximum(positive_mass, 1e-12)

        question_loss = -np.log(safe_mass)

        effective_weights = (
            weights * informative
        )

        denom = max(
            float(effective_weights.sum()), EPS
        )

        loss = float(
            (question_loss * effective_weights).sum()
            / denom
        )

        # Gradient:
        # p_i - 1[y_i=1] * p_i / P(correct group)
        conditional_positive = (
            probabilities * positives
            / safe_mass[:, None]
        )

        gradient = (
            probabilities - conditional_positive
        ) * (effective_weights[:, None] / denom)

    else:
        raise ValueError(f"Unknown objective: {objective}")

    return loss, gradient.astype(np.float32)


def train_model(x, y, weights, objective):
    params = initialize(x.shape[-1])

    first = {
        k: np.zeros_like(v)
        for k, v in params.items()
    }
    second = {
        k: np.zeros_like(v)
        for k, v in params.items()
    }

    rng = np.random.default_rng(SEED)
    step = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(len(x))
        losses = []

        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]

            scores, flat, pre, hidden = forward(
                x[ids], params
            )

            loss, grad_scores = loss_gradient(
                scores,
                y[ids],
                weights[ids],
                objective,
            )
            losses.append(loss)

            grad_out = grad_scores.reshape(-1, 1)

            dw2 = hidden.T @ grad_out
            db2 = grad_out.sum(axis=0)

            dhidden = grad_out @ params["w2"].T
            dpre = dhidden * (pre > 0.0)

            dw1 = flat.T @ dpre
            db1 = dpre.sum(axis=0)

            grads = {
                "w1": dw1,
                "b1": db1,
                "w2": dw2,
                "b2": db2,
            }

            step += 1

            # AdamW
            for key in params:
                g = np.asarray(
                    grads[key], dtype=np.float32
                )

                first[key] = (
                    0.9 * first[key] + 0.1 * g
                )
                second[key] = (
                    0.999 * second[key]
                    + 0.001 * (g * g)
                )

                m_hat = first[key] / (1.0 - 0.9 ** step)
                v_hat = second[key] / (1.0 - 0.999 ** step)

                if key in ("w1", "w2"):
                    params[key] *= (
                        1.0 - LR * WEIGHT_DECAY
                    )

                params[key] -= (
                    LR * m_hat
                    / (np.sqrt(v_hat) + 1e-8)
                ).astype(np.float32)

        print(
            f"{objective.upper():5s} "
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"Loss={np.mean(losses):.5f}"
        )

    return params


def evaluate(name, x, y, disagreement, params):
    scores, _, _, _ = forward(x, params)

    selected = np.argmax(scores, axis=1)
    rows = np.arange(len(y))

    correct = y[rows, selected] > 0.5
    counts = y.sum(axis=1)

    masks = {
        "overall": np.ones(len(y), dtype=bool),
        "disagreement": disagreement,
        "single_correct": counts == 1,
        "disagreement_single_correct": (
            disagreement & (counts == 1)
        ),
        "non_all_correct": counts < 4,
    }

    print(f"\n=== {name} ===")
    results = {}

    for metric, mask in masks.items():
        total = int(mask.sum())
        hits = int((correct & mask).sum())

        accuracy = (
            100.0 * hits / total
            if total else None
        )

        results[metric] = {
            "correct": hits,
            "total": total,
            "accuracy_percent": accuracy,
        }

        if total:
            print(
                f"{metric}: {hits}/{total} "
                f"= {accuracy:.4f}%"
            )

    return results


def normalize(x_train, x_dev=None):
    flat = x_train.reshape(-1, x_train.shape[-1])

    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), EPS)

    normalized_train = (x_train - mean) / std

    if x_dev is None:
        return normalized_train, mean, std

    normalized_dev = (x_dev - mean) / std

    return normalized_train, normalized_dev, mean, std


def main():
    started = time.perf_counter()

    bank, _ = load_inputs("train")
    x, y, correct_counts = load_training_data()

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if x.shape != (len(bank), 4, 1024):
        raise ValueError(f"Invalid input shape: {x.shape}")

    if y.shape != (len(bank), 4):
        raise ValueError(f"Invalid label shape: {y.shape}")

    disagreement = detect_disagreement(bank)

    weights = np.asarray(
        question_weights(correct_counts, "weighted"),
        dtype=np.float32,
    )

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(x))

    n_dev = max(1, int(len(x) * 0.1))
    dev_ids = order[:n_dev]
    train_ids = order[n_dev:]

    print("=== E25 Internal Development ===")
    print(f"Total TRAIN: {len(x)}")
    print(f"Internal train: {len(train_ids)}")
    print(f"Internal dev: {len(dev_ids)}")
    print(
        "Dev disagreement:",
        int(disagreement[dev_ids].sum()),
    )

    x_train, x_dev, mean, std = normalize(
        x[train_ids],
        x[dev_ids],
    )

    y_train = y[train_ids]
    y_dev = y[dev_ids]
    train_weights = weights[train_ids]
    dev_disagreement = disagreement[dev_ids]

    print("\n=== Phase 1: Weighted BCE Baseline ===")

    baseline = train_model(
        x_train,
        y_train,
        train_weights,
        objective="bce",
    )

    baseline_results = evaluate(
        "Weighted BCE baseline",
        x_dev,
        y_dev,
        dev_disagreement,
        baseline,
    )

    print("\n=== Phase 2: E25 Group-Normalized ===")

    group_model = train_model(
        x_train,
        y_train,
        train_weights,
        objective="group",
    )

    group_results = evaluate(
        "E25 internal dev",
        x_dev,
        y_dev,
        dev_disagreement,
        group_model,
    )

    print("\n=== Internal Comparison ===")

    for metric in baseline_results:
        a = baseline_results[metric]
        b = group_results[metric]

        print(
            f"{metric}: "
            f"baseline={a['correct']}/{a['total']} "
            f"E25={b['correct']}/{b['total']} "
            f"delta={b['correct'] - a['correct']:+d}"
        )

    print("\n=== Phase 3: Full TRAIN E25 ===")

    x_full, full_mean, full_std = normalize(x)

    full_model = train_model(
        x_full,
        y,
        weights,
        objective="group",
    )

    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        CHECKPOINT,
        **full_model,
        mean=full_mean.astype(np.float32),
        std=full_std.astype(np.float32),
    )

    elapsed = time.perf_counter() - started

    report = {
        "experiment": "E25",
        "objective": "multi-positive group-normalized loss",
        "seed": SEED,
        "epochs": EPOCHS,
        "internal_train": len(train_ids),
        "internal_dev": len(dev_ids),
        "baseline": baseline_results,
        "e25": group_results,
        "checkpoint": str(CHECKPOINT),
        "training_and_analysis_seconds": elapsed,
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== E25 Complete ===")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Checkpoint size: {CHECKPOINT.stat().st_size:,} bytes")
    print(f"Report: {REPORT}")
    print(f"Total seconds: {elapsed:.4f}")


def research_metrics(y, choices, disagreement):
    """Question-level accuracy; overlapping hard subsets are explicit."""
    ok = y[np.arange(len(y)), choices] > 0.5
    masks = {
        "overall": np.ones(len(y), bool),
        "disagreement": disagreement,
        "single_correct": y.sum(1) == 1,
        "disagreement_single_correct": disagreement & (y.sum(1) == 1),
        "non_all_correct": y.sum(1) < 4,
    }
    return {k: {"correct": int(ok[m].sum()), "total": int(m.sum()),
                "accuracy_percent": float(100 * ok[m].mean()) if m.any() else None}
            for k, m in masks.items()}


def research_decode(scores, answers, decoder):
    from compare_final_models import get_votes, predict_e16a, predict_e19b
    if decoder == "raw":
        return scores.argmax(1)
    if decoder == "gate":
        return predict_e16a(scores, answers, get_votes(answers))
    if decoder == "aggregate":
        return predict_e19b(scores, answers)
    raise ValueError(decoder)


def research_fit(x, y, weights, seed, decay, epochs, evaluate_x=None, saves=()):
    """E14 Adam+coupled L2; preserve optimizer state across saved epochs."""
    from train_e14 import initialize as init14, forward as forward14, bce_loss_and_grad
    p = init14(seed, 32)
    first = {k: np.zeros_like(v) for k, v in p.items()}
    second = {k: np.zeros_like(v) for k, v in p.items()}
    rng = np.random.default_rng(seed)
    step = 0
    saved = {}
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y))
        for start in range(0, len(y), 64):
            idx = order[start:start + 64]
            flat, h, logits = forward14(x[idx], p)
            _, ds = bce_loss_and_grad(logits, y[idx], weights[idx])
            ds = ds.reshape(-1, 1)
            dh = (ds @ p["w2"].T) * (h > 0)
            grads = {"w2": h.T @ ds + decay * p["w2"], "b2": ds.sum(0),
                     "w1": flat.T @ dh + decay * p["w1"], "b1": dh.sum(0)}
            step += 1
            for k, g in grads.items():
                first[k] = 0.9 * first[k] + 0.1 * g
                second[k] = 0.999 * second[k] + 0.001 * g * g
                p[k] -= 0.001 * (first[k] / (1 - 0.9**step)) / (
                    np.sqrt(second[k] / (1 - 0.999**step)) + 1e-8)
        if epoch in saves:
            elapsed = time.perf_counter() - started
            tick = time.perf_counter()
            scores = forward14(evaluate_x, p)[2].copy()
            saved[epoch] = {"scores": scores, "training_seconds": elapsed,
                            "prediction_seconds": time.perf_counter() - tick}
    return p, saved, time.perf_counter() - started


def research_normalize(x, train_idx, other_idx):
    flat = x[train_idx].reshape(-1, 1024)
    mean = flat.mean(0)
    std = np.maximum(flat.std(0), 1e-6)
    return (x[train_idx]-mean)/std, (x[other_idx]-mean)/std, mean, std


def research_main():
    """Bounded Train-only search, confirmation, refit; see saved PLAN.md."""
    import argparse
    import hashlib
    import sys
    from train_e14 import question_weights, initialize as init14, train as train14, forward as forward14
    import contextlib
    import io

    parser = argparse.ArgumentParser()
    parser.add_argument("--research", action="store_true", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.run_dir
    if not (out / "PLAN.md").exists():
        raise ValueError("Write the finite experiment plan before training")
    # This prevents accidental reruns from replacing any completed research run.
    with (out / "research-started.json").open("x") as f:
        json.dump({"seed": 20261027, "python": sys.version, "numpy": np.__version__}, f)
    started = time.perf_counter()
    bank, arrays = load_inputs("train")
    x, y, counts = load_training_data()
    del arrays
    disagreement = detect_disagreement(bank)
    answers = [[parse_answer(c["text"]) for c in row["candidates"]] for row in bank]
    weights = question_weights(counts, "weighted")
    rng = np.random.default_rng(20261027)
    dev, confirm = [], []
    fold_lists = [[], [], []]
    # Split by label count AND disagreement, never by candidate rows.
    for count in range(1, 5):
        for flag in (False, True):
            idx = rng.permutation(np.flatnonzero((counts == count) & (disagreement == flag)))
            n_confirm = round(len(idx) * 0.2)
            confirm.extend(idx[:n_confirm].tolist())
            remain = idx[n_confirm:]
            dev.extend(remain.tolist())
            for k, chunk in enumerate(np.array_split(remain, 3)):
                fold_lists[k].extend(chunk.tolist())
    dev, confirm = np.sort(dev), np.sort(confirm)
    folds = [np.sort(a) for a in fold_lists]
    assert not set(dev) & set(confirm) and len(dev) + len(confirm) == len(x)
    assert sorted(np.concatenate(folds).tolist()) == dev.tolist()
    split = {"development": dev.tolist(), "confirmation": confirm.tolist(),
             "folds": [a.tolist() for a in folds]}
    (out / "splits.json").write_text(json.dumps(split, indent=2))
    print(f"Development {len(dev)}; confirmation {len(confirm)}; folds {[len(a) for a in folds]}", flush=True)
    epochs = (6, 10, 16)
    seeds = (17, 29, 43)
    decays = (0.003, 0.01, 0.03)
    # Cache logits only for held-out TRAIN questions. Not an inference asset.
    cache = {}
    costs = []
    parity_checked = False
    for fold, valid in enumerate(folds):
        fitting = np.setdiff1d(dev, valid)
        xt, xv, _, _ = research_normalize(x, fitting, valid)
        for decay in decays:
            for seed in seeds:
                p, saved, elapsed = research_fit(xt, y[fitting], weights[fitting], seed,
                                                 decay, max(epochs), xv, epochs)
                cache[fold, decay, seed] = saved
                costs.append({"phase": "cv", "fold": fold, "decay": decay,
                              "seed": seed, "epochs": 16, "seconds": elapsed})
                print(f"CV fold{fold+1} L2={decay} seed={seed}: {elapsed:.3f}s", flush=True)
                # One exact recipe parity check catches optimizer/initialization drift.
                if not parity_checked and decay == 0.01 and seed == 17:
                    check = init14(seed, 32)
                    with contextlib.redirect_stdout(io.StringIO()):
                        train14(xt, y[fitting], weights[fitting], check, 16, 64, 0.001, decay, seed)
                    error = max(float(np.max(np.abs(check[k]-p[k]))) for k in p)
                    if error > 1e-6:
                        raise AssertionError(f"E14 parity error {error}")
                    (out / "optimizer-parity.json").write_text(json.dumps({"max_parameter_error": error}))
                    parity_checked = True
    records = []
    choices_by_name = {}
    configs = [("E26-baseline", 0.01, 10, (17,), decoder)
               for decoder in ("raw", "gate", "aggregate")]
    configs += [("E27-ensemble", d, ep, seeds, decoder) for d in decays
                for ep in epochs for decoder in ("raw", "gate", "aggregate")]
    def evaluate_config(label, decay, ep, model_seeds, decoder):
        name = f"{label}-l2{decay}-ep{ep}-{decoder}"
        choices = np.zeros(len(y), dtype=int)
        per_fold = []
        prediction_seconds = scoring_seconds = training_seconds = 0.0
        for fold, valid in enumerate(folds):
            entries = [cache[fold, decay, s][ep] for s in model_seeds]
            tick = time.perf_counter()
            scores = np.mean([e["scores"] for e in entries], axis=0)
            c = research_decode(scores, [answers[i] for i in valid], decoder)
            prediction_seconds += time.perf_counter()-tick + sum(e["prediction_seconds"] for e in entries)
            training_seconds += sum(e["training_seconds"] for e in entries)
            choices[valid] = c
            tick = time.perf_counter()
            per_fold.append(research_metrics(y[valid], c, disagreement[valid]))
            scoring_seconds += time.perf_counter()-tick
        metrics = research_metrics(y[dev], choices[dev], disagreement[dev])
        record = {"name": name, "decay": decay, "epochs": ep, "seeds": list(model_seeds),
                  "decoder": decoder, "metrics": metrics, "fold_metrics": per_fold,
                  "prediction_seconds": prediction_seconds, "scoring_seconds": scoring_seconds,
                  "training_seconds": training_seconds}
        records.append(record)
        choices_by_name[name] = choices
        return record
    for config in configs:
        evaluate_config(*config)
    baseline = next(r for r in records if r["name"].startswith("E26") and r["decoder"] == "gate")
    def priority(r):
        return np.mean([r["metrics"][k]["accuracy_percent"] for k in ("disagreement", "single_correct")])
    eligible = [r for r in records if r["name"].startswith("E27") and
                r["metrics"]["overall"]["accuracy_percent"] >= baseline["metrics"]["overall"]["accuracy_percent"]-0.5]
    selected = max(eligible, key=lambda r: (priority(r), r["metrics"]["overall"]["correct"], -r["epochs"], r["decay"]))
    # Add fixed seed ablations AFTER choosing the ensemble recipe.
    for seed in (17, 29, 43):
        evaluate_config(f"E27-ablation-seed{seed}", selected["decay"], selected["epochs"], (seed,), selected["decoder"])
    (out / "cv-results.json").write_text(json.dumps(records, indent=2))
    frozen = {"selected_cv": selected, "baseline": baseline, "selection": "Train CV only",
              "plan_sha256": hashlib.sha256((out / "PLAN.md").read_bytes()).hexdigest()}
    (out / "cv-lock.json").write_text(json.dumps(frozen, indent=2))
    print("LOCKED CV", selected["name"], selected["metrics"], flush=True)
    # Confirmation evaluated only after locking. Includes prespecified ablations.
    xt, xv, mean, std = research_normalize(x, dev, confirm)
    confirmation = {}
    confirmation_choices = {}
    def fit_bundle(name, cfg, model_seeds, weight_values):
        bundle = {"format_version": np.array([2]), "decoder": np.array(cfg["decoder"]),
                  "mean": mean, "std": std, "seeds": np.array(model_seeds)}
        all_scores = []
        elapsed = 0.0
        for j, seed in enumerate(model_seeds):
            p, _, cost = research_fit(xt, y[dev], weight_values[dev], seed,
                                      cfg["decay"], cfg["epochs"])
            elapsed += cost
            for k, v in p.items(): bundle[f"m{j}_{k}"] = v
            all_scores.append(forward14(xv, p)[2])
        tick = time.perf_counter()
        c = research_decode(np.mean(all_scores, axis=0), [answers[i] for i in confirm], cfg["decoder"])
        prediction = time.perf_counter()-tick
        tick = time.perf_counter()
        m = research_metrics(y[confirm], c, disagreement[confirm])
        scoring = time.perf_counter()-tick
        confirmation[name] = {"metrics": m, "training_seconds": elapsed,
                              "decoding_seconds": prediction, "scoring_seconds": scoring,
                              "configuration": cfg, "seeds": list(model_seeds)}
        confirmation_choices[name] = c
        with (out / f"confirmation-{name}.npz").open("xb") as f: np.savez(f, **bundle)
        return all_scores
    candidate_scores = fit_bundle("candidate", selected, seeds, weights)
    fit_bundle("baseline", baseline, (17,), weights)
    fit_bundle("uniform-ablation", selected, seeds, np.ones(len(y), np.float32))
    single = research_decode(candidate_scores[0], [answers[i] for i in confirm], selected["decoder"])
    confirmation["single-seed-ablation"] = {"metrics": research_metrics(y[confirm], single, disagreement[confirm]),
                                               "seed": 17, "training_seconds": None}
    c_metrics = confirmation["candidate"]["metrics"]
    b_metrics = confirmation["baseline"]["metrics"]
    accepted = all(c_metrics[k]["accuracy_percent"] >= b_metrics[k]["accuracy_percent"]
                   for k in ("disagreement", "single_correct")) and (
        c_metrics["overall"]["accuracy_percent"] >= b_metrics["overall"]["accuracy_percent"]-0.5)
    final_cfg = selected if accepted else baseline
    final_seeds = seeds if accepted else (17,)
    # Paired, question-level bootstrap. The hard subsets overlap by design.
    rng_ci = np.random.default_rng(20261028)
    yc = y[confirm]
    cand_ok = yc[np.arange(len(confirm)), confirmation_choices["candidate"]]
    base_ok = yc[np.arange(len(confirm)), confirmation_choices["baseline"]]
    diff = cand_ok-base_ok
    ci = {}
    for name, mask in {"overall": np.ones(len(confirm), bool), "disagreement": disagreement[confirm],
                       "single_correct": yc.sum(1)==1}.items():
        delta = diff[mask]
        draws = delta[rng_ci.integers(0, len(delta), (5000, len(delta)))].mean(1)*100
        ci[name] = {"delta_pp": float(delta.mean()*100),
                    "paired_bootstrap_95_percent": np.percentile(draws, [2.5,97.5]).tolist()}
    (out / "confirmation-results.json").write_text(json.dumps(
        {"results": confirmation, "accepted": bool(accepted), "paired_intervals": ci}, indent=2))
    np.savez(out / "train-heldout-choices.npz", dev=dev, confirm=confirm,
             labels=y, disagreement=disagreement,
             **{r["name"]: choices_by_name[r["name"]][dev] for r in records},
             **{f"confirmation_{k}": v for k, v in confirmation_choices.items()})
    print("CONFIRMATION", json.dumps({"accepted": bool(accepted), "candidate": c_metrics,
          "baseline": b_metrics, "intervals": ci}), flush=True)
    # All decisions frozen before any official Validation/Test score.
    (out / "final-lock.json").write_text(json.dumps({"accepted": bool(accepted), "config": final_cfg,
         "seeds": list(final_seeds), "reason": "predeclared confirmation gate"}, indent=2))
    full_mean = x.reshape(-1,1024).mean(0)
    full_std = np.maximum(x.reshape(-1,1024).std(0), 1e-6)
    full = (x-full_mean)/full_std
    refit_cost = {}
    for name, cfg, use_seeds in (("final", final_cfg, final_seeds), ("baseline", baseline, (17,))):
        bundle = {"format_version": np.array([2]), "decoder": np.array(cfg["decoder"]),
                  "mean": full_mean, "std": full_std, "seeds": np.array(use_seeds)}
        elapsed = 0.0
        for j, seed in enumerate(use_seeds):
            p, _, cost = research_fit(full, y, weights, seed, cfg["decay"], cfg["epochs"])
            elapsed += cost
            for k, v in p.items(): bundle[f"m{j}_{k}"] = v
        with (out / f"{name}.npz").open("xb") as f: np.savez(f, **bundle)
        refit_cost[name] = elapsed
    (out / "training-costs.json").write_text(json.dumps({"cv_trajectories": costs,
       "refit_seconds": refit_cost, "total_research_wall_seconds": time.perf_counter()-started,
       "note": "CV timing includes epoch snapshot prediction; parity check included in total wall time. Confirmation training costs in confirmation-results.json."}, indent=2))
    print("FULL TRAIN COMPLETE", refit_cost, flush=True)


if __name__ == "__main__":
    import sys
    if "--research" in sys.argv:
        research_main()
    else:
        main()
