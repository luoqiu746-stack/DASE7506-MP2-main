
"""E22-A: Hard Validation Audit.

Compare existing models on answer-disagreement questions.

Models:
    E14-B: Weighted BCE
    E16-A: Agreement-Gated Voting
    E19-B: Answer-Level Aggregation
    E21: BCE + Ranking

Read-only experiment. Does not train or modify checkpoints.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer


PREDICTION_FILES = {
    "E14-B": "runs/e14b-validation-predictions.json",
    "E16-A": "runs/e16a-validation-predictions.json",
    "E19-B": "runs/e19-validation-predictions.json",
    "E21": "runs/e21-validation-predictions.json",
    "E22": "runs/e22-validation-predictions.json",
    "E24": "runs/e24-validation-predictions.json",
    "E25": "runs/e25-validation-predictions.json",
}

OUTPUT = Path("runs/e22-hard-validation-analysis.json")


def load_predictions(path, bank):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if obj.get("split") != "validation":
        raise ValueError(f"Incorrect split: {path}")

    expected_ids = [row["id"] for row in bank]

    if obj.get("ids") != expected_ids:
        raise ValueError(f"Question IDs do not match: {path}")

    selected = np.asarray(obj["selected"], dtype=int)

    if selected.shape != (len(bank),):
        raise ValueError(
            f"Incorrect prediction shape: {path}"
        )

    if np.any((selected < 0) | (selected > 3)):
        raise ValueError(f"Invalid candidate index: {path}")

    return selected


def load_references(bank):
    with (DATA / "gsm8k_validation.jsonl").open(
        "r", encoding="utf-8"
    ) as f:
        refs = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if len(bank) != len(refs):
        raise ValueError("Reference count mismatch")

    return refs


def answer_pattern(answers):
    """Return a vote pattern such as 3-1 or 2-1-1.

    Only parsed answers are grouped. Each unparseable
    candidate is its own group, not a shared None group.
    """
    keys = []

    for j, answer in enumerate(answers):
        if answer is None:
            keys.append(("unparsed", j))
        else:
            keys.append(("answer", answer))

    counts = Counter(keys)
    sizes = sorted(counts.values(), reverse=True)

    return "-".join(str(v) for v in sizes)


def get_metrics(ok, mask):
    total = int(mask.sum())

    if total == 0:
        return {
            "correct": 0,
            "total": 0,
            "accuracy_percent": None,
        }

    hits = int((ok & mask).sum())

    return {
        "correct": hits,
        "total": total,
        "accuracy_percent": 100.0 * hits / total,
    }


def show_metrics(label, data):
    if data["total"] == 0:
        print(f"  {label}: N/A")
    else:
        print(
            f"  {label}: "
            f"{data['correct']}/{data['total']} "
            f"= {data['accuracy_percent']:.2f}%"
        )


def main():
    print("=== E22-A: Hard Validation Audit ===")

    bank, _ = load_inputs("validation")
    refs = load_references(bank)

    n = len(bank)

    correct = np.zeros((n, 4), dtype=bool)
    disagreement = np.zeros(n, dtype=bool)
    patterns = []
    unparsed_counts = []

    for i, (row, ref) in enumerate(zip(bank, refs)):
        if (
            row["id"] != ref["id"]
            or row["question"] != ref["question"]
        ):
            raise ValueError(
                f"Reference alignment error at row {i}"
            )

        gold = parse_answer(ref["answer"])

        if gold is None:
            raise ValueError(
                f"Cannot parse gold answer at row {i}"
            )

        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]

        for j, answer in enumerate(answers):
            correct[i, j] = (
                answer is not None and answer == gold
            )

        parsed_answers = [
            a for a in answers if a is not None
        ]

        # At least two DISTINCT parsed final answers.
        disagreement[i] = len(set(parsed_answers)) >= 2

        patterns.append(answer_pattern(answers))
        unparsed_counts.append(
            sum(a is None for a in answers)
        )

    correct_counts = correct.sum(axis=1)

    if np.any(correct_counts == 0):
        raise ValueError(
            "Found a question without a correct candidate"
        )

    single_correct = correct_counts == 1
    non_all_correct = correct_counts < 4

    print("\n=== 1. Dataset composition ===")
    print(f"Total questions: {n}")
    print(f"Disagreement questions: {int(disagreement.sum())}")
    print(f"Single-correct questions: {int(single_correct.sum())}")
    print(f"Non-all-correct questions: {int(non_all_correct.sum())}")
    print(
        "Questions containing unparseable candidates:",
        int(np.sum(np.asarray(unparsed_counts) > 0)),
    )

    print("\nAnswer vote patterns:")
    for pattern, count in Counter(patterns).most_common():
        print(f"  {pattern}: {count}")

    rows = np.arange(n)

    predictions = {}
    success = {}
    results = {}

    print("\n=== 2. Model accuracy comparison ===")

    for name, filename in PREDICTION_FILES.items():
        selected = load_predictions(filename, bank)
        ok = correct[rows, selected]

        predictions[name] = selected
        success[name] = ok

        metrics = {
            "overall": get_metrics(
                ok, np.ones(n, dtype=bool)
            ),
            "disagreement": get_metrics(ok, disagreement),
            "single_correct": get_metrics(ok, single_correct),
            "non_all_correct": get_metrics(ok, non_all_correct),
            "disagreement_single_correct": get_metrics(
                ok, disagreement & single_correct
            ),
        }

        results[name] = metrics

        print(f"\n{name}")
        for label, data in metrics.items():
            show_metrics(label, data)

    print("\n=== 3. Accuracy by vote pattern ===")

    pattern_results = defaultdict(dict)

    unique_patterns = sorted(
        set(patterns),
        key=lambda p: (
            -Counter(patterns)[p],
            p,
        ),
    )

    patterns_array = np.asarray(patterns)

    for pattern in unique_patterns:
        mask = patterns_array == pattern
        total = int(mask.sum())

        print(f"\nPattern {pattern} ({total} questions)")

        for name, ok in success.items():
            metrics = get_metrics(ok, mask)
            pattern_results[pattern][name] = metrics
            show_metrics(name, metrics)

    print("\n=== 4. Disagreement-only pairwise comparisons ===")

    pairwise_results = {}

    model_names = list(PREDICTION_FILES)

    for i, name_a in enumerate(model_names):
        for name_b in model_names[i + 1:]:
            a = success[name_a] & disagreement
            b = success[name_b] & disagreement

            a_only = int((a & ~b).sum())
            b_only = int((b & ~a).sum())
            both_correct = int((a & b).sum())
            both_wrong = int(
                (disagreement & ~a & ~b).sum()
            )

            key = f"{name_a} vs {name_b}"
            pairwise_results[key] = {
                "both_correct": both_correct,
                f"{name_a}_only": a_only,
                f"{name_b}_only": b_only,
                "both_wrong": both_wrong,
                "oracle": both_correct + a_only + b_only,
            }

            print(f"\n{key}")
            print(f"  Both correct: {both_correct}")
            print(f"  {name_a} only correct: {a_only}")
            print(f"  {name_b} only correct: {b_only}")
            print(f"  Both wrong: {both_wrong}")
            print(
                "  Pairwise oracle:",
                both_correct + a_only + b_only,
                "/", int(disagreement.sum()),
            )

    print("\n=== 5. Best model by difficult subset ===")

    for subset in (
        "disagreement",
        "single_correct",
        "non_all_correct",
        "disagreement_single_correct",
    ):
        ranked = sorted(
            model_names,
            key=lambda name: (
                results[name][subset]["correct"],
                name,
            ),
            reverse=True,
        )

        print(f"\n{subset}")
        for name in ranked:
            data = results[name][subset]
            show_metrics(name, data)

    summary = {
        "experiment": "E22-A",
        "split": "validation",
        "n_questions": n,
        "disagreement_questions": int(disagreement.sum()),
        "single_correct_questions": int(single_correct.sum()),
        "non_all_correct_questions": int(non_all_correct.sum()),
        "vote_pattern_counts": dict(Counter(patterns)),
        "models": results,
        "by_vote_pattern": dict(pattern_results),
        "disagreement_pairwise": pairwise_results,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {OUTPUT}")


def finalize_research(out):
    """Verify and evaluate a locked research run; never choose using Test."""
    import csv
    import hashlib
    import os
    import subprocess
    import sys
    import time
    import zipfile
    from train_e25 import research_metrics, research_decode
    from submission import predict, decode

    out = Path(out)
    lock = json.loads((out / "final-lock.json").read_text())
    if (out / "official-results.json").exists():
        raise FileExistsError("Official results already exist: do not rerun Test for selection")
    if any(os.environ.get(k) != "4" for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")):
        raise ValueError("Set all four thread environment variables to 4")
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    # Verify full-Train baseline exactly reproduces the historical learned asset.
    with np.load(out / "final.npz", allow_pickle=False) as bundle:
        arrays = {k: bundle[k].copy() for k in bundle.files}
    with np.load("runs/e14b-weighted-mlp.npz", allow_pickle=False) as old:
        old_error = max(float(np.max(np.abs(arrays[f"m0_{k}"]-old[k])))
                        for k in ("w1","b1","w2","b2")) if not lock["accepted"] else None
        if not lock["accepted"] and (old_error > 1e-6 or
                not np.allclose(arrays["mean"], old["mean"]) or not np.allclose(arrays["std"], old["std"])):
            raise AssertionError("Historical E14 baseline differs")
    train_bank, train_arrays = load_inputs("train")
    subset = [1,7,22,51,98,333,701]
    b = [train_bank[i] for i in subset]
    a = {"embeddings": train_arrays["embeddings"][subset]}
    c = predict(b, a, out / "final.npz")
    assert len(c) == len(b) and all(type(j) is int and 0 <= j < 4 for j in c)
    for i in range(len(b)):
        assert predict([b[i]], {"embeddings": a["embeddings"][i:i+1]}, out / "final.npz") == [c[i]]
    order = [6,0,4,2,1,5,3]
    assert predict([b[i] for i in order], {"embeddings": a["embeddings"][order]}, out / "final.npz") == [c[i] for i in order]
    # Parity for every decoder on synthetic scores including missing/equal answers.
    random = np.random.default_rng(20261029).normal(size=(len(b),4)).astype(np.float32)
    answer_rows = [[parse_answer(v["text"]) for v in r["candidates"]] for r in b]
    for decoder in ("raw", "gate", "aggregate"):
        assert decode(random,b,decoder) == research_decode(random,answer_rows,decoder).tolist()
    uncompressed = sum(arr.nbytes for arr in arrays.values())
    npy_uncompressed = sum(v.file_size for v in zipfile.ZipFile(out / "final.npz").infolist())
    assert npy_uncompressed < 16*2**20
    checks = {"historical_baseline_max_parameter_error": old_error,
              "subset_batch_and_reordering": "passed", "decoder_parity": "passed",
              "uncompressed_array_bytes": uncompressed, "uncompressed_npy_bytes": npy_uncompressed,
              "checkpoint_file_bytes": (out / "final.npz").stat().st_size,
              "hardware": "Apple M5 MacBook Air, 10 CPU cores, 24 GB; CPU only, four threads"}
    (out / "verification.json").write_text(json.dumps(checks,indent=2))
    source = {str(p): digest(p) for p in [Path("submission.py"),Path("train_e25.py"),Path("analyze_e22.py"),
              Path("train_e14.py"),Path("compare_final_models.py"),Path("common.py"),Path("predict.py"),Path("score_predictions.py")]}
    frozen = {"configuration": lock, "sha256_sources": source,
              "sha256_checkpoint": digest(out / "final.npz"), "status": "frozen before official evaluation"}
    with (out / "freeze.json").open("x") as f: json.dump(frozen,f,indent=2)
    rows=[]
    def add_row(exp, method, split, m, prediction=None, scoring=None, training=None):
        rows.append([exp, method, split, m["correct"],m["total"],m["accuracy_percent"],
                     prediction if prediction is not None else "N/A",
                     scoring if scoring is not None else "N/A",training if training is not None else "N/A"])
    cv = json.loads((out / "cv-results.json").read_text())
    selected_cv = json.loads((out / "cv-lock.json").read_text())["selected_cv"]["name"]
    baseline_name = lock["config"]["name"] if not lock["accepted"] else next(r["name"] for r in cv if r["name"].startswith("E26") and r["decoder"]=="gate")
    for r in cv:
        for subset_name,m in r["metrics"].items():
            if subset_name != "overall" and r["name"] not in (selected_cv,baseline_name): continue
            label = "E26" if r["name"].startswith("E26") else "E27"
            add_row(label,r["name"],"Train CV3804 / "+subset_name,m,
                    r["prediction_seconds"] if subset_name=="overall" else None,
                    r["scoring_seconds"] if subset_name=="overall" else None,
                    r["training_seconds"] if subset_name=="overall" else None)
    confirmation=json.loads((out / "confirmation-results.json").read_text())
    for name,r in confirmation["results"].items():
        for subset_name,m in r["metrics"].items():
            add_row("E28",name,"Train confirmation951 / "+subset_name,m,
                    None, r.get("scoring_seconds") if subset_name=="overall" else None,
                    r.get("training_seconds") if subset_name=="overall" else None)
    results={}
    # Validation is reported after the frozen decision. Test evaluated only once.
    for split_name in ("validation", "test"):
        prediction_path=out / f"{split_name}-predictions.json"
        score_path=out / f"{split_name}-score.json"
        command=[sys.executable,"predict.py","--method","submission.py","--checkpoint",str(out/"final.npz"),
                 "--split",split_name,"--output",str(prediction_path)]
        tick=time.perf_counter()
        pred_process=subprocess.run(command,check=True,capture_output=True,text=True)
        pred_wall=time.perf_counter()-tick
        command=[sys.executable,"score_predictions.py","--split",split_name,"--predictions",str(prediction_path),
                 "--output",str(score_path)]
        tick=time.perf_counter()
        score_process=subprocess.run(command,check=True,capture_output=True,text=True)
        score_wall=time.perf_counter()-tick
        score=json.loads(score_path.read_text())
        predictions=json.loads(prediction_path.read_text())
        (out/f"{split_name}-process.log").write_text(pred_process.stdout+score_process.stdout)
        results[split_name]={"official_score":score,"prediction_process_wall_seconds":pred_wall,
                             "scoring_process_wall_seconds":score_wall,"total_process_wall_seconds":pred_wall+score_wall}
        if pred_wall+score_wall >= 30: raise AssertionError("Local time limit exceeded")
        m={"correct":score["correct"],"total":score["n"],"accuracy_percent":score["accuracy"]}
        add_row("E28-Frozen", "E16-A fallback" if not lock["accepted"] else "CV ensemble",split_name.title(),m,
                predictions["prediction_seconds"],score["scoring_seconds"],
                json.loads((out/"training-costs.json").read_text())["refit_seconds"]["final"])
        print(split_name, json.dumps(results[split_name]),flush=True)
    # Validation-only hard-slice report; never use it to reopen selection.
    val_bank,_=load_inputs("validation")
    refs=[json.loads(line) for line in (DATA/"gsm8k_validation.jsonl").read_text().splitlines() if line.strip()]
    assert [r['id'] for r in refs] == [r['id'] for r in val_bank]
    labels=np.array([[parse_answer(c["text"])==parse_answer(ref["answer"]) for c in b["candidates"]]
                     for b,ref in zip(val_bank,refs)],dtype=np.float32)
    flags=np.array([len({a for a in [parse_answer(c["text"]) for c in b["candidates"]] if a is not None})>=2 for b in val_bank])
    choices=np.array(json.loads((out/"validation-predictions.json").read_text())["selected"])
    hard=research_metrics(labels,choices,flags)
    (out/"validation-hard-metrics.json").write_text(json.dumps(hard,indent=2))
    for k,m in hard.items():
        if k!='overall': add_row("E28-Frozen","Fixed E16-A fallback" if not lock["accepted"] else "Fixed CV ensemble", "Validation / "+k,m)
    (out/"official-results.json").write_text(json.dumps(results,indent=2))
    (out/"excel-rows.json").write_text(json.dumps(rows,indent=2))
    with (out/"experiments.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["Experiment","Method","Split","Correct","Total","Accuracy(%)","Prediction(s)","Scoring(s)","Training(s)"])
        writer.writerows(rows)
    assert {str(p):digest(p) for p in map(Path,source)} == source
    assert digest(out/"final.npz")==frozen["sha256_checkpoint"]




def select_expanded(out, group_series=False, kernel_series=False, context_series=False, process_series=False, stability_series=False, repair_series=False):
    """Official Validation selection from a Train-OOF locked shortlist."""
    import csv
    import hashlib
    import os
    import shutil
    import subprocess
    import sys
    import time
    from train_e25 import research_metrics, detect_disagreement
    from submission import predict, reasoning_features
    out=Path(out)
    if (out/'official-test-score.json').exists():raise FileExistsError('Final Test already evaluated')
    if not (out/'done-training.json').exists():raise ValueError('Training is not complete')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        if os.environ.get(key)!='4':raise ValueError('Set all thread environment variables to 4')
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    shortlist=json.loads((out/'shortlist-lock.json').read_text())['recipes']
    bank,arrays=load_inputs('validation')
    with (DATA/'gsm8k_validation.jsonl').open(encoding='utf-8') as stream:
        refs=[json.loads(line) for line in stream if line.strip()]
    assert [r['id'] for r in refs]==[r['id'] for r in bank]
    y=np.asarray([[parse_answer(c['text'])==parse_answer(r['answer']) for c in b['candidates']] for b,r in zip(bank,refs)],dtype=np.float32)
    flags=detect_disagreement(bank)
    validation=[]
    def official_eval(folder,split,tag):
        pred=out/(tag+'-predictions.json');score=out/(tag+'-score.json')
        cmd=[sys.executable,'predict.py','--method','submission.py','--checkpoint',str(folder),'--split',split,'--output',str(pred)]
        tick=time.perf_counter();p=subprocess.run(cmd,check=True,capture_output=True,text=True);pred_wall=time.perf_counter()-tick
        cmd=[sys.executable,'score_predictions.py','--split',split,'--predictions',str(pred),'--output',str(score)]
        tick=time.perf_counter();q=subprocess.run(cmd,check=True,capture_output=True,text=True);score_wall=time.perf_counter()-tick
        (out/(tag+'-process.log')).write_text(p.stdout+q.stdout)
        return json.loads(pred.read_text()),json.loads(score.read_text()),dict(prediction_process_wall_seconds=pred_wall,scoring_process_wall_seconds=score_wall,total_process_wall_seconds=pred_wall+score_wall)
    for j,recipe in enumerate(shortlist):
        folder=out/f'candidate{j}'
        asset_bytes=sum(p.stat().st_size for p in folder.iterdir())
        pred,score,timing=official_eval(folder,'validation',f'candidate{j}-validation')
        metrics=research_metrics(y,np.array(pred['selected']),flags)
        record=dict(candidate=j,recipe=recipe['name'],official_score=score,metrics=metrics,timing=timing,asset_bytes=asset_bytes)
        record['eligible']=asset_bytes<16*2**20 and timing['total_process_wall_seconds']<30
        validation.append(record)
        print('VALIDATION',json.dumps(record),flush=True)
    (out/'validation-shortlist-results.json').write_text(json.dumps(validation,indent=2))
    eligible=[r for r in validation if r['eligible']]
    best=max(eligible,key=lambda r:(r['metrics']['disagreement']['correct'],r['metrics']['overall']['correct'],r['metrics']['single_correct']['correct'],int(group_series and r['recipe'] in ('E32-parent-fixed','frozen-parent-fixed')),-r['asset_bytes']))
    selected_dir=out/f'candidate{best["candidate"]}'
    shutil.copytree(selected_dir,out/'final')
    # Verify arbitrary bank sizes, question order and candidate order without using Test.
    small=[2,5,11,22,35]
    b=[bank[i] for i in small];a={'embeddings':arrays['embeddings'][small]}
    choices=predict(b,a,out/'final')
    assert all(type(c) is int and 0<=c<4 for c in choices)
    for i in range(len(b)):
        assert predict([b[i]],{'embeddings':a['embeddings'][i:i+1]},out/'final')==[choices[i]]
    perm=[4,2,0,3,1]
    assert predict([b[i] for i in perm],{'embeddings':a['embeddings'][perm]},out/'final')==[choices[i] for i in perm]
    # Group feature invariance under candidate reordering where scores are untied.
    perm_c=[2,0,3,1]
    reordered=[dict(row,candidates=[dict(row['candidates'][i],index=j) for j,i in enumerate(perm_c)]) for row in b]
    reordered_choices=predict(reordered,{'embeddings':a['embeddings'][:,perm_c]},out/'final')
    for i,(old,new) in enumerate(zip(choices,reordered_choices)):
        assert (perm_c[new]==old or parse_answer(b[i]['candidates'][perm_c[new]]['text'])==parse_answer(b[i]['candidates'][old]['text']))
    sources={str(p):digest(p) for p in [Path('submission.py'),Path('train_scorer.py'),Path('train_e25.py'),Path('train_e14.py'),Path('compare_final_models.py'),Path('analyze_e22.py'),Path('requirements.txt'),Path('common.py'),Path('predict.py'),Path('score_predictions.py')]}
    assets={p.name:digest(p) for p in (out/'final').iterdir()}
    freeze={'selected_by':'Train OOF shortlist then official Validation disagreement, not Test',
            'selection':best,'source_sha256':sources,'asset_sha256':assets,'status':'frozen before Test',
            'verification':{'singleton_batch_question_order_candidate_order':'passed'}}
    (out/'freeze.json').write_text(json.dumps(freeze,indent=2))
    print('FINAL LOCKED',best['candidate'],best['recipe'],flush=True)
    # No more model/configuration selection after this point.
    reused_test=False
    if group_series and best['recipe'] in ('E32-parent-fixed','frozen-parent-fixed'):
        parent=Path(json.loads((out/'shortlist-lock.json').read_text())['parent_run'])
        parent_result=json.loads((parent/'final-results.json').read_text())
        idx=parent_result['validation']['candidate']
        parent_validation=json.loads((parent/f'candidate{idx}-validation-predictions.json').read_text())
        this_validation=json.loads((out/f'candidate{best["candidate"]}-validation-predictions.json').read_text())
        assert parent_validation['selected']==this_validation['selected']
        for p in (parent/'final').iterdir():
            if p.name!='config.json':assert digest(out/'final'/p.name)==digest(p)
        prediction=json.loads((parent/'official-test-predictions.json').read_text())
        test_score=parent_result['test']['official_score']
        test_timing=parent_result['test']['timing']
        for suffix in ('predictions','score'):
            shutil.copy2(parent/f'official-test-{suffix}.json',out/f'official-test-{suffix}.json')
        reused_test=True
    else:
        prediction,test_score,test_timing=official_eval(out/'final','test','official-test')
    final_result=dict(validation=best,test=dict(official_score=test_score,timing=test_timing,reused_frozen_parent_result=reused_test),freeze=freeze)
    (out/'final-results.json').write_text(json.dumps(final_result,indent=2))
    print('FROZEN TEST',json.dumps(final_result['test']),flush=True)
    assert sources=={p:digest(p) for p in sources}
    assert assets=={p.name:digest(p) for p in (out/'final').iterdir()}
    cv=json.loads((out/'cv-results.json').read_text());rows=[]
    def add(exp,name,split,metrics,pred=None,score=None,training=None):
        rows.append([exp,name,split,metrics['correct'],metrics['total'],metrics['accuracy_percent'],pred if pred is not None else 'N/A',score if score is not None else 'N/A',training if training is not None else 'N/A'])
    for r in cv:
        if group_series:
            family=('E43' if r['group_weight']==1 else 'E44') if repair_series else ('E41' if r['group_weight']==1 else 'E42') if stability_series else ('E39' if r['group_weight']==1 else 'E40') if process_series else ('E37' if r['group_weight']==1 else 'E38') if context_series else ('E35' if r['group_weight']==1 else 'E36') if kernel_series else ('E33' if r['group_weight']==1 else 'E34')
        else:
            names=[n for n,w in r['members']]
            family='E32' if len(names)>1 else 'E31' if names[0].startswith('text') else 'E30' if names[0].startswith('relative') else 'E29'
        add(family,r['name'],'Train OOF / overall',r['metrics']['overall'],r.get('prediction_seconds'),r.get('scoring_seconds'))
        add(family,r['name'],'Train OOF / disagreement',r['metrics']['disagreement'])
    costs=json.loads((out/'training-costs.json').read_text())
    for r in validation:
        idx=r['candidate'];p=json.loads((out/f'candidate{idx}-validation-predictions.json').read_text())
        for subset,m in r['metrics'].items():
            add('E44-Validation' if repair_series else 'E42-Validation' if stability_series else 'E40-Validation' if process_series else 'E38-Validation' if context_series else 'E36-Validation' if kernel_series else 'E34-Validation' if group_series else 'E32-Validation',r['recipe'],'Validation / '+subset,m,p['prediction_seconds'] if subset=='overall' else None,r['official_score']['scoring_seconds'] if subset=='overall' else None)
    m=dict(correct=test_score['correct'],total=test_score['n'],accuracy_percent=test_score['accuracy'])
    add('E44-Frozen' if repair_series else 'E42-Frozen' if stability_series else 'E40-Frozen' if process_series else 'E38-Frozen' if context_series else 'E36-Frozen' if kernel_series else 'E34-Frozen' if group_series else 'E32-Frozen',best['recipe'],'Test (inherited, not rerun)' if reused_test else 'Test',m,None if reused_test else prediction['prediction_seconds'],None if reused_test else test_score['scoring_seconds'])
    # Search costs are shared. The one explicit cost record prevents double counting.
    rows.append(['E43-E44 cost' if repair_series else 'E41-E42 cost' if stability_series else 'E39-E40 cost' if process_series else 'E37-E38 cost' if context_series else 'E35-E36 cost' if kernel_series else 'E33-E34 cost' if group_series else 'E29-E32 cost','All unique CV and full-Train fits','Training pipeline','N/A','N/A','N/A','N/A','N/A',costs['total_seconds']])
    (out/'excel-rows.json').write_text(json.dumps(rows,indent=2))
    with (out/'experiments.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['Experiment','Method','Split','Correct','Total','Accuracy(%)','Prediction(s)','Scoring(s)','Training(s)']);writer.writerows(rows)

def diagnose_process(out):
    """Train-only paired analysis for the E39/E40 predeclared experiment."""
    import time
    from submission import numeric_process_trace
    from train_e25 import detect_disagreement,research_metrics
    from catboost import CatBoost
    out=Path(out);start=time.perf_counter();bank,arrays=load_inputs('train')
    with (DATA/'gsm8k_train.jsonl').open(encoding='utf-8') as f:refs=[json.loads(s) for s in f if s.strip()]
    assert [r['id'] for r in refs]==[r['id'] for r in bank]
    y=np.array([[parse_answer(c['text'])==parse_answer(r['answer']) for c in b['candidates']] for b,r in zip(bank,refs)])
    hard=detect_disagreement(bank);valid=np.array([[parse_answer(c['text']) is not None for c in b['candidates']] for b in bank])
    cache=np.load(out/'oof-final.npz',allow_pickle=False);base=cache['baseline']
    old_choice=np.where(valid,base,-1e6).argmax(1);old=y[np.arange(len(y)),old_choice]
    folds=[np.array(v) for v in json.loads((out/'folds.json').read_text())]
    short=json.loads((out/'shortlist-lock.json').read_text())['recipes'];comparisons=[]
    for r in short:
        if r['group_model'] is None:s=base
        else:
            raw=cache[r['group_model']];s=r['group_weight']*(raw-raw.mean(1,keepdims=True))/r['scale']+(1-r['group_weight'])*base
        choice=np.where(valid,s,-1e6).argmax(1);ok=y[np.arange(len(y)),choice]
        paired={}
        for name,mask in [('overall',np.ones(len(y),dtype=bool)),('disagreement',hard),('single_correct',y.sum(1)==1)]:
            d=ok[mask].astype(int)-old[mask].astype(int);rng=np.random.default_rng(20261027)
            bootstrap=np.array([rng.choice(d,len(d),replace=True).mean()*100 for _ in range(2000)])
            paired[name]=dict(gained=int((d==1).sum()),lost=int((d==-1).sum()),delta_correct=int(d.sum()),
                delta_percentage_points=float(d.mean()*100),paired_bootstrap_95_percentile=np.percentile(bootstrap,[2.5,97.5]).tolist())
        comparisons.append(dict(name=r['name'],metrics=research_metrics(y,choice,hard),paired=paired,
            fold_hard_delta=[int((ok[v]&hard[v]).sum()-(old[v]&hard[v]).sum()) for v in folds]))
    coverage=[]
    for good in (False,True):
        count=has=incorrect=total_equations=0
        for i,row in enumerate(bank):
            for j,c in enumerate(row['candidates']):
                if bool(y[i,j])!=good:continue
                eq,_=numeric_process_trace(c['text']);count+=1;has+=bool(eq)
                incorrect+=any(not e['valid'] for e in eq);total_equations+=len(eq)
        coverage.append(dict(correct_final_answer=good,candidates=count,with_equations=has,with_inconsistent_equations=incorrect,equations=total_equations))
    importances=[]
    for p in sorted(out.glob('fold*-process*.cbm')):
        m=CatBoost();m.load_model(str(p));imp=np.asarray(m.get_feature_importance(thread_count=4));offset=116 if 'process-meta' in p.name else 1140
        importances.append(dict(model=p.name,process_importance_percent=float(imp[offset:].sum()),top_process_indices=np.argsort(imp[offset:])[-10:][::-1].tolist()))
    control_file=Path('runs/expanded-20261027-215926/oof-final.npz')
    if control_file.exists() and "control-800" in cache and 'repair' not in out.name:
        with np.load(control_file,allow_pickle=False) as prev:parity=float(np.abs(cache['control-800']-prev['full-d5-800']).max())
        assert parity<1e-5,parity
    else:parity=None
    result=dict(comparisons=comparisons,process_parser_coverage=coverage,importance=importances,
        old_control_max_score_difference=parity,diagnostic_seconds=time.perf_counter()-start,
        caveat='Repeated development folds; bootstrap is descriptive and does not adjust for model selection.')
    (out/'train-diagnostics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='importance'},indent=2))


def diagnose_repair(out):
    """Paired Train analysis and arithmetic-repair coverage; no Test access."""
    from fractions import Fraction
    from submission import arithmetic_repair_traces
    import time
    out=Path(out);diagnose_process(out);start=time.perf_counter();bank,_=load_inputs('train')
    with (DATA/'gsm8k_train.jsonl').open() as f:refs=[json.loads(s) for s in f if s.strip()]
    assert [r['id'] for r in bank]==[r['id'] for r in refs]
    modes=[dict(mode=name,anchored=0,changed=0,changed_to_gold_tolerance=0,changed_away_from_gold_tolerance=0,
                changed_to_supplied_answer=0,propagation_equations=0) for name in ('local','protected','permissive')]
    close=lambda a,b: a is not None and b is not None and abs(a-b)<=1e-5*max(1.,abs(a),abs(b))
    for row,ref in zip(bank,refs):
        traces=arithmetic_repair_traces(row);gold=float(Fraction(parse_answer(ref['answer'])))
        for t in traces:
            for i,m in enumerate(t['modes']):
                record=modes[i];record['anchored']+=m['anchored'];record['propagation_equations']+=m['propagated']
                if m['proposal'] is None or close(m['proposal'],t['answer']):continue
                record['changed']+=1
                record['changed_to_gold_tolerance']+=int(close(m['proposal'],gold) and not close(t['answer'],gold))
                record['changed_away_from_gold_tolerance']+=int(not close(m['proposal'],gold) and close(t['answer'],gold))
                record['changed_to_supplied_answer']+=int(any(close(m['proposal'],q['answer']) for q in traces))
    result=dict(modes=modes,seconds=time.perf_counter()-start,
                caveat='Tolerant numeric diagnostic of intermediate proposals, not official candidate selection accuracy.')
    (out/'repair-coverage.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__ == "__main__":
    import sys
    if "--diagnose-repair" in sys.argv:
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument("--diagnose-repair",type=Path,required=True)
        diagnose_repair(parser.parse_args().diagnose_repair)
    elif "--select-repair" in sys.argv:
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument("--select-repair",type=Path,required=True)
        select_expanded(parser.parse_args().select_repair,group_series=True,repair_series=True)
    elif "--select-stability" in sys.argv:
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument("--select-stability",type=Path,required=True)
        select_expanded(parser.parse_args().select_stability,group_series=True,stability_series=True)
    elif "--diagnose-process" in sys.argv:
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument("--diagnose-process",type=Path,required=True)
        diagnose_process(parser.parse_args().diagnose_process)
    elif "--select-process" in sys.argv:
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument("--select-process",type=Path,required=True)
        select_expanded(parser.parse_args().select_process,group_series=True,process_series=True)
    elif "--select-context" in sys.argv:
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument("--select-context",type=Path,required=True)
        select_expanded(parser.parse_args().select_context,group_series=True,context_series=True)
    elif "--select-kernel" in sys.argv:
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument("--select-kernel",type=Path,required=True)
        select_expanded(parser.parse_args().select_kernel,group_series=True,kernel_series=True)
    elif "--select-groups" in sys.argv:
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument("--select-groups",type=Path,required=True)
        select_expanded(parser.parse_args().select_groups,group_series=True)
    elif "--select-expanded" in sys.argv:
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument("--select-expanded",type=Path,required=True)
        select_expanded(parser.parse_args().select_expanded)
    elif "--finalize-run" in sys.argv:
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument("--finalize-run",type=Path,required=True)
        finalize_research(parser.parse_args().finalize_run)
    else:
        main()
