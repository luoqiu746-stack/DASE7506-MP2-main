
"""E10: Bradley-Terry Pairwise Ranking MLP.

Train a 1024 -> 32 -> 1 scorer with pairwise ranking loss.
Uses only official TRAIN references and frozen embeddings.

NumPy implementation, CPU compatible.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer


INPUT_DIM = 1024
CANDIDATES = 4


def initialize(seed, hidden):
    """Initialize MLP parameters."""
    rng = np.random.default_rng(seed)

    return {
        "w1": (
            rng.standard_normal((INPUT_DIM, hidden))
            * np.sqrt(2.0 / INPUT_DIM)
        ).astype(np.float32),

        "b1": np.zeros(hidden, dtype=np.float32),

        "w2": (
            rng.standard_normal((hidden, 1))
            * np.sqrt(1.0 / hidden)
        ).astype(np.float32),

        "b2": np.zeros(1, dtype=np.float32),
    }


def forward(x, params):
    """Forward pass.

    x: (batch_questions, 4, 1024)

    Returns:
        hidden: (batch_questions * 4, hidden_dim)
        scores: (batch_questions, 4)
    """
    batch_questions = x.shape[0]

    flat_x = x.reshape(-1, INPUT_DIM)

    hidden = np.maximum(
        flat_x @ params["w1"] + params["b1"],
        0.0
    )

    flat_scores = (
        hidden @ params["w2"] + params["b2"]
    )

    scores = flat_scores.reshape(batch_questions, CANDIDATES)

    return hidden, scores


def pair_loss_and_grad(scores, labels):
    """Compute Bradley-Terry loss and gradients.

    Compare every correct candidate against every incorrect
    candidate within the same question.

    scores: (batch_questions, 4)
    labels: (batch_questions, 4)

    Returns:
        loss: average pairwise loss
        ds: derivative with respect to scores
        n_pairs: number of valid positive-negative pairs
    """
    positive = labels > 0.5
    negative = ~positive

    # mask[b, i, j] is True when:
    # candidate i is correct and candidate j is incorrect.
    mask = (
        positive[:, :, None]
        & negative[:, None, :]
    )

    n_pairs = int(mask.sum())

    if n_pairs == 0:
        return (
            0.0,
            np.zeros_like(scores, dtype=np.float32),
            0
        )

    # diff[b, i, j] = score_i - score_j
    diff = (
        scores[:, :, None]
        - scores[:, None, :]
    )

    active_diff = diff[mask]

    # Softplus(-diff), numerically stable.
    loss = float(
        np.logaddexp(0.0, -active_diff).mean()
    )

    # d loss / d diff = -sigmoid(-diff)
    sigmoid_negative = (
        1.0 / (1.0 + np.exp(np.clip(diff, -30, 30)))
    )

    contributions = np.where(
        mask,
        sigmoid_negative,
        0.0
    )

    # Positive candidate gradients: negative (raise score).
    # Negative candidate gradients: positive (lower score).
    ds = (
        -contributions.sum(axis=2)
        + contributions.sum(axis=1)
    ) / n_pairs

    return loss, ds.astype(np.float32), n_pairs


def evaluate_loss(x, y, params):
    """Evaluate average pairwise loss without updating weights."""
    _, scores = forward(x, params)

    loss, _, n_pairs = pair_loss_and_grad(scores, y)

    if n_pairs == 0:
        raise ValueError(
            "No positive-negative pairs in evaluation data"
        )

    return loss


def train(
    x,
    y,
    params,
    epochs,
    batch_size,
    lr,
    weight_decay,
    seed,
    x_dev=None,
    y_dev=None,
):
    """Train MLP with Adam and pairwise ranking loss."""
    rng = np.random.default_rng(seed)

    adam_m = {
        key: np.zeros_like(value)
        for key, value in params.items()
    }

    adam_v = {
        key: np.zeros_like(value)
        for key, value in params.items()
    }

    step = 0

    best_epoch = 1
    best_dev_loss = float("inf")

    hidden_dim = params["w1"].shape[1]

    for epoch in range(1, epochs + 1):

        # Shuffle questions, not individual candidates.
        order = rng.permutation(len(y))

        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]

            xb = x[indices]
            yb = y[indices]

            # Forward:
            # hidden: (batch_questions * 4, hidden_dim)
            # scores: (batch_questions, 4)
            hidden, scores = forward(xb, params)

            _, ds, n_pairs = pair_loss_and_grad(
                scores, yb
            )

            if n_pairs == 0:
                continue

            # FIX: reshape the score gradients into candidate rows.
            ds = ds.reshape(-1, 1)

            flat_x = xb.reshape(-1, INPUT_DIM)

            # Shapes:
            # hidden.T: (hidden_dim, batch_questions * 4)
            # ds:       (batch_questions * 4, 1)
            # w2 grad:  (hidden_dim, 1)
            grads = {
                "w2": (
                    hidden.T @ ds
                    + weight_decay * params["w2"]
                ),
                "b2": ds.sum(axis=0),
            }

            # Backpropagate through ReLU.
            dh = (
                (ds @ params["w2"].T)
                * (hidden > 0)
            )

            grads["w1"] = (
                flat_x.T @ dh
                + weight_decay * params["w1"]
            )

            grads["b1"] = dh.sum(axis=0)

            # Adam update.
            step += 1

            for key in params:
                grad = grads[key]

                adam_m[key] = (
                    0.9 * adam_m[key]
                    + 0.1 * grad
                )

                adam_v[key] = (
                    0.999 * adam_v[key]
                    + 0.001 * (grad * grad)
                )

                m_hat = (
                    adam_m[key]
                    / (1.0 - 0.9 ** step)
                )

                v_hat = (
                    adam_v[key]
                    / (1.0 - 0.999 ** step)
                )

                params[key] -= (
                    lr * m_hat
                    / (np.sqrt(v_hat) + 1e-8)
                )

        # Evaluate on internal development set.
        if x_dev is not None:
            train_loss = evaluate_loss(x, y, params)
            dev_loss = evaluate_loss(
                x_dev, y_dev, params
            )

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_epoch = epoch

            print(
                f"Epoch {epoch:2d} | "
                f"train pair loss {train_loss:.5f} | "
                f"dev pair loss {dev_loss:.5f}",
                flush=True
            )

        else:
            print(
                f"Final training epoch {epoch}/{epochs}",
                flush=True
            )

    return best_epoch


def main():
    timer = time.perf_counter()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/e10-pairwise.npz")
    )

    parser.add_argument(
        "--epochs", type=int, default=30
    )

    parser.add_argument(
        "--hidden", type=int, default=32
    )

    parser.add_argument(
        "--batch-size", type=int, default=64
    )

    parser.add_argument(
        "--lr", type=float, default=0.001
    )

    parser.add_argument(
        "--weight-decay", type=float, default=0.01
    )

    parser.add_argument(
        "--seed", type=int, default=17
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # 1. Load official training data.
    # --------------------------------------------------
    bank, arrays = load_inputs("train")

    embeddings = arrays["embeddings"].astype(
        np.float32
    )

    if embeddings.shape != (
        len(bank), CANDIDATES, INPUT_DIM
    ):
        raise ValueError(
            f"Unexpected embeddings shape: "
            f"{embeddings.shape}"
        )

    with open(
        DATA / "gsm8k_train.jsonl",
        encoding="utf-8"
    ) as file:
        references = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if len(bank) != len(references):
        raise ValueError(
            "Training bank and reference count mismatch"
        )

    # --------------------------------------------------
    # 2. Construct correctness labels using TRAIN only.
    # --------------------------------------------------
    labels = []

    for row, ref in zip(bank, references):

        if (
            row["id"] != ref["id"]
            or row["question"] != ref["question"]
        ):
            raise ValueError(
                "Training reference alignment mismatch"
            )

        gold_answer = parse_answer(ref["answer"])

        if gold_answer is None:
            raise ValueError(
                f"Unparseable gold answer: {row['id']}"
            )

        row_labels = [
            float(
                parse_answer(candidate["text"])
                == gold_answer
            )
            for candidate in row["candidates"]
        ]

        labels.append(row_labels)

    labels = np.asarray(
        labels, dtype=np.float32
    )

    if labels.shape != (
        len(bank), CANDIDATES
    ):
        raise ValueError("Incorrect labels shape")

    # --------------------------------------------------
    # 3. Internal 90/10 split by question.
    # --------------------------------------------------
    rng = np.random.default_rng(args.seed)

    order = rng.permutation(len(bank))

    n_dev = max(
        1, int(len(bank) * 0.1)
    )

    dev_indices = order[:n_dev]
    train_indices = order[n_dev:]

    x_train = embeddings[train_indices]
    y_train = labels[train_indices]

    x_dev = embeddings[dev_indices]
    y_dev = labels[dev_indices]

    # Normalization statistics based only on
    # the internal training portion.
    flat_train = x_train.reshape(-1, INPUT_DIM)

    mean = flat_train.mean(axis=0)

    std = np.maximum(
        flat_train.std(axis=0), 1e-6
    )

    x_train = (x_train - mean) / std
    x_dev = (x_dev - mean) / std

    print(f"Train questions: {len(train_indices)}")
    print(f"Internal dev questions: {len(dev_indices)}")
    print(f"Hidden dimension: {args.hidden}")
    print(
        f"Batch size (questions): {args.batch_size}"
    )

    # --------------------------------------------------
    # 4. Phase 1: find best epoch using internal dev.
    # --------------------------------------------------
    print(
        "\nPhase 1: choose epoch using internal dev"
    )

    params = initialize(
        args.seed, args.hidden
    )

    best_epoch = train(
        x_train,
        y_train,
        params,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.seed,
        x_dev,
        y_dev
    )

    print(f"\nBest epoch: {best_epoch}")

    # --------------------------------------------------
    # 5. Phase 2: retrain from scratch on full TRAIN.
    # --------------------------------------------------
    full_x = embeddings
    full_y = labels

    flat_full = full_x.reshape(-1, INPUT_DIM)

    full_mean = flat_full.mean(axis=0)

    full_std = np.maximum(
        flat_full.std(axis=0), 1e-6
    )

    full_x = (full_x - full_mean) / full_std

    print("\nPhase 2: retrain on full TRAIN")

    final_params = initialize(
        args.seed, args.hidden
    )

    train(
        full_x,
        full_y,
        final_params,
        best_epoch,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.seed
    )

    # --------------------------------------------------
    # 6. Save inference checkpoint.
    # --------------------------------------------------
    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )

    np.savez(
        args.output,
        **final_params,
        mean=full_mean.astype(np.float32),
        std=full_std.astype(np.float32)
    )

    elapsed = time.perf_counter() - timer

    print(f"\nCheckpoint: {args.output}")
    print(
        f"Checkpoint size: "
        f"{args.output.stat().st_size:,} bytes"
    )
    print(
        f"Training pipeline seconds: {elapsed:.3f}"
    )




def expanded_main():
    """E29-E32: new input evidence and nonlinear learners; Train OOF selection."""
    import hashlib
    import shutil
    from catboost import CatBoostClassifier, CatBoostRanker, Pool
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.preprocessing import normalize
    from sklearn.linear_model import Ridge
    from train_e14 import question_weights
    from train_e25 import research_fit, research_normalize, research_metrics, detect_disagreement
    from submission import reasoning_features, text_documents, decode
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--expanded',action='store_true',required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir
    with (out/'started.json').open('x') as f:json.dump({'seed':7306,'phase':'expanded training'},f)
    start=time.perf_counter()
    bank,arr=load_inputs('train');x=arr['embeddings']
    with (DATA/'gsm8k_train.jsonl').open(encoding='utf-8') as stream:
        refs=[json.loads(line) for line in stream if line.strip()]
    assert [r['id'] for r in bank]==[r['id'] for r in refs]
    y=np.asarray([[float(parse_answer(c['text'])==parse_answer(r['answer'])) for c in row['candidates']] for row,r in zip(bank,refs)],dtype=np.float32)
    counts=y.sum(1).astype(int);weights=question_weights(counts,'weighted')
    disagree=detect_disagreement(bank)
    rng=np.random.default_rng(7306);folds=[[] for _ in range(3)]
    for c in range(1,5):
        for d in (False,True):
            idx=rng.permutation(np.flatnonzero((counts==c)&(disagree==d)))
            for j,ids in enumerate(np.array_split(idx,3)):folds[j].extend(ids.tolist())
    folds=[np.sort(ids) for ids in folds]
    (out/'folds.json').write_text(json.dumps([a.tolist() for a in folds]))
    print('E29-E32 fold sizes',list(map(len,folds)),flush=True)
    features={m:reasoning_features(bank,x,m) for m in ('metadata','full','relative','embedding')}
    np.savez(out/'train-labels.npz',labels=y,disagreement=disagree)
    documents=text_documents(bank)
    vectorizer=HashingVectorizer(n_features=65536,ngram_range=(1,2),alternate_sign=False,norm=None)
    texts=vectorizer.transform(documents);texts.data=np.log1p(texts.data)
    specs=[dict(name='metadata',features='metadata',depth=5,kind='classifier'),
           dict(name='full-d5',features='full',depth=5,kind='classifier'),
           dict(name='full-d7',features='full',depth=7,kind='classifier'),
           dict(name='relative-rank',features='relative',depth=5,kind='ranker'),
           dict(name='embedding-ablation',features='embedding',depth=5,kind='classifier')]
    (out/'model-specs.json').write_text(json.dumps(specs,indent=2))
    oof={};costs=[]
    def store(name,ids,score):
        if name not in oof:oof[name]=np.zeros(y.shape,dtype=np.float32)
        oof[name][ids]=score
    def tree_fit(spec,fitting):
        feat=features[spec['features']]
        params=dict(iterations=800,depth=spec['depth'],learning_rate=0.05,l2_leaf_reg=5.,
                    random_seed=17,thread_count=4,verbose=False,allow_writing_files=False,border_count=64)
        if spec['kind']=='classifier':
            model=CatBoostClassifier(loss_function='Logloss',**params)
            model.fit(feat[fitting].reshape(-1,feat.shape[-1]),y[fitting].reshape(-1),sample_weight=np.repeat(weights[fitting],4))
        else:
            model=CatBoostRanker(loss_function='YetiRank',**params)
            # All-correct groups have zero ranking information.
            fitting=fitting[counts[fitting]<4]
            pool=Pool(feat[fitting].reshape(-1,feat.shape[-1]),label=y[fitting].reshape(-1),group_id=np.repeat(np.arange(len(fitting)),4))
            model.fit(pool)
        return model
    for fold,valid in enumerate(folds):
        fitting=np.setdiff1d(np.arange(len(bank)),valid)
        xt,xv,_,_=research_normalize(x,fitting,valid)
        scores=[]
        for seed in (17,29,43):
            tick=time.perf_counter()
            p,saved,elapsed=research_fit(xt,y[fitting],weights[fitting],seed,.01,16,xv,(10,16))
            costs.append(dict(fold=fold,model=f'mlp{seed}',seconds=time.perf_counter()-tick))
            if seed==17:store('mlp',valid,saved[10]['scores'])
            scores.append(saved[16]['scores'])
        store('mlp3',valid,np.mean(scores,axis=0))
        fitrows=np.concatenate([np.arange(4*i,4*i+4) for i in fitting]);valrows=np.concatenate([np.arange(4*i,4*i+4) for i in valid])
        df=np.asarray((texts[fitrows]>0).sum(0)).ravel()
        idf=np.log((1+len(fitrows))/(1+df))+1
        text_train=normalize(texts[fitrows].multiply(idf));text_valid=normalize(texts[valrows].multiply(idf))
        for alpha in (1.,10.,100.):
            tick=time.perf_counter()
            model=Ridge(alpha=alpha,solver='lsqr',max_iter=250,tol=1e-4)
            model.fit(text_train,y[fitting].reshape(-1),sample_weight=np.repeat(weights[fitting],4))
            store(f'text{alpha:g}',valid,model.predict(text_valid).reshape(-1,4))
            costs.append(dict(fold=fold,model=f'text{alpha:g}',seconds=time.perf_counter()-tick))
            print(f'fold{fold+1} text alpha={alpha} {costs[-1]["seconds"]:.2f}s',flush=True)
        for spec in specs:
            tick=time.perf_counter();model=tree_fit(spec,fitting)
            feat=features[spec['features']][valid].reshape(-1,features[spec['features']].shape[-1])
            for trees in (300,800):
                kwargs={'prediction_type':'RawFormulaVal'} if spec['kind']=='classifier' else {}
                score=model.predict(feat,ntree_end=trees,thread_count=4,**kwargs).reshape(-1,4)
                store(spec['name']+f'-{trees}',valid,score)
            elapsed=time.perf_counter()-tick
            costs.append(dict(fold=fold,model=spec['name'],seconds=elapsed))
            model.save_model(str(out/f'fold{fold}-{spec["name"]}.cbm'))
            np.savez(out/'oof-progress.npz',**oof)
            (out/'training-costs-progress.json').write_text(json.dumps(costs,indent=2))
            print(f'fold{fold+1} {spec["name"]}: {elapsed:.2f}s',flush=True)
    np.savez(out/'oof-final.npz',**oof)
    scales={name:max(float(np.std(s-s.mean(1,keepdims=True))),1e-6) for name,s in oof.items()}
    normalized={name:(s-s.mean(1,keepdims=True))/scales[name] for name,s in oof.items()}
    parse_mask=np.array([[parse_answer(c['text']) is not None for c in row['candidates']] for row in bank])
    configs=[];records=[]
    def record(members,decoder):
        score=sum(normalized[name]*w for name,w in members)
        score=np.where(parse_mask,score,-1e6)
        tick=time.perf_counter();choices=np.array(decode(score,bank,decoder))
        prediction=time.perf_counter()-tick
        tick=time.perf_counter();metrics=research_metrics(y,choices,disagree);scoring=time.perf_counter()-tick
        per_fold=[research_metrics(y[v],choices[v],disagree[v]) for v in folds]
        name='+'.join(f'{n}x{w:g}' for n,w in members)+'/'+decoder
        r=dict(name=name,members=members,decoder=decoder,metrics=metrics,fold_metrics=per_fold,
               prediction_seconds=prediction,scoring_seconds=scoring)
        records.append(r);return r
    for name in oof:
        for decoder in ('raw','gate','aggregate'):record([(name,1.)],decoder)
    # Finite fusion matrix: tree+MLP, then fixed text augmentation. No Test access.
    for tree in [n for n in oof if n not in ('mlp','mlp3') and not n.startswith('text') and 'ablation' not in n]:
        for w in (.25,.5,.75):
            for decoder in ('raw','aggregate'):
                record([(tree,w),('mlp3',1-w)],decoder)
    ranked=sorted(records,key=lambda r:(r['metrics']['disagreement']['correct'],r['metrics']['overall']['correct']),reverse=True)
    top_nontext=ranked[:5]
    for r in top_nontext:
        for text in ('text1','text10','text100'):
            for w in (.15,.3):record([(n,weight*(1-w)) for n,weight in r['members']]+[(text,w)],r['decoder'])
    ranked=sorted(records,key=lambda r:(r['metrics']['disagreement']['correct'],r['metrics']['overall']['correct']),reverse=True)
    # Top five recipes, plus the known E16-A baseline. No simultaneous-slice veto.
    shortlisted=[]
    for r in ranked:
        if len(shortlisted)==5:break
        shortlisted.append(r)
    baseline=record([('mlp',scales['mlp'])],'gate') # scale factor restores raw historical logit scale.
    shortlisted.append(baseline)
    (out/'cv-results.json').write_text(json.dumps(records,indent=2))
    (out/'shortlist-lock.json').write_text(json.dumps({'recipes':shortlisted,'scales':scales,'selection':'Train OOF disagreement; official Validation will select'},indent=2))
    print('CV SHORTLIST',[(r['name'],r['metrics']['disagreement']['correct']) for r in shortlisted],flush=True)
    # Fit every required component once on all supplied TRAIN.
    needed={n for r in shortlisted for n,w in r['members']}
    component_info={}
    full_mean=x.reshape(-1,1024).mean(0);full_std=np.maximum(x.reshape(-1,1024).std(0),1e-6)
    full=(x-full_mean)/full_std
    for name in sorted(needed):
        tick=time.perf_counter()
        if name in ('mlp','mlp3'):
            use_seeds=(17,) if name=='mlp' else (17,29,43)
            members=[]
            for j,seed in enumerate(use_seeds):
                ep=10 if name=='mlp' else 16
                p,_,_=research_fit(full,y,weights,seed,.01,ep)
                file=f'{name}-{j}.npz';np.savez(out/file,**p,mean=full_mean,std=full_std)
                members.append(dict(kind='mlp',file=file,weight=1/len(use_seeds),scale=scales[name]))
            component_info[name]=members
        elif name.startswith('text'):
            df=np.asarray((texts>0).sum(0)).ravel();idf=np.log((1+len(documents))/(1+df))+1
            matrix=normalize(texts.multiply(idf));alpha=float(name[4:])
            m=Ridge(alpha=alpha,solver='lsqr',max_iter=250,tol=1e-4)
            m.fit(matrix,y.reshape(-1),sample_weight=np.repeat(weights,4))
            file=name+'.npz';np.savez(out/file,weights=m.coef_.astype(np.float32),intercept=np.asarray(m.intercept_),idf=idf.astype(np.float32))
            component_info[name]=[dict(kind='text',file=file,weight=1.,scale=scales[name])]
        else:
            specname,trees=name.rsplit('-',1)
            spec=next(s for s in specs if s['name']==specname)
            m=tree_fit(spec,np.arange(len(bank)));m.shrink(int(trees))
            file=name+'.cbm';m.save_model(str(out/file))
            component_info[name]=[dict(kind='catboost',file=file,weight=1.,scale=scales[name],features=spec['features'])]
        costs.append(dict(fold='full_train',model=name,seconds=time.perf_counter()-tick))
    # Standalone checkpoint folders, no weights shared outside their directory.
    for j,r in enumerate(shortlisted):
        folder=out/f'candidate{j}';folder.mkdir()
        members=[]
        for component,weight in r['members']:
            for member in component_info[component]:
                m=member.copy();m['weight']*=weight;members.append(m)
                if not (folder/m['file']).exists():shutil.copy2(out/m['file'],folder/m['file'])
        config=dict(members=members,decoder=r['decoder'],mask_unparsed=True,recipe=r['name'])
        (folder/'config.json').write_text(json.dumps(config,indent=2))
    costs_total=time.perf_counter()-start
    (out/'training-costs.json').write_text(json.dumps({'fits':costs,'total_seconds':costs_total},indent=2))
    (out/'done-training.json').write_text(json.dumps({'total_seconds':costs_total,'candidates':len(shortlisted)}))
    print('TRAINING COMPLETE',costs_total,flush=True)




def group_main():
    """E33-E34: answer-group learning with question intent and fixed parent fusion."""
    import argparse
    import shutil
    from catboost import CatBoostClassifier,CatBoostRanker,Pool
    from submission import answer_group_features,decode
    from train_e25 import research_metrics,detect_disagreement
    parser=argparse.ArgumentParser()
    parser.add_argument('--answer-groups',action='store_true',required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--parent-run',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir;parent=args.parent_run
    with (out/'started.json').open('x') as f:json.dump({'phase':'answer-group training','parent':str(parent.resolve())},f)
    tick_all=time.perf_counter()
    bank,arrays=load_inputs('train');x=arrays['embeddings']
    with (DATA/'gsm8k_train.jsonl').open(encoding='utf-8') as stream:refs=[json.loads(s) for s in stream if s.strip()]
    assert [b['id'] for b in bank]==[r['id'] for r in refs]
    y=np.array([[parse_answer(c['text'])==parse_answer(r['answer']) for c in b['candidates']] for b,r in zip(bank,refs)],dtype=np.float32)
    disagree=detect_disagreement(bank)
    folds=[np.array(v) for v in json.loads((parent/'folds.json').read_text())]
    with np.load(parent/'oof-final.npz',allow_pickle=False) as cached:
        base_tree=cached['full-d5-800'].copy();base_mlp=cached['mlp3'].copy()
    parent_scales=json.loads((parent/'shortlist-lock.json').read_text())['scales']
    base=.5*(base_tree-base_tree.mean(1,keepdims=True))/parent_scales['full-d5-800']+.5*(base_mlp-base_mlp.mean(1,keepdims=True))/parent_scales['mlp3']
    parent_config=json.loads((parent/'final/config.json').read_text())
    assert parent_config['recipe']=='full-d5-800x0.5+mlp3x0.5/raw'
    features={}
    for mode in ('g-meta','g-context','g-full'):
        f,mapping,qids=answer_group_features(bank,x,mode)
        features[mode]=f
    targets=np.array([y[i,ids[0]] for i,ids in mapping],dtype=np.float32)
    rows_per_question=np.bincount(qids,minlength=len(bank))
    specs=[dict(name='g-meta-q5',mode='g-meta',depth=5,kind='rank'),
           dict(name='g-context-q5',mode='g-context',depth=5,kind='rank'),
           dict(name='g-full-q5',mode='g-full',depth=5,kind='rank'),
           dict(name='g-full-q7',mode='g-full',depth=7,kind='rank'),
           dict(name='g-full-c5',mode='g-full',depth=5,kind='classifier')]
    (out/'specs.json').write_text(json.dumps(specs,indent=2))
    (out/'folds.json').write_text(json.dumps([v.tolist() for v in folds]))
    print('Unique answer groups',len(qids),'hard questions',int(disagree.sum()),'features',{k:v.shape for k,v in features.items()},flush=True)
    oof={};costs=[]
    def fit(spec,fit_questions):
        ids=np.flatnonzero(np.isin(qids,fit_questions)&disagree[qids])
        f=features[spec['mode']][ids];t=targets[ids]
        params=dict(iterations=800,depth=spec['depth'],learning_rate=.05,l2_leaf_reg=5.,border_count=64,random_seed=17,thread_count=4,verbose=False,allow_writing_files=False)
        if spec['kind']=='rank':
            model=CatBoostRanker(loss_function='QuerySoftMax',**params)
            model.fit(Pool(f,label=t,group_id=qids[ids]))
        else:
            model=CatBoostClassifier(loss_function='Logloss',**params)
            model.fit(f,t,sample_weight=1/rows_per_question[qids[ids]])
        return model
    for fold,valid in enumerate(folds):
        fit_questions=np.setdiff1d(np.arange(len(bank)),valid)
        vi=np.flatnonzero(np.isin(qids,valid))
        for spec in specs:
            tick=time.perf_counter();m=fit(spec,fit_questions)
            for trees in (300,800):
                kwargs={'prediction_type':'RawFormulaVal'} if spec['kind']=='classifier' else {}
                scores=m.predict(features[spec['mode']][vi],ntree_end=trees,thread_count=4,**kwargs)
                key=f'{spec["name"]}-{trees}'
                if key not in oof:oof[key]=np.zeros(y.shape,dtype=np.float32)
                for row,value in zip(vi,scores):
                    i,ids=mapping[row];oof[key][i,ids]=value
            elapsed=time.perf_counter()-tick
            costs.append(dict(phase='cv',fold=fold,model=spec['name'],seconds=elapsed))
            m.save_model(str(out/f'fold{fold}-{spec["name"]}.cbm'))
            np.savez(out/'oof-progress.npz',**oof)
            print(f'group fold{fold+1} {spec["name"]} {elapsed:.3f}s',flush=True)
    scales={k:max(float(np.std(v-v.mean(1,keepdims=True))),1e-6) for k,v in oof.items()}
    parse_mask=np.array([[parse_answer(c['text']) is not None for c in b['candidates']] for b in bank])
    records=[]
    for key,scores in oof.items():
        normalized=(scores-scores.mean(1,keepdims=True))/scales[key]
        for weight in (.25,.5,.75,1.):
            s=weight*normalized+(1-weight)*base
            choices=np.where(parse_mask,s,-1e6).argmax(1)
            metrics=research_metrics(y,choices,disagree)
            records.append(dict(name=f'{key}/group-weight{weight:g}',group_model=key,group_weight=weight,scale=scales[key],
                                metrics=metrics,fold_metrics=[research_metrics(y[v],choices[v],disagree[v]) for v in folds]))
    baseline_choices=np.where(parse_mask,base,-1e6).argmax(1)
    baseline=dict(name='E32-parent-fixed',group_model=None,group_weight=0.,metrics=research_metrics(y,baseline_choices,disagree),
                  fold_metrics=[research_metrics(y[v],baseline_choices[v],disagree[v]) for v in folds])
    ranked=sorted(records,key=lambda r:(r['metrics']['disagreement']['correct'],r['metrics']['overall']['correct'],r['metrics']['single_correct']['correct']),reverse=True)
    shortlist=ranked[:3]
    solo=next(r for r in ranked if r['group_weight']==1.)
    if solo['name'] not in {r['name'] for r in shortlist}:shortlist.append(solo)
    shortlist.append(baseline)
    (out/'cv-results.json').write_text(json.dumps(records+[baseline],indent=2))
    (out/'shortlist-lock.json').write_text(json.dumps({'recipes':shortlist,'parent_run':str(parent.resolve()),'rule':'Train OOF top3 plus best group-only, then fixed E32 baseline'},indent=2))
    np.savez(out/'oof-final.npz',**oof,baseline=base,labels=y,disagreement=disagree)
    print('GROUP SHORTLIST',[(r['name'],r['metrics']['disagreement']['correct'],r['metrics']['single_correct']['correct']) for r in shortlist],flush=True)
    models_needed={r['group_model'] for r in shortlist if r['group_model']}
    for key in sorted(models_needed):
        specname,trees=key.rsplit('-',1);spec=next(s for s in specs if s['name']==specname)
        tick=time.perf_counter();m=fit(spec,np.arange(len(bank)));m.shrink(int(trees));m.save_model(str(out/(key+'.cbm')))
        costs.append(dict(phase='full_train',model=key,seconds=time.perf_counter()-tick))
    for j,recipe in enumerate(shortlist):
        folder=out/f'candidate{j}';folder.mkdir();members=[]
        weight=recipe['group_weight']
        if weight>0:
            key=recipe['group_model'];specname,trees=key.rsplit('-',1);spec=next(s for s in specs if s['name']==specname)
            shutil.copy2(out/(key+'.cbm'),folder/(key+'.cbm'))
            members.append(dict(kind='answer_tree',file=key+'.cbm',features=spec['mode'],weight=weight,scale=recipe['scale']))
        if weight<1:
            for original in parent_config['members']:
                member=original.copy();member['weight']*=1-weight;members.append(member)
                shutil.copy2(parent/'final'/member['file'],folder/member['file'])
        config=dict(members=members,decoder='raw',mask_unparsed=True,recipe=recipe['name'])
        (folder/'config.json').write_text(json.dumps(config,indent=2))
    total=time.perf_counter()-tick_all
    (out/'training-costs.json').write_text(json.dumps({'fits':costs,'total_seconds':total,'reused_parent_pipeline_seconds':228.40996879100567},indent=2))
    (out/'done-training.json').write_text(json.dumps({'total_seconds':total,'candidates':len(shortlist)}))
    print('GROUP TRAINING COMPLETE',total,flush=True)



def kernel_main():
    """E35-E36: Train-fitted PCA + Nystroem RBF logistic verifiers."""
    import argparse
    import shutil
    from sklearn.decomposition import PCA
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import LogisticRegression
    from train_e14 import question_weights
    from train_e25 import research_metrics,detect_disagreement
    parser=argparse.ArgumentParser()
    parser.add_argument('--kernel',action='store_true',required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--parent-run',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir;parent=args.parent_run
    with (out/'started.json').open('x') as f:json.dump({'phase':'Nystroem PCA logistic','parent':str(parent.resolve())},f)
    begun=time.perf_counter()
    bank,arrays=load_inputs('train');x=arrays['embeddings'];flat=x.reshape(-1,1024)
    with (DATA/'gsm8k_train.jsonl').open(encoding='utf-8') as f:refs=[json.loads(s) for s in f if s.strip()]
    assert [b['id'] for b in bank]==[r['id'] for r in refs]
    labels=np.array([[parse_answer(c['text'])==parse_answer(r['answer']) for c in b['candidates']] for b,r in zip(bank,refs)],dtype=np.float32)
    valid_answers=np.array([[parse_answer(c['text']) is not None for c in b['candidates']] for b in bank])
    hard=detect_disagreement(bank);weights=np.repeat(question_weights(labels.sum(1).astype(int),'weighted'),4)
    folds=[np.array(v) for v in json.loads((parent/'folds.json').read_text())]
    (out/'folds.json').write_text(json.dumps([v.tolist() for v in folds]))
    scales_parent=json.loads((parent/'shortlist-lock.json').read_text())['scales']
    with np.load(parent/'oof-final.npz',allow_pickle=False) as data:
        a=data['full-d5-800'];b=data['mlp3'];base=.5*(a-a.mean(1,keepdims=True))/scales_parent['full-d5-800']+.5*(b-b.mean(1,keepdims=True))/scales_parent['mlp3']
    oof={};costs=[];parity=[]
    def pca_fit(fitting):
        rows=(fitting[:,None]*4+np.arange(4)).ravel()
        p=PCA(n_components=128,svd_solver='randomized',random_state=17)
        p.fit(flat[rows])
        mean=p.mean_.astype(np.float64);components=p.components_.astype(np.float64)
        scale=np.sqrt(np.maximum(p.explained_variance_.astype(np.float64),1e-8))
        z=((flat.astype(np.float64)-mean)@components.T)/scale
        return mean,components,scale,z
    def train_logistic(phi,fit_rows,C):
        clf=LogisticRegression(C=C,solver='lbfgs',max_iter=250,tol=1e-5)
        clf.fit(phi[fit_rows],labels.reshape(-1)[fit_rows],sample_weight=weights[fit_rows])
        return clf
    for fold,held in enumerate(folds):
        fitting=np.setdiff1d(np.arange(len(bank)),held)
        tick=time.perf_counter();mean,components,scale,z=pca_fit(fitting)
        costs.append(dict(phase='pca',fold=fold,seconds=time.perf_counter()-tick))
        train_mask=np.isin(np.repeat(np.arange(len(bank)),4),fitting)&valid_answers.ravel()
        hard_mask=train_mask&np.repeat(hard,4)
        proto_rows=np.flatnonzero(hard_mask)
        held_rows=(held[:,None]*4+np.arange(4)).ravel()
        for dim in (64,128):
            for factor in (.25,1.,4.):
                tick=time.perf_counter();gamma=factor/dim
                kernel=Nystroem(kernel='rbf',gamma=gamma,n_components=512,random_state=17)
                kernel.fit(z[proto_rows,:dim]);phi=kernel.transform(z[:,:dim])
                costs.append(dict(phase='kernel_map',fold=fold,dim=dim,factor=factor,seconds=time.perf_counter()-tick))
                for subset,mask in (('all',train_mask),('hard',hard_mask)):
                    fit_rows=np.flatnonzero(mask)
                    for C in (.1,1.,10.):
                        name=f'rbf-p{dim}-g{factor:g}-{subset}-C{C:g}'
                        tick=time.perf_counter();clf=train_logistic(phi,fit_rows,C)
                        score=clf.decision_function(phi[held_rows]).reshape(-1,4)
                        if name not in oof:oof[name]=np.zeros(labels.shape,dtype=np.float32)
                        oof[name][held]=score
                        costs.append(dict(phase='logistic',fold=fold,model=name,seconds=time.perf_counter()-tick,iterations=int(clf.n_iter_[0])))
                print(f'kernel fold{fold+1} PCA{dim} gamma{factor:g} done',flush=True)
                np.savez(out/'oof-progress.npz',**oof)
                (out/'costs-progress.json').write_text(json.dumps(costs,indent=2))
    normalized={};scales={};records=[]
    for name,s in oof.items():
        scale=max(float(np.std(s-s.mean(1,keepdims=True))),1e-6);scales[name]=scale
        normalized[name]=(s-s.mean(1,keepdims=True))/scale
        for weight in (.2,.4,.6,.8,1.):
            score=weight*normalized[name]+(1-weight)*base
            choice=np.where(valid_answers,score,-1e6).argmax(1)
            records.append(dict(name=f'{name}/kernel-weight{weight:g}',group_model=name,group_weight=weight,scale=scale,
                 metrics=research_metrics(labels,choice,hard),fold_metrics=[research_metrics(labels[v],choice[v],hard[v]) for v in folds]))
    baseline=dict(name='E32-parent-fixed',group_model=None,group_weight=0.,metrics=research_metrics(labels,np.where(valid_answers,base,-1e6).argmax(1),hard))
    ranked=sorted(records,key=lambda r:(r['metrics']['disagreement']['correct'],r['metrics']['overall']['correct'],r['metrics']['single_correct']['correct']),reverse=True)
    shortlist=ranked[:3];solo=next(r for r in ranked if r['group_weight']==1.)
    if solo['name'] not in {r['name'] for r in shortlist}:shortlist.append(solo)
    shortlist.append(baseline)
    (out/'cv-results.json').write_text(json.dumps(records+[baseline],indent=2))
    (out/'shortlist-lock.json').write_text(json.dumps({'recipes':shortlist,'parent_run':str(parent.resolve()),'rule':'Train OOF top3 + best standalone kernel + fixed baseline'},indent=2))
    np.savez(out/'oof-final.npz',**oof,baseline=base)
    print('KERNEL SHORTLIST',[(r['name'],r['metrics']['disagreement']['correct']) for r in shortlist],flush=True)
    tick=time.perf_counter();mean,components,pc_scale,z=pca_fit(np.arange(len(bank)))
    costs.append(dict(phase='full_pca',seconds=time.perf_counter()-tick))
    proto_rows=np.flatnonzero(valid_answers.ravel()&np.repeat(hard,4))
    maps={};parent_config=json.loads((parent/'final/config.json').read_text())
    for name in sorted({r['group_model'] for r in shortlist if r['group_model']}):
        import re
        match=re.fullmatch(r'rbf-p(\d+)-g([\d.]+)-(all|hard)-C([\d.]+)',name)
        dim,factor,subset,C=match.groups();dim=int(dim);factor=float(factor);C=float(C)
        tick=time.perf_counter();key=(dim,factor)
        if key not in maps:
            kernel=Nystroem(kernel='rbf',gamma=factor/dim,n_components=512,random_state=17)
            kernel.fit(z[proto_rows,:dim]);maps[key]=(kernel,kernel.transform(z[:,:dim]))
        kernel,phi=maps[key]
        fit_rows=np.flatnonzero(valid_answers.ravel()&(np.repeat(hard,4) if subset=='hard' else True))
        clf=train_logistic(phi,fit_rows,C)
        coefficient=kernel.normalization_.T@clf.coef_.ravel()
        landmarks=kernel.components_
        test_z=z[:32,:dim]
        distance=np.maximum((test_z*test_z).sum(1)[:,None]+(landmarks*landmarks).sum(1)[None,:]-2*test_z@landmarks.T,0)
        manual=np.exp(-(factor/dim)*distance)@coefficient+clf.intercept_[0]
        error=float(np.max(np.abs(manual-clf.decision_function(phi[:32]))))
        if error>1e-6:raise AssertionError(f'Kernel export parity {error}')
        parity.append(dict(model=name,max_score_error=error))
        np.savez(out/(name+'.npz'),mean=mean,components=components[:dim],pc_scale=pc_scale[:dim],
                 landmarks=landmarks,kernel_coef=coefficient,intercept=clf.intercept_[0],gamma=np.array(factor/dim))
        costs.append(dict(phase='full_kernel',model=name,seconds=time.perf_counter()-tick))
    for j,recipe in enumerate(shortlist):
        folder=out/f'candidate{j}';folder.mkdir();members=[];weight=recipe['group_weight']
        if weight>0:
            name=recipe['group_model'];shutil.copy2(out/(name+'.npz'),folder/(name+'.npz'))
            members.append(dict(kind='nystrom',file=name+'.npz',weight=weight,scale=recipe['scale']))
        if weight<1:
            for original in parent_config['members']:
                member=original.copy();member['weight']*=1-weight;members.append(member)
                shutil.copy2(parent/'final'/member['file'],folder/member['file'])
        (folder/'config.json').write_text(json.dumps(dict(members=members,decoder='raw',mask_unparsed=True,recipe=recipe['name']),indent=2))
    total=time.perf_counter()-begun
    (out/'kernel-export-parity.json').write_text(json.dumps(parity,indent=2))
    (out/'training-costs.json').write_text(json.dumps({'fits':costs,'total_seconds':total},indent=2))
    (out/'done-training.json').write_text(json.dumps({'total_seconds':total,'candidates':len(shortlist)}))
    print('KERNEL TRAINING COMPLETE',total,flush=True)



def context_main():
    """E37-E38: candidate-level question intent, hard-only and ordered boosting."""
    import argparse
    import shutil
    from catboost import CatBoostClassifier
    from submission import reasoning_features
    from train_e14 import question_weights
    from train_e25 import research_metrics,detect_disagreement
    parser=argparse.ArgumentParser();parser.add_argument('--context',action='store_true',required=True)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--parent-run',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir;parent=args.parent_run
    with (out/'started.json').open('x') as f:json.dump({'phase':'context candidate models','parent':str(parent.resolve())},f)
    start=time.perf_counter();bank,arr=load_inputs('train');x=arr['embeddings']
    with (DATA/'gsm8k_train.jsonl').open(encoding='utf-8') as f:refs=[json.loads(s) for s in f if s.strip()]
    y=np.array([[parse_answer(c['text'])==parse_answer(r['answer']) for c in b['candidates']] for b,r in zip(bank,refs)],dtype=np.float32)
    assert [b['id'] for b in bank]==[r['id'] for r in refs]
    hard=detect_disagreement(bank);weights=question_weights(y.sum(1).astype(int),'weighted')
    valid_answers=np.array([[parse_answer(c['text']) is not None for c in b['candidates']] for b in bank])
    folds=[np.array(v) for v in json.loads((parent/'folds.json').read_text())]
    pr=json.loads((parent/'final-results.json').read_text());recipe=pr['validation']['recipe']
    pc=json.loads((parent/'final/config.json').read_text())
    with np.load(parent/'oof-final.npz',allow_pickle=False) as cache:
        base=cache['baseline'].copy()
        if recipe!='E32-parent-fixed':
            r=next(r for r in json.loads((parent/'cv-results.json').read_text()) if r['name']==recipe)
            raw=cache[r['group_model']];normalized=(raw-raw.mean(1,keepdims=True))/r['scale']
            base=r['group_weight']*normalized+(1-r['group_weight'])*base
    feature={m:reasoning_features(bank,x,m) for m in ('full','context')}
    specs=[dict(name='control',mode='full',seed=17,subset='all',boosting='Plain'),
           *[dict(name=f'context{s}',mode='context',seed=s,subset='all',boosting='Plain') for s in (17,29,43)],
           dict(name='context-hard',mode='context',seed=17,subset='hard',boosting='Plain'),
           dict(name='context-ordered',mode='context',seed=17,subset='all',boosting='Ordered')]
    (out/'specs.json').write_text(json.dumps(specs,indent=2));(out/'folds.json').write_text(json.dumps([v.tolist() for v in folds]))
    oof={};costs=[]
    def fit(spec,fitting):
        if spec['subset']=='hard':fitting=fitting[hard[fitting]]
        f=feature[spec['mode']]
        model=CatBoostClassifier(iterations=800,depth=5,learning_rate=.05,l2_leaf_reg=5.,border_count=128,
             boosting_type=spec['boosting'],loss_function='Logloss',random_seed=spec['seed'],thread_count=4,verbose=False,allow_writing_files=False)
        model.fit(f[fitting].reshape(-1,f.shape[-1]),y[fitting].ravel(),sample_weight=np.repeat(weights[fitting],4))
        return model
    for fold,valid in enumerate(folds):
        fitting=np.setdiff1d(np.arange(len(bank)),valid)
        for spec in specs:
            tick=time.perf_counter();m=fit(spec,fitting);f=feature[spec['mode']][valid]
            for trees in (300,800):
                key=f'{spec["name"]}-{trees}'
                if key not in oof:oof[key]=np.zeros(y.shape,dtype=np.float32)
                oof[key][valid]=m.predict(f.reshape(-1,f.shape[-1]),prediction_type='RawFormulaVal',ntree_end=trees,thread_count=4).reshape(-1,4)
            elapsed=time.perf_counter()-tick;costs.append(dict(phase='cv',fold=fold,model=spec['name'],seconds=elapsed))
            m.save_model(str(out/f'fold{fold}-{spec["name"]}.cbm'))
            np.savez(out/'oof-progress.npz',**oof)
            print(f'context fold{fold+1} {spec["name"]} {elapsed:.3f}s',flush=True)
        for trees in (300,800):
            key=f'context3-{trees}'
            if key not in oof:oof[key]=np.zeros(y.shape,dtype=np.float32)
            oof[key][valid]=np.mean([oof[f'context{s}-{trees}'][valid] for s in (17,29,43)],axis=0)
    records=[]
    for name,s in oof.items():
        scale=max(float(np.std(s-s.mean(1,keepdims=True))),1e-6);normalized=(s-s.mean(1,keepdims=True))/scale
        for weight in (.25,.5,.75,1.):
            score=weight*normalized+(1-weight)*base;choice=np.where(valid_answers,score,-1e6).argmax(1)
            records.append(dict(name=f'{name}/context-weight{weight:g}',group_model=name,group_weight=weight,scale=scale,
              metrics=research_metrics(y,choice,hard),fold_metrics=[research_metrics(y[v],choice[v],hard[v]) for v in folds]))
    baseline=dict(name='frozen-parent-fixed',group_model=None,group_weight=0.,metrics=research_metrics(y,np.where(valid_answers,base,-1e6).argmax(1),hard))
    ranked=sorted(records,key=lambda r:(r['metrics']['disagreement']['correct'],r['metrics']['overall']['correct'],r['metrics']['single_correct']['correct']),reverse=True)
    shortlist=ranked[:3];solo=next(r for r in ranked if r['group_weight']==1.)
    if solo['name'] not in {r['name'] for r in shortlist}:shortlist.append(solo)
    shortlist.append(baseline)
    (out/'cv-results.json').write_text(json.dumps(records+[baseline],indent=2));np.savez(out/'oof-final.npz',**oof,baseline=base)
    (out/'shortlist-lock.json').write_text(json.dumps({'recipes':shortlist,'parent_run':str(parent.resolve()),'rule':'OOF hard top3 + best standalone + current Validation incumbent'},indent=2))
    print('CONTEXT SHORTLIST',[(r['name'],r['metrics']['disagreement']['correct']) for r in shortlist],flush=True)
    needed=set()
    for r in shortlist:
        if r['group_model']:
            name,trees=r['group_model'].rsplit('-',1)
            needed.update((f'context{s}',int(trees)) for s in (17,29,43)) if name=='context3' else needed.add((name,int(trees)))
    for name in sorted({n for n,t in needed}):
        spec=next(s for s in specs if s['name']==name);tick=time.perf_counter();m=fit(spec,np.arange(len(bank)))
        for model_name,trees in sorted(needed):
            if model_name==name:
                piece=m.copy();piece.shrink(trees);piece.save_model(str(out/f'{name}-{trees}.cbm'))
        costs.append(dict(phase='full_train',model=name,seconds=time.perf_counter()-tick))
    for j,r in enumerate(shortlist):
        folder=out/f'candidate{j}';folder.mkdir();members=[];weight=r['group_weight']
        if weight>0:
            name,trees=r['group_model'].rsplit('-',1);names=[f'context{s}' for s in (17,29,43)] if name=='context3' else [name]
            for n in names:
                spec=next(s for s in specs if s['name']==n);file=f'{n}-{trees}.cbm';shutil.copy2(out/file,folder/file)
                members.append(dict(kind='catboost',file=file,features=spec['mode'],weight=weight/len(names),scale=r['scale']))
        if weight<1:
            for original in pc['members']:
                member=original.copy();member['weight']*=1-weight;members.append(member);shutil.copy2(parent/'final'/member['file'],folder/member['file'])
        (folder/'config.json').write_text(json.dumps(dict(members=members,decoder='raw',mask_unparsed=True,recipe=r['name']),indent=2))
    total=time.perf_counter()-start
    (out/'training-costs.json').write_text(json.dumps({'fits':costs,'total_seconds':total},indent=2));(out/'done-training.json').write_text(json.dumps({'total_seconds':total,'candidates':len(shortlist)}))
    print('CONTEXT TRAINING COMPLETE',total,flush=True)

def process_main():
    """E39-E40: compound arithmetic and numeric process completeness."""
    import argparse
    import shutil
    from catboost import CatBoostClassifier
    from submission import reasoning_features
    from train_e14 import question_weights
    from train_e25 import research_metrics,detect_disagreement
    parser=argparse.ArgumentParser();parser.add_argument('--process',action='store_true',required=True)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--parent-run',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir;parent=args.parent_run
    with (out/'started.json').open('x') as f:json.dump({'phase':'numeric process models','parent':str(parent.resolve())},f)
    start=time.perf_counter();bank,arr=load_inputs('train');x=arr['embeddings']
    with (DATA/'gsm8k_train.jsonl').open(encoding='utf-8') as f:refs=[json.loads(s) for s in f if s.strip()]
    y=np.array([[parse_answer(c['text'])==parse_answer(r['answer']) for c in b['candidates']] for b,r in zip(bank,refs)],dtype=np.float32)
    assert [b['id'] for b in bank]==[r['id'] for r in refs]
    hard=detect_disagreement(bank);weights=question_weights(y.sum(1).astype(int),'weighted')
    valid_answers=np.array([[parse_answer(c['text']) is not None for c in b['candidates']] for b in bank])
    folds=[np.array(v) for v in json.loads((parent/'folds.json').read_text())]
    pr=json.loads((parent/'final-results.json').read_text());recipe=pr['validation']['recipe']
    pc=json.loads((parent/'final/config.json').read_text())
    with np.load(parent/'oof-final.npz',allow_pickle=False) as cache:
        base=cache['baseline'].copy()
        if recipe!='E32-parent-fixed':
            r=next(r for r in json.loads((parent/'cv-results.json').read_text()) if r['name']==recipe)
            raw=cache[r['group_model']];normalized=(raw-raw.mean(1,keepdims=True))/r['scale']
            base=r['group_weight']*normalized+(1-r['group_weight'])*base
    feature={m:reasoning_features(bank,x,m) for m in ('full','process','process-meta')}
    assert all(np.isfinite(a).all() for a in feature.values())
    print('Process feature dimensions', {k:v.shape[-1] for k,v in feature.items()},flush=True)
    specs=[dict(name='control',mode='full',seed=17,subset='all',boosting='Plain',depth=5),
           *[dict(name=f'process{s}',mode='process',seed=s,subset='all',boosting='Plain',depth=5) for s in (17,29)],
           dict(name='process-deep',mode='process',seed=17,subset='all',boosting='Plain',depth=7),
           dict(name='process-meta',mode='process-meta',seed=17,subset='all',boosting='Plain',depth=5),
           dict(name='process-hard',mode='process',seed=17,subset='hard',boosting='Plain',depth=5)]
    (out/'specs.json').write_text(json.dumps(specs,indent=2));(out/'folds.json').write_text(json.dumps([v.tolist() for v in folds]))
    oof={};costs=[]
    def fit(spec,fitting):
        if spec['subset']=='hard':fitting=fitting[hard[fitting]]
        f=feature[spec['mode']]
        model=CatBoostClassifier(iterations=800,depth=spec['depth'],learning_rate=.05,l2_leaf_reg=5.,border_count=64,
             boosting_type=spec['boosting'],loss_function='Logloss',random_seed=spec['seed'],thread_count=4,verbose=False,allow_writing_files=False)
        model.fit(f[fitting].reshape(-1,f.shape[-1]),y[fitting].ravel(),sample_weight=np.repeat(weights[fitting],4))
        return model
    for fold,valid in enumerate(folds):
        fitting=np.setdiff1d(np.arange(len(bank)),valid)
        for spec in specs:
            tick=time.perf_counter();m=fit(spec,fitting);f=feature[spec['mode']][valid]
            for trees in (300,800):
                key=f'{spec["name"]}-{trees}'
                if key not in oof:oof[key]=np.zeros(y.shape,dtype=np.float32)
                oof[key][valid]=m.predict(f.reshape(-1,f.shape[-1]),prediction_type='RawFormulaVal',ntree_end=trees,thread_count=4).reshape(-1,4)
            elapsed=time.perf_counter()-tick;costs.append(dict(phase='cv',fold=fold,model=spec['name'],seconds=elapsed))
            m.save_model(str(out/f'fold{fold}-{spec["name"]}.cbm'))
            np.savez(out/'oof-progress.npz',**oof)
            print(f'process fold{fold+1} {spec["name"]} {elapsed:.3f}s',flush=True)
        for trees in (300,800):
            key=f'process2-{trees}'
            if key not in oof:oof[key]=np.zeros(y.shape,dtype=np.float32)
            oof[key][valid]=np.mean([oof[f'process{s}-{trees}'][valid] for s in (17,29)],axis=0)
    records=[]
    for name,s in oof.items():
        scale=max(float(np.std(s-s.mean(1,keepdims=True))),1e-6);normalized=(s-s.mean(1,keepdims=True))/scale
        for weight in (.25,.5,.75,1.):
            score=weight*normalized+(1-weight)*base;choice=np.where(valid_answers,score,-1e6).argmax(1)
            records.append(dict(name=f'{name}/process-weight{weight:g}',group_model=name,group_weight=weight,scale=scale,
              metrics=research_metrics(y,choice,hard),fold_metrics=[research_metrics(y[v],choice[v],hard[v]) for v in folds]))
    baseline=dict(name='frozen-parent-fixed',group_model=None,group_weight=0.,metrics=research_metrics(y,np.where(valid_answers,base,-1e6).argmax(1),hard))
    ranked=sorted(records,key=lambda r:(r['metrics']['disagreement']['correct'],r['metrics']['overall']['correct'],r['metrics']['single_correct']['correct']),reverse=True)
    shortlist=[];routes=set()
    for r in ranked:
        route=r['group_model'].rsplit('-',1)[0]
        if route not in routes:
            shortlist.append(r);routes.add(route)
        if len(shortlist)==3:break
    solo=next(r for r in ranked if r['group_weight']==1.)
    if solo['name'] not in {r['name'] for r in shortlist}:shortlist.append(solo)
    shortlist.append(baseline)
    (out/'cv-results.json').write_text(json.dumps(records+[baseline],indent=2));np.savez(out/'oof-final.npz',**oof,baseline=base)
    (out/'shortlist-lock.json').write_text(json.dumps({'recipes':shortlist,'parent_run':str(parent.resolve()),'rule':'OOF best3 distinct routes + best standalone + current Validation incumbent'},indent=2))
    print('PROCESS SHORTLIST',[(r['name'],r['metrics']['disagreement']['correct']) for r in shortlist],flush=True)
    needed=set()
    for r in shortlist:
        if r['group_model']:
            name,trees=r['group_model'].rsplit('-',1)
            needed.update((f'process{s}',int(trees)) for s in (17,29)) if name=='process2' else needed.add((name,int(trees)))
    for name in sorted({n for n,t in needed}):
        spec=next(s for s in specs if s['name']==name);tick=time.perf_counter();m=fit(spec,np.arange(len(bank)))
        for model_name,trees in sorted(needed):
            if model_name==name:
                piece=m.copy();piece.shrink(trees);piece.save_model(str(out/f'{name}-{trees}.cbm'))
        costs.append(dict(phase='full_train',model=name,seconds=time.perf_counter()-tick))
    for j,r in enumerate(shortlist):
        folder=out/f'candidate{j}';folder.mkdir();members=[];weight=r['group_weight']
        if weight>0:
            name,trees=r['group_model'].rsplit('-',1);names=[f'process{s}' for s in (17,29)] if name=='process2' else [name]
            for n in names:
                spec=next(s for s in specs if s['name']==n);file=f'{n}-{trees}.cbm';shutil.copy2(out/file,folder/file)
                members.append(dict(kind='catboost',file=file,features=spec['mode'],weight=weight/len(names),scale=r['scale']))
        if weight<1:
            for original in pc['members']:
                member=original.copy();member['weight']*=1-weight;members.append(member);shutil.copy2(parent/'final'/member['file'],folder/member['file'])
        (folder/'config.json').write_text(json.dumps(dict(members=members,decoder='raw',mask_unparsed=True,recipe=r['name']),indent=2))
    total=time.perf_counter()-start
    (out/'training-costs.json').write_text(json.dumps({'fits':costs,'total_seconds':total},indent=2));(out/'done-training.json').write_text(json.dumps({'total_seconds':total,'candidates':len(shortlist)}))
    print('PROCESS TRAINING COMPLETE',total,flush=True)

def repair_main():
    """E43-E44: arithmetic repair and candidate agreement."""
    import argparse
    import shutil
    from catboost import CatBoostClassifier
    from submission import reasoning_features
    from train_e14 import question_weights
    from train_e25 import research_metrics,detect_disagreement
    parser=argparse.ArgumentParser();parser.add_argument('--repair',action='store_true',required=True)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--parent-run',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir;parent=args.parent_run
    with (out/'started.json').open('x') as f:json.dump({'phase':'arithmetic repair models','parent':str(parent.resolve())},f)
    start=time.perf_counter();bank,arr=load_inputs('train');x=arr['embeddings']
    with (DATA/'gsm8k_train.jsonl').open(encoding='utf-8') as f:refs=[json.loads(s) for s in f if s.strip()]
    y=np.array([[parse_answer(c['text'])==parse_answer(r['answer']) for c in b['candidates']] for b,r in zip(bank,refs)],dtype=np.float32)
    assert [b['id'] for b in bank]==[r['id'] for r in refs]
    hard=detect_disagreement(bank);weights=question_weights(y.sum(1).astype(int),'weighted')
    valid_answers=np.array([[parse_answer(c['text']) is not None for c in b['candidates']] for b in bank])
    folds=[np.array(v) for v in json.loads((parent/'folds.json').read_text())]
    pr=json.loads((parent/'final-results.json').read_text());recipe=pr['validation']['recipe']
    pc=json.loads((parent/'final/config.json').read_text())
    with np.load(parent/'oof-final.npz',allow_pickle=False) as cache:
        base=cache['baseline'].copy()
        if recipe!='E32-parent-fixed':
            r=next(r for r in json.loads((parent/'cv-results.json').read_text()) if r['name']==recipe)
            raw=cache[r['group_model']];normalized=(raw-raw.mean(1,keepdims=True))/r['scale']
            base=r['group_weight']*normalized+(1-r['group_weight'])*base
    feature={m:reasoning_features(bank,x,m) for m in ('process-meta','repair','repair-full','repair-only')}
    assert all(np.isfinite(a).all() for a in feature.values())
    print('Process feature dimensions', {k:v.shape[-1] for k,v in feature.items()},flush=True)
    specs=[dict(name='control',mode='process-meta',seed=17,subset='all',boosting='Plain',depth=5),
           *[dict(name=f'repair{s}',mode='repair',seed=s,subset='all',boosting='Plain',depth=5) for s in (17,29)],
           dict(name='repair-deep',mode='repair',seed=17,subset='all',boosting='Plain',depth=6),
           dict(name='repair-full',mode='repair-full',seed=17,subset='all',boosting='Plain',depth=5),
           dict(name='repair-only',mode='repair-only',seed=17,subset='all',boosting='Plain',depth=5)]
    (out/'specs.json').write_text(json.dumps(specs,indent=2));(out/'folds.json').write_text(json.dumps([v.tolist() for v in folds]))
    oof={};costs=[]
    def fit(spec,fitting):
        if spec['subset']=='hard':fitting=fitting[hard[fitting]]
        f=feature[spec['mode']]
        model=CatBoostClassifier(iterations=800,depth=spec['depth'],learning_rate=.05,l2_leaf_reg=5.,border_count=64,
             boosting_type=spec['boosting'],loss_function='Logloss',random_seed=spec['seed'],thread_count=4,verbose=False,allow_writing_files=False)
        model.fit(f[fitting].reshape(-1,f.shape[-1]),y[fitting].ravel(),sample_weight=np.repeat(weights[fitting],4))
        return model
    for fold,valid in enumerate(folds):
        fitting=np.setdiff1d(np.arange(len(bank)),valid)
        for spec in specs:
            tick=time.perf_counter();m=fit(spec,fitting);f=feature[spec['mode']][valid]
            for trees in (300,800):
                key=f'{spec["name"]}-{trees}'
                if key not in oof:oof[key]=np.zeros(y.shape,dtype=np.float32)
                oof[key][valid]=m.predict(f.reshape(-1,f.shape[-1]),prediction_type='RawFormulaVal',ntree_end=trees,thread_count=4).reshape(-1,4)
            elapsed=time.perf_counter()-tick;costs.append(dict(phase='cv',fold=fold,model=spec['name'],seconds=elapsed))
            m.save_model(str(out/f'fold{fold}-{spec["name"]}.cbm'))
            np.savez(out/'oof-progress.npz',**oof)
            print(f'repair fold{fold+1} {spec["name"]} {elapsed:.3f}s',flush=True)
        for trees in (300,800):
            key=f'repair2-{trees}'
            if key not in oof:oof[key]=np.zeros(y.shape,dtype=np.float32)
            oof[key][valid]=np.mean([oof[f'repair{s}-{trees}'][valid] for s in (17,29)],axis=0)
    records=[]
    for name,s in oof.items():
        scale=max(float(np.std(s-s.mean(1,keepdims=True))),1e-6);normalized=(s-s.mean(1,keepdims=True))/scale
        for weight in (.25,.5,.75,1.):
            score=weight*normalized+(1-weight)*base;choice=np.where(valid_answers,score,-1e6).argmax(1)
            records.append(dict(name=f'{name}/repair-weight{weight:g}',group_model=name,group_weight=weight,scale=scale,
              metrics=research_metrics(y,choice,hard),fold_metrics=[research_metrics(y[v],choice[v],hard[v]) for v in folds]))
    baseline=dict(name='frozen-parent-fixed',group_model=None,group_weight=0.,metrics=research_metrics(y,np.where(valid_answers,base,-1e6).argmax(1),hard))
    ranked=sorted(records,key=lambda r:(r['metrics']['disagreement']['correct'],r['metrics']['overall']['correct'],r['metrics']['single_correct']['correct']),reverse=True)
    shortlist=[];routes=set()
    for r in ranked:
        name=r['group_model'].rsplit('-',1)[0]
        route='repair' if name in ('repair17','repair29','repair2','repair-deep') else name
        if route not in routes:
            shortlist.append(r);routes.add(route)
        if len(shortlist)==3:break
    solo=next(r for r in ranked if r['group_weight']==1.)
    if solo['name'] not in {r['name'] for r in shortlist}:shortlist.append(solo)
    shortlist.append(baseline)
    (out/'cv-results.json').write_text(json.dumps(records+[baseline],indent=2));np.savez(out/'oof-final.npz',**oof,baseline=base)
    (out/'shortlist-lock.json').write_text(json.dumps({'recipes':shortlist,'parent_run':str(parent.resolve()),'rule':'OOF best3 distinct feature families + best standalone + current Validation incumbent'},indent=2))
    print('REPAIR SHORTLIST',[(r['name'],r['metrics']['disagreement']['correct']) for r in shortlist],flush=True)
    needed=set()
    for r in shortlist:
        if r['group_model']:
            name,trees=r['group_model'].rsplit('-',1)
            needed.update((f'repair{s}',int(trees)) for s in (17,29)) if name=='repair2' else needed.add((name,int(trees)))
    for name in sorted({n for n,t in needed}):
        spec=next(s for s in specs if s['name']==name);tick=time.perf_counter();m=fit(spec,np.arange(len(bank)))
        for model_name,trees in sorted(needed):
            if model_name==name:
                piece=m.copy();piece.shrink(trees);piece.save_model(str(out/f'{name}-{trees}.cbm'))
        costs.append(dict(phase='full_train',model=name,seconds=time.perf_counter()-tick))
    for j,r in enumerate(shortlist):
        folder=out/f'candidate{j}';folder.mkdir();members=[];weight=r['group_weight']
        if weight>0:
            name,trees=r['group_model'].rsplit('-',1);names=[f'repair{s}' for s in (17,29)] if name=='repair2' else [name]
            for n in names:
                spec=next(s for s in specs if s['name']==n);file=f'{n}-{trees}.cbm';shutil.copy2(out/file,folder/file)
                members.append(dict(kind='catboost',file=file,features=spec['mode'],weight=weight/len(names),scale=r['scale']))
        if weight<1:
            for original in pc['members']:
                member=original.copy();member['weight']*=1-weight;members.append(member);shutil.copy2(parent/'final'/member['file'],folder/member['file'])
        (folder/'config.json').write_text(json.dumps(dict(members=members,decoder='raw',mask_unparsed=True,recipe=r['name']),indent=2))
    total=time.perf_counter()-start
    (out/'training-costs.json').write_text(json.dumps({'fits':costs,'total_seconds':total},indent=2));(out/'done-training.json').write_text(json.dumps({'total_seconds':total,'candidates':len(shortlist)}))
    print('REPAIR TRAINING COMPLETE',total,flush=True)

def stability_main():
    """E41-E42: seeded ensembles and pairwise preference on process features."""
    import argparse
    import shutil
    from catboost import CatBoostClassifier,CatBoostRanker,Pool
    from submission import reasoning_features
    from train_e14 import question_weights
    from train_e25 import research_metrics,detect_disagreement
    parser=argparse.ArgumentParser();parser.add_argument('--stability',action='store_true',required=True)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--parent-run',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir;parent=args.parent_run
    with (out/'started.json').open('x') as f:json.dump({'phase':'process stability and ranking','parent':str(parent.resolve())},f)
    start=time.perf_counter();bank,arr=load_inputs('train');x=arr['embeddings']
    with (DATA/'gsm8k_train.jsonl').open(encoding='utf-8') as f:refs=[json.loads(s) for s in f if s.strip()]
    y=np.array([[parse_answer(c['text'])==parse_answer(r['answer']) for c in b['candidates']] for b,r in zip(bank,refs)],dtype=np.float32)
    assert [b['id'] for b in bank]==[r['id'] for r in refs]
    hard=detect_disagreement(bank);weights=question_weights(y.sum(1).astype(int),'weighted')
    valid_answers=np.array([[parse_answer(c['text']) is not None for c in b['candidates']] for b in bank])
    folds=[np.array(v) for v in json.loads((parent/'folds.json').read_text())]
    pr=json.loads((parent/'final-results.json').read_text());recipe=pr['validation']['recipe']
    pc=json.loads((parent/'final/config.json').read_text())
    with np.load(parent/'oof-final.npz',allow_pickle=False) as cache:
        base=cache['baseline'].copy()
        if recipe!='E32-parent-fixed':
            r=next(r for r in json.loads((parent/'cv-results.json').read_text()) if r['name']==recipe)
            raw=cache[r['group_model']];normalized=(raw-raw.mean(1,keepdims=True))/r['scale']
            base=r['group_weight']*normalized+(1-r['group_weight'])*base
    feature={'process-meta':reasoning_features(bank,x,'process-meta')}
    assert all(np.isfinite(a).all() for a in feature.values())
    print('Process feature dimensions', {k:v.shape[-1] for k,v in feature.items()},flush=True)
    specs=[dict(name=f'{family}-s{seed}',family=family,mode='process-meta',seed=seed,depth=depth,kind=kind)
           for family,depth,kind in [('bce5',5,'classifier'),('bce6',6,'classifier'),('pair5',5,'ranker')]
           for seed in (17,29,43)]
    groups={family+'-mean3':[f'{family}-s{s}' for s in (17,29,43)] for family in ('bce5','bce6','pair5')}
    (out/'specs.json').write_text(json.dumps(specs,indent=2));(out/'folds.json').write_text(json.dumps([v.tolist() for v in folds]))
    oof={};costs=[]
    def fit(spec,fitting):
        features=feature['process-meta']
        params=dict(iterations=1200,depth=spec['depth'],learning_rate=.05,l2_leaf_reg=5.,border_count=64,
             random_seed=spec['seed'],thread_count=4,verbose=False,allow_writing_files=False)
        if spec['kind']=='classifier':
            model=CatBoostClassifier(loss_function='Logloss',**params)
            model.fit(features[fitting].reshape(-1,features.shape[-1]),y[fitting].ravel(),sample_weight=np.repeat(weights[fitting],4))
        else:
            fitting=fitting[y[fitting].sum(1)<4]
            pool=Pool(features[fitting].reshape(-1,features.shape[-1]),y[fitting].ravel(),
                      group_id=np.repeat(np.arange(len(fitting)),4),group_weight=np.repeat(weights[fitting],4))
            model=CatBoostRanker(loss_function='PairLogit',**params);model.fit(pool)
        return model
    for fold,valid in enumerate(folds):
        fitting=np.setdiff1d(np.arange(len(bank)),valid)
        for spec in specs:
            tick=time.perf_counter();m=fit(spec,fitting);f=feature[spec['mode']][valid]
            for trees in (800,1200):
                key=f'{spec["name"]}-{trees}'
                if key not in oof:oof[key]=np.zeros(y.shape,dtype=np.float32)
                kwargs={'prediction_type':'RawFormulaVal'} if spec['kind']=='classifier' else {}
                oof[key][valid]=m.predict(f.reshape(-1,f.shape[-1]),ntree_end=trees,thread_count=4,**kwargs).reshape(-1,4)
            elapsed=time.perf_counter()-tick;costs.append(dict(phase='cv',fold=fold,model=spec['name'],seconds=elapsed))
            m.save_model(str(out/f'fold{fold}-{spec["name"]}.cbm'))
            np.savez(out/'oof-progress.npz',**oof)
            print(f'stability fold{fold+1} {spec["name"]} {elapsed:.3f}s',flush=True)
        for group,names in groups.items():
            for trees in (800,1200):
                key=f'{group}-{trees}'
                if key not in oof:oof[key]=np.zeros(y.shape,dtype=np.float32)
                oof[key][valid]=np.mean([oof[f'{name}-{trees}'][valid] for name in names],axis=0)
    records=[]
    for name,s in oof.items():
        scale=max(float(np.std(s-s.mean(1,keepdims=True))),1e-6);normalized=(s-s.mean(1,keepdims=True))/scale
        for weight in (.25,.5,.75,1.):
            score=weight*normalized+(1-weight)*base;choice=np.where(valid_answers,score,-1e6).argmax(1)
            records.append(dict(name=f'{name}/stability-weight{weight:g}',group_model=name,group_weight=weight,scale=scale,
              metrics=research_metrics(y,choice,hard),fold_metrics=[research_metrics(y[v],choice[v],hard[v]) for v in folds]))
    baseline=dict(name='frozen-parent-fixed',group_model=None,group_weight=0.,metrics=research_metrics(y,np.where(valid_answers,base,-1e6).argmax(1),hard))
    ranked=sorted(records,key=lambda r:(r['metrics']['disagreement']['correct'],r['metrics']['overall']['correct'],r['metrics']['single_correct']['correct']),reverse=True)
    shortlist=[];routes=set()
    for r in ranked:
        route=r['group_model'].split('-')[0]
        if route not in routes:
            shortlist.append(r);routes.add(route)
        if len(shortlist)==3:break
    solo=next(r for r in ranked if r['group_weight']==1.)
    if solo['name'] not in {r['name'] for r in shortlist}:shortlist.append(solo)
    shortlist.append(baseline)
    (out/'cv-results.json').write_text(json.dumps(records+[baseline],indent=2));np.savez(out/'oof-final.npz',**oof,baseline=base)
    (out/'shortlist-lock.json').write_text(json.dumps({'recipes':shortlist,'parent_run':str(parent.resolve()),'rule':'OOF best3 distinct objective/depth families + best standalone + current Validation incumbent'},indent=2))
    print('STABILITY SHORTLIST',[(r['name'],r['metrics']['disagreement']['correct']) for r in shortlist],flush=True)
    needed=set()
    for r in shortlist:
        if r['group_model']:
            name,trees=r['group_model'].rsplit('-',1)
            needed.update((n,int(trees)) for n in groups.get(name,[name]))
    for name in sorted({n for n,t in needed}):
        spec=next(s for s in specs if s['name']==name);tick=time.perf_counter();m=fit(spec,np.arange(len(bank)))
        for model_name,trees in sorted(needed):
            if model_name==name:
                piece=m.copy();piece.shrink(trees);piece.save_model(str(out/f'{name}-{trees}.cbm'))
        costs.append(dict(phase='full_train',model=name,seconds=time.perf_counter()-tick))
    for j,r in enumerate(shortlist):
        folder=out/f'candidate{j}';folder.mkdir();members=[];weight=r['group_weight']
        if weight>0:
            name,trees=r['group_model'].rsplit('-',1);names=groups.get(name,[name])
            for n in names:
                spec=next(s for s in specs if s['name']==n);file=f'{n}-{trees}.cbm';shutil.copy2(out/file,folder/file)
                members.append(dict(kind='catboost',file=file,features=spec['mode'],weight=weight/len(names),scale=r['scale']))
        if weight<1:
            for original in pc['members']:
                member=original.copy();member['weight']*=1-weight;members.append(member);shutil.copy2(parent/'final'/member['file'],folder/member['file'])
        (folder/'config.json').write_text(json.dumps(dict(members=members,decoder='raw',mask_unparsed=True,recipe=r['name']),indent=2))
    total=time.perf_counter()-start
    (out/'training-costs.json').write_text(json.dumps({'fits':costs,'total_seconds':total},indent=2));(out/'done-training.json').write_text(json.dumps({'total_seconds':total,'candidates':len(shortlist)}))
    print('STABILITY TRAINING COMPLETE',total,flush=True)

if __name__ == "__main__":
    import sys
    if "--repair" in sys.argv:
        repair_main()
    elif "--stability" in sys.argv:
        stability_main()
    elif "--process" in sys.argv:
        process_main()
    elif "--context" in sys.argv:
        context_main()
    elif "--kernel" in sys.argv:
        kernel_main()
    elif "--answer-groups" in sys.argv:
        group_main()
    elif "--expanded" in sys.argv:
        expanded_main()
    else:
        main()
