
"""MP2 frozen candidate scorer. Versioned bundles or legacy E25 checkpoints.

Only uses passed inputs and the explicitly supplied checkpoint. No references.
"""

import numpy as np


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("Checkpoint is required")

    x = np.asarray(
        arrays["embeddings"],
        dtype=np.float32,
    )

    n = len(bank)

    if x.shape != (n, 4, 1024):
        raise ValueError(f"Invalid embeddings shape: {x.shape}")

    if n == 0:
        return []
    if not np.isfinite(x).all():
        raise ValueError("Embeddings must be finite")
    if checkpoint.is_dir():
        return expanded_predict(bank, x, checkpoint)
    with np.load(checkpoint, allow_pickle=False) as model:
        flat = (x.reshape(-1, 1024) - model["mean"]) / model["std"]
        if "format_version" in model:
            if int(model["format_version"][0]) != 2:
                raise ValueError("Unsupported checkpoint version")
            logits = []
            for j in range(len(model["seeds"])):
                hidden = np.maximum(flat @ model[f"m{j}_w1"] + model[f"m{j}_b1"], 0)
                logits.append((hidden @ model[f"m{j}_w2"] + model[f"m{j}_b2"]).reshape(n, 4))
            scores = np.mean(logits, axis=0)
            decoder = str(model["decoder"].item())
        else:
            hidden = np.maximum(flat @ model["w1"] + model["b1"], 0)
            scores = (hidden @ model["w2"] + model["b2"]).reshape(n, 4)
            decoder = "raw"
    return decode(scores, bank, decoder)


def decode(scores, bank, decoder):
    from collections import Counter, defaultdict
    from common import parse_answer
    if decoder == "raw":
        return scores.argmax(1).astype(int).tolist()
    selected = []
    for logits, row in zip(scores, bank):
        answers = [parse_answer(c["text"]) for c in row["candidates"]]
        if decoder == "gate":
            counts = Counter(a for a in answers if a is not None)
            votes = np.array([counts[a]/4 if a is not None else 0 for a in answers])
            raw = int(logits.argmax())
            ordered = np.sort(logits)
            gap = float(ordered[-1]-ordered[-2]) / max(float(logits.std()), 1e-6)
            support = counts[answers[raw]] if answers[raw] is not None else 0
            confident_minority = support < max(counts.values(), default=0) and gap >= 1.0
            selected.append(int(np.argmax(logits + (0 if confident_minority else 0.5)*votes)))
        elif decoder == "aggregate":
            z = logits.astype(np.float64)-logits.max()
            probabilities = np.exp(z)
            probabilities /= probabilities.sum()
            groups = defaultdict(list)
            for j, answer in enumerate(answers):
                groups[("answer", answer) if answer is not None else ("unparsed", j)].append(j)
            best = max(groups.values(), key=lambda ids: (sum(probabilities[j] for j in ids),
                                                        max(logits[j] for j in ids)))
            selected.append(int(max(best, key=lambda j: float(logits[j]))))
        else:
            raise ValueError(f"Unknown decoder: {decoder}")
    return selected


def reasoning_features(bank, embeddings, mode="full"):
    """Fixed input-only features; no labels, reference access or ID features."""
    import re
    from collections import Counter
    from fractions import Fraction
    from common import parse_answer
    number = r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?"
    equation = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+*×/÷−-])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")
    cues = ("therefore","however","wait","mistake","incorrect","assum","approximately","round", "remaining","total", "each","per", "percent", "original", "discount", "half", "twice", "difference", "ratio", "####", "boxed", "final answer", "numeric answer", "<", "let ", "check", "step", "we need", "answer:")
    all_features=[]
    for row in bank:
        question=row["question"].lower()
        q_tokens=set(re.findall(r"[a-z]+",question))
        qnums=[]
        for s in re.findall(number,question):
            try:qnums.append(float(Fraction(s.replace(',',''))))
            except (ValueError,ZeroDivisionError):pass
        answers=[parse_answer(c['text']) for c in row['candidates']]
        counts=Counter(a for a in answers if a is not None)
        raw=[]
        for j,c in enumerate(row['candidates']):
            text=c['text']; low=text.lower(); tokens=re.findall(r'[a-z]+',low)
            token_set=set(tokens); nums=re.findall(number,text)
            ans=float(Fraction(answers[j])) if answers[j] is not None else 0.
            valid=invalid=0
            for match in equation.finditer(text):
                aa,op,bb,cc=match.groups(); aa,bb,cc=map(float,(aa,bb,cc))
                if op in ('/','÷') and abs(bb)<1e-12:continue
                expected=aa+bb if op=='+' else aa-bb if op in ('-','−') else aa*bb if op in ('*','×') else aa/bb
                if abs(expected-cc)<=1e-5*max(1.,abs(expected),abs(cc)):valid+=1
                else:invalid+=1
            # Algebraic relations to prompt numbers are evidence, never hard rules.
            n=len(qnums); rels=[0.,0.,0.,0.,0.,0.]
            if n:
                close=lambda v: abs(v-ans)<=1e-5*max(1.,abs(v),abs(ans))
                rels[:3]=[float(any(close(v) for v in qnums)),float(close(sum(qnums))),float(close(np.prod(qnums[:10])))]
                for aa in qnums[:12]:
                    for bb in qnums[:12]:
                        rels[3]=max(rels[3],float(close(aa-bb)))
                        rels[4]=max(rels[4],float(close(aa*bb)))
                        if abs(bb)>1e-10:rels[5]=max(rels[5],float(close(aa/bb)))
            f=[np.log1p(len(text)),np.log1p(len(tokens)),np.log1p(len(text.splitlines())),
               len(token_set)/max(len(tokens),1),len(q_tokens&token_set)/max(len(q_tokens),1),
               np.log1p(len(nums)),np.log1p(len(qnums)),float(answers[j] is not None),
               counts.get(answers[j],0)/4 if answers[j] is not None else 0,
               max(counts.values(),default=0)/4,len(counts)/4,
               np.sign(ans)*np.log1p(abs(ans)),float(ans<0),float(ans==0),float(ans.is_integer()),
               np.log1p(valid),np.log1p(invalid),invalid/max(valid+invalid,1),
               text.count('='),text.count('\\frac'),text.count('**'),text.count('$'),text.count('%')]
            f+=rels
            f += [np.log1p(low.count(cue)) for cue in cues]
            raw.append(f)
        raw=np.asarray(raw,dtype=np.float32)
        # Relative metadata allows comparisons without learning candidate positions.
        all_features.append(np.concatenate([raw,raw-raw.mean(0,keepdims=True)],axis=1))
    metadata=np.asarray(all_features,dtype=np.float32)
    if mode in ('repair','repair-full','repair-only'):
        parts=[metadata,repair_features(bank)]
        if mode!='repair-only':parts.append(process_features(bank))
        if mode=='repair-full':parts.insert(0,embeddings)
        return np.concatenate(parts,axis=2).astype(np.float32)
    if mode in ('process','process-meta'):
        extra=process_features(bank)
        parts=[metadata,extra] if mode=='process-meta' else [embeddings,metadata,extra]
        return np.concatenate(parts,axis=2).astype(np.float32)
    if mode=='metadata':return metadata
    if mode=='embedding':return np.asarray(embeddings,dtype=np.float32)
    if mode=='relative':return np.concatenate([embeddings,embeddings-embeddings.mean(1,keepdims=True),metadata],axis=2)
    if mode=='context':
        cues=('total','each','per','left','remain','more','less','original','pay','paid','cost','profit',
              'earn','percent','discount','after','before','twice','half','average','ratio','difference',
              'how many','how much','how long','how far','speed','rate','hour','minute','day','week','year',
              'altogether','combined','rest','additional','increase','decrease','together','return',
              'sell','bought','buy','spend','spent','save','saved')
        context=[]
        for row in bank:
            question=row['question'].lower();tail=re.split(r'[.!]\s+',question)[-1]
            context.append([np.log1p(t.count(c)) for t in (question,tail) for c in cues])
        context=np.repeat(np.asarray(context,dtype=np.float32)[:,None,:],4,axis=1)
        return np.concatenate([embeddings,metadata,context],axis=2)
    return np.concatenate([embeddings,metadata],axis=2).astype(np.float32)


def text_documents(bank):
    import re
    documents=[]
    for row in bank:
        for candidate in row['candidates']:
            low=candidate['text'].lower()
            low=re.sub(r'\d+(?:[.,/]\d+)*',' NUM ',low)
            question=re.sub(r'\d+(?:[.,/]\d+)*',' NUM ',row['question'].lower())
            documents.append(low+' '+' '.join('q_'+s for s in re.findall('[a-z]+',question)))
    return documents


def expanded_scores(bank, x, checkpoint):
    import json
    from pathlib import Path
    config=json.loads((Path(checkpoint)/'config.json').read_text())
    logits=[]
    for member in config['members']:
        kind=member['kind']
        if kind=='catboost':
            from catboost import CatBoost
            m=CatBoost();m.load_model(str(Path(checkpoint)/member['file']))
            features=reasoning_features(bank,x,member['features'])
            s=m.predict(features.reshape(-1,features.shape[-1]),prediction_type='RawFormulaVal',thread_count=4).reshape(-1,4)
        elif kind=='answer_tree':
            from catboost import CatBoost
            m=CatBoost();m.load_model(str(Path(checkpoint)/member['file']))
            features,mapping,group_ids=answer_group_features(bank,x,member['features'])
            values=m.predict(features,prediction_type='RawFormulaVal',thread_count=4)
            s=np.zeros((len(bank),4),dtype=np.float64)
            for value,(i,ids) in zip(values,mapping):s[i,ids]=value
        elif kind=='mlp':
            with np.load(Path(checkpoint)/member['file'],allow_pickle=False) as m:
                z=(x.reshape(-1,1024)-m['mean'])/m['std']
                h=np.maximum(z@m['w1']+m['b1'],0)
                s=(h@m['w2']+m['b2']).reshape(-1,4)
        elif kind=='nystrom':
            with np.load(Path(checkpoint)/member['file'],allow_pickle=False) as m:
                z=((x.reshape(-1,1024).astype(np.float64)-m['mean'])@m['components'].T)/m['pc_scale']
                landmarks=m['landmarks']
                distance=np.maximum((z*z).sum(1)[:,None]+(landmarks*landmarks).sum(1)[None,:]-2*z@landmarks.T,0)
                kernel=np.exp(-float(m['gamma'])*distance)
                s=(kernel@m['kernel_coef']+m['intercept']).reshape(-1,4)
        elif kind=='text':
            from sklearn.feature_extraction.text import HashingVectorizer
            from sklearn.preprocessing import normalize
            with np.load(Path(checkpoint)/member['file'],allow_pickle=False) as m:
                vectorizer=HashingVectorizer(n_features=65536,ngram_range=(1,2),alternate_sign=False,norm=None)
                matrix=vectorizer.transform(text_documents(bank))
                matrix.data=np.log1p(matrix.data)
                matrix=normalize(matrix.multiply(m['idf']))
                s=(matrix@m['weights']+m['intercept']).reshape(-1,4)
        else:raise ValueError(kind)
        # Fusion uses question-centered scores and train-OOF fitted scale.
        logits.append(member['weight']*(s-s.mean(1,keepdims=True))/member['scale'])
    scores=sum(logits)
    if config.get('mask_unparsed',True):
        from common import parse_answer
        for i,row in enumerate(bank):
            for j,c in enumerate(row['candidates']):
                if parse_answer(c['text']) is None:scores[i,j]=-1e6
    return scores,config['decoder']


def expanded_predict(bank,x,checkpoint):
    scores,decoder=expanded_scores(bank,x,checkpoint)
    return decode(scores,bank,decoder)


def answer_group_features(bank,x,mode='g-full'):
    """One row per equivalent parsed answer, with no reference/ID features."""
    import re
    from common import parse_answer
    metadata=reasoning_features(bank,x,'metadata')
    cues=('total','each','per','left','remain','more','less','original','pay','paid','cost','profit',
          'earn','percent','discount','after','before','twice','half','average','ratio','difference',
          'how many','how much','how long','how far','speed','rate','hour','minute','day','week','year',
          'altogether','combined','rest','additional','increase','decrease','together','return',
          'sell','bought','buy','spend','spent','save','saved')
    features=[];mapping=[];group_ids=[]
    for i,row in enumerate(bank):
        groups={}
        for j,c in enumerate(row['candidates']):
            answer=parse_answer(c['text'])
            if answer is not None:groups.setdefault(answer,[]).append(j)
        if not groups:groups={('unparsed',j):[j] for j in range(4)}
        question=row['question'].lower()
        tail=re.split(r'[.!]\s+',question)[-1]
        context=np.asarray([np.log1p(t.count(c)) for t in (question,tail) for c in cues],dtype=np.float32)
        question_mean=x[i].mean(0)
        for ids in groups.values():
            meta=metadata[i,ids]
            summary=np.concatenate([meta.mean(0),meta.min(0),meta.max(0),meta.std(0),
                np.array([len(ids)/4.,len(groups)/4.,sum(len(a) for a in groups.values())/4.],dtype=np.float32)])
            if mode=='g-meta':f=summary
            elif mode=='g-context':f=np.concatenate([summary,context])
            else:
                emb=x[i,ids].mean(0)
                f=np.concatenate([emb,emb-question_mean,summary,context])
            features.append(f);mapping.append((i,ids));group_ids.append(i)
    return np.asarray(features,dtype=np.float32),mapping,np.asarray(group_ids,dtype=np.int64)


def numeric_process_trace(text):
    """Conservative arithmetic extraction. Unsupported prose/algebra is skipped.

    Only AST numeric constants and +,-,*,/ are evaluated; never eval text.
    This is evidence about written equations, not a symbolic solution oracle.
    """
    import ast
    import re
    import math
    normalized=text.replace('**','').replace('$','').replace(',', '')
    for old,new in ((r'\times','*'),(r'\cdot','*'),(r'\div','/'),('×','*'),('÷','/'),('−','-'),(r'\left',''),(r'\right',''),(r'\%','%')):
        normalized=normalized.replace(old,new)
    for _ in range(5):
        updated=re.sub(r'\\(?:dfrac|tfrac|frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}',r'((\1)/(\2))',normalized)
        if updated==normalized:break
        normalized=updated
    normalized=re.sub(r'(\d+(?:\.\d+)?)\s*%',r'(\1/100)',normalized)
    normalized=re.sub(r'\\boxed\s*\{([^{}]*)\}',r'\1',normalized)
    normalized=re.sub(r'\\(?:text|mathrm)\s*\{[^{}]*\}', ' UNIT ', normalized)
    expression=r'[\d.()+*/ \t-]+'
    def calculate(s):
        s=s.strip().rstrip('.')
        if not s or len(s)>200:return None
        try:
            tree=ast.parse(s,mode='eval');leaves=[];operators=[]
            if sum(1 for _ in ast.walk(tree))>80:return None
            def walk(node):
                if isinstance(node,ast.Constant) and type(node.value) in (int,float):
                    value=float(node.value)
                    if not math.isfinite(value) or abs(value)>1e15:raise ValueError()
                    leaves.append(value);return value
                if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
                    v=walk(node.operand);return -v if isinstance(node.op,ast.USub) else v
                if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
                    a=walk(node.left);b=walk(node.right)
                    operators.append(type(node.op).__name__)
                    v=a+b if isinstance(node.op,ast.Add) else a-b if isinstance(node.op,ast.Sub) else a*b if isinstance(node.op,ast.Mult) else a/b
                    if not math.isfinite(v) or abs(v)>1e15:raise ValueError()
                    return v
                raise ValueError()
            return walk(tree.body),leaves,operators
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError):return None
    equations=[]
    for line in normalized.splitlines():
        for match in re.finditer('=',line):
            left=re.search('('+expression+r')$',line[:match.start()])
            right=re.match('('+expression+r')',line[match.end():])
            if not left or not right:continue
            ls,rs=left.group(1).strip(),right.group(1).strip()
            # Avoid treating a numeric suffix of an algebraic product as a full LHS.
            prefix=line[:match.start()-len(left.group(1))].rstrip()
            if prefix and (prefix[-1].isalnum() or prefix[-1] in '^_') and ls and ls[0].isdigit():continue
            a=calculate(ls);b=calculate(rs)
            if a is None or b is None or not a[2]:continue
            delta=abs(a[0]-b[0])/max(1.,abs(a[0]),abs(b[0]))
            equations.append(dict(lhs=a[0],rhs=b[0],leaves=a[1],ops=a[2],error=delta,valid=delta<=1e-5,lhs_expression=ls,rhs_expression=rs))
    return equations,normalized


def process_features(bank):
    """Candidate-permutation equivariant input-only process descriptors."""
    import re
    from fractions import Fraction
    from common import parse_answer
    def key(v):return round(float(v),6)
    def numbers(t):
        return [float(v) for v in re.findall(r'(?<![\w.])\d+(?:\.\d+)?',t.replace(',',''))]
    def tokens(t):return {v.rstrip('s') for v in re.findall(r'[a-z]{3,}',t.lower())}
    constants={0.,1.,2.,3.,4.,7.,10.,12.,24.,30.,60.,100.,365.,.5,.25}
    cues=('remain','left','total','each','per','half','twice','average','ratio','percent',
          'after','before','more','less','difference','altogether','combined','rest',
          'hour','minute','day','week','month','year','cost','profit','earn','spend')
    all_rows=[]
    for row in bank:
        q=row['question'].lower();qnums=set(map(key,numbers(q)))
        for word,value in (('one',1),('two',2),('three',3),('four',4),('five',5),('six',6),('seven',7),('eight',8),('nine',9),('ten',10),('half',.5),('twice',2)):
            if re.search(r'\b'+word+r'\b',q):qnums.add(float(value))
        qtail=re.split(r'[.!]\s+',q)[-1];qt=tokens(qtail)
        traces=[numeric_process_trace(c['text']) for c in row['candidates']]
        answers=[parse_answer(c['text']) for c in row['candidates']]
        answers=[float(Fraction(a)) if a is not None else None for a in answers]
        raw=[]
        for j,(eq,text) in enumerate(traces):
            low=text.lower();ans=answers[j];ak=key(ans) if ans is not None else None
            leaves=[key(v) for e in eq for v in e['leaves']];rhs=[key(e['rhs']) for e in eq]
            leafset=set(leaves);rhset=set(rhs);tn=set(map(key,numbers(text)))
            errs=[e['error'] for e in eq];correct=sum(e['valid'] for e in eq)
            coverage=lambda values:len(qnums&values)/max(len(qnums),1)
            f=[np.log1p(len(eq)),np.log1p(correct),np.log1p(len(eq)-correct),
               (len(eq)-correct)/max(len(eq),1),max(errs,default=0.),np.mean(errs) if errs else 0.,
               float(ak in rhset),float(bool(rhs) and ak==rhs[-1]),float(ak in leafset),
               sum(v==ak for v in rhs)/max(len(rhs),1),coverage(tn),coverage(leafset),
               np.log1p(len(qnums-leafset)),np.log1p(len(qnums-tn)),np.log1p(len(leafset)),np.log1p(len(rhset)),
               np.log1p(len(leafset-qnums-constants)),np.log1p(len(rhset-qnums)),
               float(ak in qnums),np.log1p(len(qnums)),
               max((len(e['ops']) for e in eq),default=0),np.mean([len(e['ops']) for e in eq]) if eq else 0.]
            for op in ('Add','Sub','Mult','Div'):
                f.extend([sum(e['ops'].count(op) for e in eq),float(bool(eq) and op in eq[-1]['ops'])])
            available=qnums|constants;unexplained=0;depths={v:0 for v in available};provenance={v:{v} for v in qnums}
            for e in eq:
                ls=set(map(key,e['leaves']));v=key(e['rhs'])
                unexplained+=len(ls-available)
                newdepth=1+max((depths.get(k,0) for k in ls),default=0)
                newprov=set().union(*(provenance.get(k,set()) for k in ls))
                depths[v]=max(depths.get(v,0),newdepth);provenance[v]=provenance.get(v,set())|newprov;available.add(v)
            f.extend([np.log1p(unexplained),depths.get(ak,0),max(depths.values(),default=0),coverage(provenance.get(ak,set()))])
            first=next((i for i,v in enumerate(rhs) if v==ak),len(rhs))
            f.extend([max(0,len(rhs)-1-first),sum(ak in set(map(key,e['leaves'])) for e in eq[first+1:])])
            others=[k for k in range(4) if k!=j]
            f.extend([sum(ak in {key(e['rhs']) for e in traces[k][0]} for k in others),
                      sum(ak in {key(v) for e in traces[k][0] for v in e['leaves']} for k in others),
                      sum(answers[k] is not None and key(answers[k]) in rhset for k in others),
                      sum(answers[k] is not None and key(answers[k]) in leafset for k in others)])
            sentences=[s for s in re.split(r'[\n.!?]+',low) if len(tokens(s))>=3]
            tail=' '.join(sentences[-3:]);fulltokens=tokens(low)
            f.extend([len(qt&tokens(tail))/max(len(qt),1),len(qt&fulltokens)/max(len(qt),1)])
            for cue in cues:
                f.extend([float(cue in qtail and cue in tail),float(cue in q and cue not in low)])
            for a,b,constant in (('hour','minute',60),('week','day',7),('year','month',12),('percent','',100)):
                active=a in q and (not b or b in q)
                f.extend([float(active),float(active and constant in leafset)])
            raw.append(f)
        raw=np.asarray(raw,dtype=np.float32)
        # Absolute and centered descriptors keep candidate order out of learned inputs.
        all_rows.append(np.concatenate([raw,raw-raw.mean(0,keepdims=True)],axis=1))
    return np.asarray(all_rows,dtype=np.float32)


def arithmetic_repair_traces(row):
    """Recompute stated equations, without inventing a solution or output answer.

    Numeric substitution is ambiguous when distinct quantities share a value.
    Local-only, protected and permissive traces are therefore exposed as evidence.
    """
    import ast
    import re
    from fractions import Fraction
    from common import parse_answer
    def key(v):return round(float(v),6)
    def close(a,b):return abs(a-b)<=1e-5*max(1.,abs(a),abs(b))
    question_numbers={key(float(v)) for v in re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?',row['question'].replace(',',''))}
    protected=question_numbers|{0.,1.,2.,3.,4.,7.,10.,12.,24.,30.,60.,100.,365.,.5,.25}
    def calculate(expression,replacements):
        tree=ast.parse(expression.strip().rstrip('.'),mode='eval')
        def walk(n):
            if isinstance(n,ast.Constant) and type(n.value) in (float,int):return replacements.get(key(n.value),float(n.value))
            if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
                value=walk(n.operand);return -value if isinstance(n.op,ast.USub) else value
            if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
                a=walk(n.left);b=walk(n.right)
                return a+b if isinstance(n.op,ast.Add) else a-b if isinstance(n.op,ast.Sub) else a*b if isinstance(n.op,ast.Mult) else a/b
            raise ValueError('Unsupported arithmetic')
        value=walk(tree.body)
        if not np.isfinite(value) or abs(value)>1e15:raise ValueError('Unbounded value')
        return value
    result=[]
    for c in row['candidates']:
        clean=re.sub(r'^\s*(?:[-•]|\d+[.)])\s+', '', c['text'],flags=re.MULTILINE)
        eq,_=numeric_process_trace(clean)
        parsed=parse_answer(c['text']);answer=float(Fraction(parsed)) if parsed is not None else None
        modes=[]
        for mode in ('local','protected','permissive'):
            corrections={};outputs=[];changed=propagated=failures=0
            for e in eq:
                try:value=calculate(e['lhs_expression'],corrections if mode!='local' else {})
                except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError):
                    value=e['lhs'];failures+=1
                changed+=not close(value,e['rhs']);propagated+=not close(value,e['lhs'])
                outputs.append(value);rk=key(e['rhs'])
                if mode=='permissive' or (mode=='protected' and rk not in protected):
                    if close(value,e['rhs']):corrections.pop(rk,None)
                    else:corrections[rk]=value
            anchors=[i for i,e in enumerate(eq) if answer is not None and close(answer,e['rhs'])]
            proposal=outputs[anchors[-1]] if anchors else answer
            modes.append(dict(proposal=proposal,anchored=bool(anchors),changed=changed,propagated=propagated,
                              failures=failures,outputs=outputs))
        result.append(dict(answer=answer,equations=eq,modes=modes))
    return result


def repair_features(bank):
    """Input-only corrected-value agreement; never a hard candidate override."""
    def close(a,b):return a is not None and b is not None and abs(a-b)<=1e-5*max(1.,abs(a),abs(b))
    def distance(a,b):return abs(a-b)/max(1.,abs(a),abs(b)) if a is not None and b is not None else 1.
    rows=[]
    for row in bank:
        traces=arithmetic_repair_traces(row);raw=[]
        for j,t in enumerate(traces):
            ans=t['answer'];eq=t['equations'];n=len(eq);bad=sum(not e['valid'] for e in eq)
            f=[np.log1p(n),np.log1p(bad),bad/max(n,1),max((e['error'] for e in eq),default=0.)]
            for mode in range(3):
                own=t['modes'][mode];proposal=own['proposal'];others=[x['modes'][mode] for x in traces]
                valid=[x['proposal'] for x in others if x['proposal'] is not None]
                original_votes=sum(close(ans,x['answer']) for x in traces)
                votes=sum(close(ans,x['proposal']) for x in others)
                corrected_votes=sum(close(ans,x['modes'][mode]['proposal']) and not close(x['answer'],x['modes'][mode]['proposal']) for x in traces)
                f.extend([float(own['anchored']),float(close(ans,proposal)),distance(ans,proposal),
                          np.log1p(own['changed']),np.log1p(own['propagated']),own['changed']/max(n,1),own['failures'],
                          votes/4,corrected_votes/4,(votes-original_votes)/4,
                          sum(close(ans,x['proposal']) and x['anchored'] for x in others)/4,
                          sum(close(proposal,x['answer']) for x in traces)/4,
                          sum(close(ans,v) for v in own['outputs'])/max(n,1),
                          sum(any(close(ans,v) for v in x['outputs']) for x in others)/4,
                          min((distance(ans,v) for v in valid),default=1.),
                          np.mean([distance(ans,v) for v in valid]) if valid else 1.,
                          float(bool(own['outputs']) and close(ans,own['outputs'][-1])),
                          sum(x['anchored'] for x in others)/4])
            raw.append(f)
        raw=np.asarray(raw,dtype=np.float32);rows.append(np.concatenate([raw,raw-raw.mean(0,keepdims=True)],axis=1))
    return np.asarray(rows,dtype=np.float32)
