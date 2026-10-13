"""Manuscript-only analysis; reads frozen outputs and never edits experiment sources.

Normalize unambiguous item-label keys and use each person's common unshown-item
mask across A arms. No new model calls. Persist only aggregate results.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
EXP = Path('<PROJECT_ROOT>/experiments_rescue')
spec = importlib.util.spec_from_file_location('C', EXP / 'common.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
sources = ['common.py', 'RESULTS.md', 'expA/selections.json', 'expA/build_expA.py',
           'expA/run_expA.py', 'expA/analyze_A.py', 'expA/RESULTS_A.json',
           'expB/RESULTS_B.json', 'expB/score_ladder.py']
selections = {int(k): v for k, v in json.loads((EXP / 'expA/selections.json').read_text()).items()}
ev = {p: v['eval_items'] for p, v in selections.items() if 'eval_items' in v}
expected = set(C.TRUTH.columns)
results = {}
for model in ['flash', 'qwen37', 'minimax']:
    path = f'expA/out/{model}.jsonl'
    sources.append(path)
    rows = [json.loads(line) for line in (EXP / path).read_text().splitlines()]
    arms = {a: {} for a in ['A0', 'A1', 'A2', 'A3', 'A4']}
    normalized = 0
    for row in rows:
        if not row['ok']:
            continue
        pred = {}
        changed = False
        for key, value in row['pred'].items():
            new = key if key in expected else key.split(':', 1)[0]
            assert new in expected and new not in pred
            assert 1 <= value <= 7
            pred[new] = value
            changed |= new != key
        normalized += int(changed)
        arms[row['arm']][row['pid']] = pred
    matrices = {a: pd.DataFrame.from_dict(v, orient='index').reindex(columns=C.TRUTH.columns).astype(float)
                for a, v in arms.items()}
    people = sorted(set.intersection(*(set(m.index) for m in matrices.values())))
    common = {p: [it for it in ev[p] if all(pd.notna(m.loc[p, it]) for m in matrices.values())] for p in people}
    dist = {a: C.distinctive(m.loc[people], common) for a, m in matrices.items()}
    out = {'n': len(people), 'normalized_label_records': normalized,
           'evaluation_item_count': {'min': min(map(len, common.values())), 'median': float(np.median(list(map(len, common.values()))))},
           'removed_eval_cells': sum(len(ev[p])-len(common[p]) for p in people)}
    for a, m in matrices.items():
        masked = m.loc[people].copy()
        for p in people:
            masked.loc[p, [it for it in masked.columns if it not in common[p]]] = np.nan
        amplitude, direction = C.amplitude(masked)
        out[a] = {'distinctive': C.boot_mean(dist[a]), 'amplitude': amplitude, 'direction': direction}
    for a, b in [('A1','A0'), ('A1','A2'), ('A1','A3'), ('A4','A0'), ('A4','A1'), ('A2','A0'), ('A3','A0')]:
        delta = (dist[a] - dist[b]).dropna()
        rng = np.random.default_rng(1)
        boot = [rng.choice(delta.values, len(delta)).mean() for _ in range(2000)]
        t = stats.ttest_1samp(delta.values, 0.0)
        out[f'{a}-{b}'] = {'mean': float(delta.mean()), 'ci95': list(np.percentile(boot,[2.5,97.5])), 'n': len(delta), 't': float(t.statistic), 'p': float(t.pvalue)}
    results[model] = out
    print(model, json.dumps(out), flush=True)

ready = json.loads((HERE / 'EXPERIMENT_A_READY.json').read_text())
for m in results:
    for k, v in results[m].items():
        if '-' in k and isinstance(v, dict) and 'p' in v:
            r = ready[m][k]
            assert abs(r['mean'] - v['mean']) < 1e-12 and np.allclose(r['ci95'], v['ci95'], atol=1e-12), (m, k)
keys = [(m, k) for m in results for k in results[m] if '-' in k and isinstance(results[m][k], dict) and 'p' in results[m][k]]
ps = np.array([results[m][k]['p'] for m, k in keys]); order = np.argsort(ps); n = len(ps)
adj = np.empty(n); run = 0.0
for rank, idx in enumerate(order):
    run = max(run, min(1.0, (n - rank) * ps[idx])); adj[idx] = run
paired = {m: {} for m in results}
for (m, k), a in zip(keys, adj):
    v = results[m][k]; paired[m][k] = {'mean': v['mean'], 'ci95': v['ci95'], 'n': v['n'], 'p': v['p'], 'p_holm': float(a)}
(HERE / 'EXPERIMENT_A_PAIRED.json').write_text(json.dumps({'family_size': n, 'test': 'paired t-test on per-person differences; Holm across all contrasts and models', 'contrasts': paired}, indent=2) + '\n')
print('family', n)
for (m, k), a in zip(keys, adj): print(m, k, '%.4f' % results[m][k]['mean'], 'p=%.2g' % results[m][k]['p'], 'holm=%.2g' % a)
