#!/usr/bin/env python3
"""Exp A analysis. Per model and arm, on each person's EVALUATION items only (never shown in any arm): distinctive accuracy
(paired arm contrasts, bootstrap over people), amplitude and flatness on items, and wrong-person reassignment of A1.
Reading rule (fixed before results): the model 'routes toward the person' if A1 beats A0, A2 and A3 on
distinctive accuracy (paired 95% intervals above 0) AND A1's amplitude is below A0's. -> expA/RESULTS_A.json"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as C
HERE = os.path.dirname(os.path.abspath(__file__))
sel = {int(k): v for k, v in json.load(open(f"{HERE}/selections.json")).items()}
EV = {p: v["eval_items"] for p, v in sel.items() if "eval_items" in v}
res = {}
def paired(a, b, B=2000, seed=1):
    j = pd.concat([a, b], axis=1).dropna(); d = (j.iloc[:, 0] - j.iloc[:, 1]).values; rng = np.random.default_rng(seed)
    m = [rng.choice(d, len(d)).mean() for _ in range(B)]; return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int(len(d))]
for m in sys.argv[1:] or ["flash", "qwen37", "minimax"]:
    f = f"{HERE}/out/{m}.jsonl"
    if not os.path.exists(f): continue
    recs = [json.loads(l) for l in open(f)]; recs = [r for r in recs if r["ok"]]
    arms = {}
    for arm in ("A0", "A1", "A2", "A3", "A4"):
        d = {r["pid"]: r["pred"] for r in recs if r["arm"] == arm}
        if d: arms[arm] = pd.DataFrame.from_dict(d, orient="index").astype(float)
    common_p = sorted(set.intersection(*[set(a.index) for a in arms.values()]))
    dist = {arm: C.distinctive(a.loc[common_p], EV) for arm, a in arms.items()}
    # amplitude on the items that are evaluation items for most people: use, per item, only people for whom it is an eval item
    def amp_eval(a):
        b = a.loc[common_p].copy()
        for p in b.index: b.loc[p, [c for c in b.columns if c not in EV.get(p, [])]] = np.nan
        return C.jackknife_amp(b, B=int(os.environ.get("JK", 100)))
    out = {"n_people": len(common_p)}
    for arm in arms:
        out[arm] = {"distinctive": C.boot_mean(dist[arm]), "amp": amp_eval(arms[arm]), "flat": C.flatness(arms[arm].loc[common_p])}
    for a, b in [("A1", "A0"), ("A1", "A2"), ("A1", "A3"), ("A4", "A0"), ("A4", "A1"), ("A2", "A0"), ("A3", "A0")]:
        if a in dist and b in dist: out[f"{a}-{b}"] = paired(dist[a], dist[b])
    # wrong-person reassignment of A1 (fixed circular map over sorted ids), scored on the WRONG person's eval items
    if "A1" in arms:
        pp = common_p; wmap = {p: pp[(k + 1) % len(pp)] for k, p in enumerate(pp)}
        w = arms["A1"].loc[pp].copy(); w.index = [wmap[p] for p in pp]
        out["A1_wrongperson"] = C.boot_mean(C.distinctive(w, EV)); out["A1-A1wrong"] = paired(dist["A1"], C.distinctive(w, EV))
    res[m] = out
    print(m, json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv[0], 3) if isinstance(vv, (list, tuple)) else round(vv, 3)) for kk, vv in v.items()}) for k, v in out.items()}))
json.dump(res, open(f"{HERE}/RESULTS_A.json", "w"), indent=1)
