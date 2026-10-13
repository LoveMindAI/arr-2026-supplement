#!/usr/bin/env python3
"""The ladder: distinctive accuracy of every predictor in every cohort, recomputed from the original saved prediction and reference files
(an independent check on the v3 tables, and the data behind the ladder figure). Distinctive accuracy = mean over people of the
within-person Spearman of (prediction - norm) vs (truth - norm); norm = the participant-disjoint population profile the original analysis saved.
Rungs: population norm (0 by construction), LLM cells (model x form), human-trained ridge, and where the data exist, the partner's real
answers (Harmonies) and the person's own earlier answers (Twin). 95% CIs: 2,000 person bootstraps (whole couples in Harmonies).
Wrong-person: each prediction scored against every other person's truth, averaged. -> analysis/ladder.json"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
PKG = "<DATA_ROOT>/03_OCTOBER_EXPANSION"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis")
rng = np.random.default_rng(20261009)

def rr(A):
    R = np.full(A.shape, np.nan)
    for i in range(len(A)):
        m = ~np.isnan(A[i])
        if m.sum() >= 3 and np.nanstd(A[i]) > 0: R[i, m] = rankdata(A[i, m])
    return R
def pairwise(P, Y, N):
    """C[i, j] = Spearman(P_i - N_i, Y_j - N_j) over items both observe (complete targets assumed per row)."""
    A, B = P - N, Y - N
    C = np.full((len(A), len(B)), np.nan)
    RA = rr(A); RB = rr(B)
    for i in range(len(A)):
        if np.isnan(RA[i]).all(): continue
        for j in range(len(B)):
            m = ~np.isnan(A[i]) & ~np.isnan(B[j])
            if m.sum() < 3: continue
            a = rankdata(A[i, m]); b = rankdata(B[j, m])
            if a.std() == 0 or b.std() == 0: continue
            C[i, j] = np.corrcoef(a, b)[0, 1]
    return C
def own(P, Y, N):
    v = []
    for i in range(len(P)):
        m = ~np.isnan(P[i]) & ~np.isnan(Y[i])
        a, b = P[i, m] - N[i, m] if N.ndim == 2 else P[i, m] - N[m], Y[i, m] - (N[i, m] if N.ndim == 2 else N[m])
        v.append(np.corrcoef(rankdata(a), rankdata(b))[0, 1] if m.sum() >= 3 and np.std(a) > 0 and np.std(b) > 0 else np.nan)
    return np.array(v)
def ci(v, groups=None):
    v = np.asarray(v, float)
    if groups is None:
        ok = v[~np.isnan(v)]; bs = [np.mean(rng.choice(ok, len(ok))) for _ in range(2000)]
    else:
        g = np.asarray(groups); units = np.unique(g); rows = {u: np.where(g == u)[0] for u in units}
        bs = [np.nanmean(np.concatenate([v[rows[u]] for u in rng.choice(units, len(units))])) for _ in range(2000)]
    return {"mean": float(np.nanmean(v)), "ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "n": int(np.sum(~np.isnan(v)))}

res = {}
# ---------------- ESCS media (8 models x 3 conditions; ridge; disjoint norm)
P = pd.read_csv(f"{PKG}/offline_reproduction_candidate/data/core/MEDIA469_ITEM_PREDICTIONS.csv")
Rg = pd.read_csv(f"{PKG}/offline_reproduction_candidate/data/core/MEDIA469_ECOSYSTEM/HUMAN_RIDGE_PREDICTIONS_LONG.csv")
truth = P[(P.model == "flash") & (P.condition == "C-fac")].pivot(index="escs_id", columns="item", values="truth")
norm = P[(P.model == "flash") & (P.condition == "C-fac")].pivot(index="escs_id", columns="item", values="disjoint_norm")
ids, items = truth.index, truth.columns
Y, N = truth.values, norm.values
out = {"n_people": len(ids), "n_items": len(items), "cells": {}}
for (m, c), g in P.groupby(["model", "condition"]):
    M = g.pivot(index="escs_id", columns="item", values="prediction").reindex(index=ids, columns=items).values
    out["cells"][f"{m}|{c}"] = ci(own(M, Y, N))
rid = Rg.pivot(index="escs_id", columns="item", values="prediction").reindex(index=[i for i in ids if i in set(Rg.escs_id)], columns=items)
ki = [list(ids).index(i) for i in rid.index]
out["ridge"] = ci(own(rid.values, Y[ki], N[ki]))
res["ESCS media (US)"] = out

# ---------------- Twin, Harmonies (expanded files); Indonesia (October additions)
E = np.load(f"{PKG}/offline_reproduction_candidate/data/expanded_predictions.npz", allow_pickle=True)
R = np.load(f"{PKG}/offline_reproduction_candidate/data/expanded_reference.npz", allow_pickle=True)
for task, label in [("twin__policy_support", "Twin policy (US)"), ("twin__risk_benefit", "Twin risk/benefit (US)"), ("harmonies__genres", "Harmonies music (DE)"), ("indonesia__stories", "Indonesia stories (ID)")]:
    if task.startswith("indonesia"):
        Y = E[f"{task}__truth"].astype(float); N = E[f"{task}__norm"].astype(float); groups = None; ref = None
    else:
        Y = R[f"{task}__truth"].astype(float); N = R[f"{task}__population"].astype(float); ref = R
        assert list(E[f"{task}__ids"]) == list(R[f"{task}__ids"])
        groups = np.array(R[f"{task}__groups"]) if task.startswith("harmonies") else None
    N2 = N if N.ndim == 2 else np.tile(N, (len(Y), 1))
    out = {"n_people": len(Y), "n_items": Y.shape[1], "cells": {}}
    for m in ["sol", "glm", "flash"]:
        for c in ["numeric", "structured", "portrait"]:
            M = E[f"{task}__{m}__{c}"].astype(float); C = pairwise(M, Y, N2)
            o = np.diag(C); w = np.array([np.nanmean(np.delete(C[i], i)) for i in range(len(C))])
            out["cells"][f"{m}|{c}"] = {**ci(o, groups), "wrong_person": float(np.nanmean(w)), "gap": ci(o - w, groups)}
    if task.startswith("indonesia"):   # the original Indonesian ridge (649 reference people outside the 150); added Oct 11 ()
        Zi = np.load(f"{PKG}/offline_reproduction_candidate/data/indonesia_reference.npz", allow_pickle=True)
        assert list(Zi["ids"]) == list(E[f"{task}__ids"]) and np.allclose(Zi["truth"].astype(float), Y, equal_nan=True)
        out["ridge"] = ci(own(Zi["ridge"].astype(float), Y, N2), groups)
    if ref is not None:
        out["ridge"] = ci(own(R[f"{task}__ridge"].astype(float), Y, N2), groups)
        if f"{task}__earlier_self" in R.files: out["earlier_self"] = ci(own(R[f"{task}__earlier_self"].astype(float), Y, N2), groups)
        if task.startswith("harmonies"):
            g = list(groups); partner = [[j for j in range(len(g)) if g[j] == g[i] and j != i][0] for i in range(len(g))]
            out["partner_real"] = ci(own(Y[partner], Y, N2), groups)
    res[label] = out
json.dump(res, open(os.path.join(OUT, "ladder.json"), "w"), indent=1)
f = lambda d: f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
for k, v in res.items():
    cells = [d["mean"] for d in v["cells"].values()]
    print(f"{k:24s} n={v['n_people']:3d} LLM cells {min(cells):+.3f} to {max(cells):+.3f}" + "".join(f" | {r} {f(v[r])}" for r in ["ridge", "earlier_self", "partner_real"] if r in v))
