#!/usr/bin/env python3
"""Amplitude of the human-trained ridge: the reference point for the models' amplitudes.
The ridge predicts each person's answers from the same personality scores, fitted on OTHER people's real answers (ESCS: 267
participants outside the evaluated sample, 35 trait scores; Twin and Harmonies: the original participant-disjoint / couple-grouped
ridge). If the amplitude statistic behaves, a predictor fitted to real people should land near 1.
Statistic, items, people and jackknife are exactly those of the model amplitudes:
  ESCS 35 scores  -> as trait_amplitude.py   (OLS on the 35 IPIP-NEO scores; 200 delete-half samples over people)
  five domains    -> as amplitude_cohorts.py (OLS on the five domains; 300 delete-half samples; whole couples in Harmonies)
The ESCS ridge file covers all 469 people; we keep the 467 scored ones (the models' file), as everywhere else.
Indonesia: the original ridge from the 649-person reference (indonesia_reference.npz; added Oct 11 after the review merge,).
Saved predictions only; no new generation. -> analysis/ridge_amplitude.json"""
import json, os
import numpy as np, pandas as pd
ROOT = "<DATA_ROOT>"
D = f"{ROOT}/03_OCTOBER_EXPANSION/offline_reproduction_candidate/data"
EA = f"{ROOT}/03_OCTOBER_EXPANSION/expansion_execution/external_additions/analysis"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis")
DOM = ["E", "A", "C", "N", "O"]

def T(X, y):
    A = np.column_stack([np.ones(len(y)), X]); b, *_ = np.linalg.lstsq(A, y, rcond=None); f = A @ b
    if np.sum((y - y.mean()) ** 2) == 0: return 0.0, f          # constant outcome: personality explains none of it (was 0/0 -> NaN; the original analysis, Oct 11)
    r2 = 1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2); n, p = len(y), X.shape[1]
    return (1 - (1 - r2) * (n - 1) / (n - p - 1)) * np.var(y, ddof=1), f

def amp(X, Y, P, rows, min_n):
    th = tm = 0.0; dirs = []
    for j in range(Y.shape[1]):
        m = rows[~np.isnan(Y[rows, j]) & ~np.isnan(P[rows, j])]
        if len(m) < min_n: continue
        a, fa = T(X[m], Y[m, j]); b, fb = T(X[m], P[m, j]); th += max(a, 0); tm += max(b, 0)
        if np.std(fa) > 0 and np.std(fb) > 0: dirs.append(np.corrcoef(fa, fb)[0, 1])
    return (np.sqrt(tm / th) if th > 0 else np.nan), float(np.mean(dirs))

def run(name, X, Y, P, n_half, min_n, groups=None, seed=20261010):
    rng = np.random.default_rng(seed); n = len(Y); units = np.unique(groups) if groups is not None else np.arange(n)
    est, dr = amp(X, Y, P, np.arange(n), min_n); halves = []
    for _ in range(n_half):
        s = rng.choice(units, len(units) // 2, replace=False)
        rows = np.concatenate([np.where(groups == u)[0] for u in s]) if groups is not None else s
        halves.append(amp(X, Y, P, rows, min_n)[0])
    se = float(np.nanstd(halves)); lo, hi = est - 1.96 * se, est + 1.96 * se
    print(f"{name:34s} ridge amplitude x{est:.2f} [{lo:.2f}, {hi:.2f}]  direction r {dr:.2f}  (n {n})", flush=True)
    return {"amplitude_ratio": float(est), "ci95": [float(lo), float(hi)], "direction_r": dr, "n_people": int(n), "n_undefined_halves": int(np.sum(np.isnan(halves)))}

res = {}
# --- ESCS
M = pd.read_csv(f"{D}/core/MEDIA469_ITEM_PREDICTIONS.csv")
truth = M[(M.model == "flash") & (M.condition == "C-fac")].pivot(index="escs_id", columns="item", values="truth")
ids, items = truth.index, truth.columns
Rg = pd.read_csv(f"{D}/core/MEDIA469_ECOSYSTEM/HUMAN_RIDGE_PREDICTIONS_LONG.csv")
rid = Rg.pivot(index="escs_id", columns="item", values="prediction").reindex(index=ids, columns=items)
rt = Rg.pivot(index="escs_id", columns="item", values="truth").reindex(index=ids, columns=items)
assert np.allclose(rt.values, truth.values, equal_nan=True), "ridge file's truth differs from the models' truth"
Xs = pd.read_csv(f"{ROOT}/05_ORIGINAL_INPUTS_AND_DEPENDENCIES/escs_data/ipip_neo_scored.csv").set_index("id")
C35 = [c for c in Xs.columns if c != "id"]; assert len(C35) == 35, len(C35)
dcols = ["E_domain", "A_domain", "C_domain", "N_domain", "O_domain"]
# trait_amplitude.py keeps items with >= 100 people and drops a person per item only if missing; same here
res["ESCS media (US) | 35 trait scores"] = run("ESCS media (US) | 35 trait scores", Xs.loc[ids, C35].values, truth.values, rid.values, 200, 100)
res["ESCS media (US) | five domains"] = run("ESCS media (US) | five domains", Xs.loc[ids, dcols].values, truth.values, rid.values, 300, 40)
# --- Twin + Harmonies (the original ridge in expanded_reference.npz)
people = {p["id"]: p for p in json.load(open(f"{EA}/twin_harmonies/PEOPLE.json"))}
R = np.load(f"{D}/expanded_reference.npz", allow_pickle=True)
for task, label in [("twin__policy_support", "Twin policy (US)"), ("twin__risk_benefit", "Twin risk (US)"), ("harmonies__genres", "Harmonies music (DE)")]:
    tid = list(R[f"{task}__ids"]); X = np.array([[people[i]["domains"][d] for d in DOM] for i in tid])
    g = np.array(R[f"{task}__groups"]) if task.startswith("harmonies") else None
    res[label] = run(label, X, R[f"{task}__truth"].astype(float), R[f"{task}__ridge"].astype(float), 300, 40, groups=g)
# --- Indonesia (TIPI five domains; the original ridge, 649 reference people outside the 150)
Zi = np.load(f"{D}/indonesia_reference.npz", allow_pickle=True)
ind = {p["id"]: p for p in json.load(open(f"{ROOT}/03_OCTOBER_EXPANSION/expansion_execution/external_additions/indonesia/PEOPLE.json"))}
Zp = np.load(f"{EA}/PRIVATE_PREDICTIONS.npz", allow_pickle=True)
iid = list(Zi["ids"]); assert iid == list(Zp["indonesia__stories__ids"]), "Indonesia id order differs"
assert np.allclose(Zi["truth"].astype(float), Zp["indonesia__stories__truth"].astype(float), equal_nan=True), "Indonesia truth differs"
X = np.array([[ind[i]["domains"][d] for d in DOM] for i in iid])
res["Indonesia stories (ID)"] = run("Indonesia stories (ID)", X, Zi["truth"].astype(float), Zi["ridge"].astype(float), 300, 40)
json.dump(res, open(os.path.join(OUT, "ridge_amplitude.json"), "w"), indent=1)
