#!/usr/bin/env python3
"""The stereotype test across cohorts and cultures. Same amplitude statistic as trait_amplitude.py,
but with the five domain scores every cohort has, so cohorts are comparable:
  ESCS (US; IPIP-NEO domains; 109 media items; 8 models, facet condition, people's 5 domain scores as predictors),
  Twin (US; BFI-44; policy and risk/benefit), Harmonies (Germany; BFI-44; 23 music genres; couples), Indonesia (TIPI; 32 story features).
Amplitude ratio = sqrt(sum_items T_model / sum_items T_human), T = adjusted R^2 (OLS on 5 domains) x variance across people:
how far personality spreads people apart in the model's predictions relative to how far it spreads real people.
Direction r = mean over items of corr(OLS-fitted model component, OLS-fitted human component).
95% interval from a delete-half jackknife over people (couples kept whole in Harmonies; 300 half-samples drawn without
replacement; SE = SD of the half-sample estimates, Shao & Wu 1989). A with-replacement bootstrap is NOT used: duplicated rows
distort the adjusted R^2 the ratio is built on (first run, Oct 9: intervals that excluded their own estimates). -> analysis/amplitude_cohorts.json"""
import json, os
import numpy as np, pandas as pd
PKG = "<DATA_ROOT>/03_OCTOBER_EXPANSION"
EA = f"{PKG}/expansion_execution/external_additions/analysis"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis")
rng = np.random.default_rng(20261009)
DOM = ["E", "A", "C", "N", "O"]

def T(X, y):
    A = np.column_stack([np.ones(len(y)), X]); b, *_ = np.linalg.lstsq(A, y, rcond=None); f = A @ b
    if np.sum((y - y.mean()) ** 2) == 0: return 0.0, f          # constant outcome: personality explains none of it (was 0/0 -> NaN; the original analysis, Oct 11)
    r2 = 1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2); n, p = len(y), X.shape[1]
    return (1 - (1 - r2) * (n - 1) / (n - p - 1)) * np.var(y, ddof=1), f

def amp(X, Y, P, rows):
    th = tm = 0.0; dirs = []
    for j in range(Y.shape[1]):
        m = rows[~np.isnan(Y[rows, j]) & ~np.isnan(P[rows, j])]
        if len(m) < 40: continue
        a, fa = T(X[m], Y[m, j]); b, fb = T(X[m], P[m, j]); th += max(a, 0); tm += max(b, 0)
        if np.std(fa) > 0 and np.std(fb) > 0: dirs.append(np.corrcoef(fa, fb)[0, 1])
    return np.sqrt(tm / th) if th > 0 else np.nan, float(np.mean(dirs))

def run(name, X, Y, preds, groups=None):
    out = {}; n = len(Y); units = np.unique(groups) if groups is not None else np.arange(n)
    for k, P in preds.items():
        est, dr = amp(X, Y, P, np.arange(n)); halves = []
        for _ in range(300):
            s = rng.choice(units, len(units) // 2, replace=False)
            rows = np.concatenate([np.where(groups == u)[0] for u in s]) if groups is not None else s
            halves.append(amp(X, Y, P, rows)[0])
        se = float(np.nanstd(halves)); lo, hi = est - 1.96 * se, est + 1.96 * se
        out[k] = {"amplitude_ratio": float(est), "ci95": [float(lo), float(hi)], "direction_r": dr, "n_undefined_halves": int(np.sum(np.isnan(halves)))}   # undefined = no personality-linked variance in people's answers
        print(f"{name:22s} {k:22s} amplitude x{est:.2f} [{lo:.2f}, {hi:.2f}]  direction r {dr:.2f}", flush=True)
    return out

res = {}
# --- ESCS (5 domains as predictors)
P = pd.read_csv(f"{PKG}/offline_reproduction_candidate/data/core/MEDIA469_ITEM_PREDICTIONS.csv")
Xe = pd.read_csv("<DATA_ROOT>/05_ORIGINAL_INPUTS_AND_DEPENDENCIES/escs_data/ipip_neo_scored.csv").set_index("id")
dcols = ["E_domain", "A_domain", "C_domain", "N_domain", "O_domain"]
sub = P[P.condition == "C-fac"]
truth = sub[sub.model == "flash"].pivot(index="escs_id", columns="item", values="truth")
ids = truth.index; Xm = Xe.loc[ids, dcols].values
preds = {m: sub[sub.model == m].pivot(index="escs_id", columns="item", values="prediction").loc[ids, truth.columns].values for m in sorted(sub.model.unique())}
res["ESCS media (US)"] = run("ESCS media (US)", Xm, truth.values, preds)
# --- Twin + Harmonies
people = {p["id"]: p for p in json.load(open(f"{EA}/twin_harmonies/PEOPLE.json"))}
E = np.load(f"{PKG}/offline_reproduction_candidate/data/expanded_predictions.npz"); R = np.load(f"{PKG}/offline_reproduction_candidate/data/expanded_reference.npz")
for task, label in [("twin__policy_support", "Twin policy (US)"), ("twin__risk_benefit", "Twin risk (US)"), ("harmonies__genres", "Harmonies music (DE)")]:
    tid = list(R[f"{task}__ids"]); X = np.array([[people[i]["domains"][d] for d in DOM] for i in tid]); Y = R[f"{task}__truth"]
    pr = {k.split("__", 2)[2]: E[k] for k in E.keys() if k.startswith(task + "__") and k.split("__")[-1] in ("numeric", "structured", "portrait")}
    assert list(E[f"{task}__ids"]) == tid
    res[label] = run(label, X, Y, pr, groups=np.array(R[f"{task}__groups"]) if task.startswith("harmonies") else None)
# --- Indonesia
Z = np.load(f"{EA}/PRIVATE_PREDICTIONS.npz", allow_pickle=True)
ind = {p["id"]: p for p in json.load(open(f"{PKG}/expansion_execution/external_additions/indonesia/PEOPLE.json"))}
iid = list(Z["indonesia__stories__ids"]); X = np.array([[ind[i]["domains"][d] for d in DOM] for i in iid]); Y = Z["indonesia__stories__truth"]
pr = {k.split("__", 2)[2]: Z[k] for k in Z.keys() if k.startswith("indonesia__stories__") and k.split("__")[-1] in ("numeric", "structured", "portrait")}
res["Indonesia stories (ID)"] = run("Indonesia stories (ID)", X, Y, pr)
json.dump(res, open(os.path.join(OUT, "amplitude_cohorts.json"), "w"), indent=1)
