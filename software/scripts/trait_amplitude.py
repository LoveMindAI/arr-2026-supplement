#!/usr/bin/env python3
"""How far apart does personality push people's tastes, in real people vs in model predictions?
In the facet condition the model's only input is the 35 trait scores, so a high R^2 of its predictions on traits is expected by
construction. The meaningful comparison is AMPLITUDE: the variance across people that traits account for.
Per item: T = adjusted R^2 (OLS on the 35 scores) x variance across people. Adjusted R^2 x variance is an approximately unbiased
estimate of the trait-explained variance whether or not the outcome is noisy, so noise in the single-item human ratings does not
shrink T_human. (Noise in the observed trait scores does attenuate the human side somewhat; stated as a caveat.)
Amplitude ratio = sqrt(T_model / T_human): how many times larger, in SD units, the trait-driven differences between people are
in the model's predictions than in people's real ratings. Direction = correlation, across people, of the two OLS-fitted components.
Items with T_human <= 0 (traits explain nothing measurable) are counted but excluded from the per-item ratio; the pooled ratio
uses sums over all items. 95% interval from a delete-half jackknife over people (200 half-samples without replacement). -> analysis/trait_amplitude.json"""
import json, os
import numpy as np, pandas as pd
PKG = "<DATA_ROOT>"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis")
P = pd.read_csv(f"{PKG}/03_OCTOBER_EXPANSION/offline_reproduction_candidate/data/core/MEDIA469_ITEM_PREDICTIONS.csv")
X = pd.read_csv(f"{PKG}/05_ORIGINAL_INPUTS_AND_DEPENDENCIES/escs_data/ipip_neo_scored.csv").set_index("id")
COLS = [c for c in X.columns if c != "id"]

def fit(Xm, y):
    A = np.column_stack([np.ones(len(y)), Xm]); b, *_ = np.linalg.lstsq(A, y, rcond=None); f = A @ b
    if np.sum((y - y.mean()) ** 2) == 0: return 0.0, f          # constant outcome: personality explains none of it (was 0/0 -> NaN; the original analysis, Oct 11)
    n, p = len(y), Xm.shape[1]; r2 = 1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2)
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1); return adj * np.var(y, ddof=1), f

def run(model, cond, idx=None):
    truth = P[(P.model == model) & (P.condition == cond)].pivot(index="escs_id", columns="item", values="truth")
    pred = P[(P.model == model) & (P.condition == cond)].pivot(index="escs_id", columns="item", values="prediction")
    Th, Tm, dirs, per = 0.0, 0.0, [], []
    for it in truth.columns:
        df = pd.DataFrame({"y": truth[it], "p": pred[it]}).join(X[COLS]).dropna()
        if idx is not None: df = df.loc[[i for i in idx if i in df.index]]
        if len(df) < 100: continue
        th, fh = fit(df[COLS].values, df.y.values); tm, fm = fit(df[COLS].values, df.p.values)
        Th += max(th, 0); Tm += max(tm, 0)
        dirs.append(np.corrcoef(fh, fm)[0, 1])
        if th > 0: per.append(np.sqrt(max(tm, 0) / th))
    return np.sqrt(Tm / Th), float(np.nanmean(dirs)), per

res = {}
people = P.escs_id.unique(); rng = np.random.default_rng(20261009)
for cond in ["C-fac", "C-cardBF", "C-cardFull"]:
    for model in sorted(P.model.unique()):
        pooled, dr, per = run(model, cond)
        halves = []                                    # delete-half jackknife (as in amplitude_cohorts.py; Shao & Wu 1989)
        for _ in range(200):
            halves.append(run(model, cond, idx=list(rng.choice(people, len(people) // 2, replace=False)))[0])
        se = float(np.std(halves)); lo, hi = pooled - 1.96 * se, pooled + 1.96 * se
        res[f"{model}|{cond}"] = {"amplitude_ratio_pooled": float(pooled), "ci95": [float(lo), float(hi)], "direction_r": dr,
                                  "per_item_ratio_median": float(np.median(per)), "items_with_human_signal": len(per)}
        print(f"{model:10s} {cond:11s} amplitude x{pooled:.2f} [{lo:.2f}, {hi:.2f}]  per-item median x{np.median(per):.2f} ({len(per)} items with human signal)  direction r {dr:.2f}", flush=True)
json.dump(res, open(os.path.join(OUT, "trait_amplitude.json"), "w"), indent=1)
