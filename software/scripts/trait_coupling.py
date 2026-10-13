#!/usr/bin/env python3
"""Trait-to-taste coupling in models vs people.
For each of the 109 media items: how much of the variation ACROSS PEOPLE do the 35 IPIP-NEO domain+facet scores explain
  - in the people's real ratings (truth), and
  - in each model's predicted ratings (per model x condition)?
Cross-validated R^2 (ridge, alpha chosen by inner CV; 10 outer folds by person). Same people and items on both sides.
Direction: correlation across people between the out-of-fold trait-fitted model rating and the out-of-fold trait-fitted human rating
(does the model's trait->taste mapping point the same way as people's?).
Inputs (read-only): the original offline reproduction package + ESCS ipip_neo_scored.csv. -> analysis/trait_coupling.json"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
PKG = "<DATA_ROOT>"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis")
P = pd.read_csv(f"{PKG}/03_OCTOBER_EXPANSION/offline_reproduction_candidate/data/core/MEDIA469_ITEM_PREDICTIONS.csv")
X = pd.read_csv(f"{PKG}/05_ORIGINAL_INPUTS_AND_DEPENDENCIES/escs_data/ipip_neo_scored.csv").set_index("id")
COLS = [c for c in X.columns if c != "id"]                 # 30 facets + 5 domains = 35
ALPHAS = np.logspace(-2, 4, 25)

def oof(Xm, y, seed=0):
    """Out-of-fold ridge predictions (10 folds by person)."""
    pred = np.full(len(y), np.nan)
    for tr, te in KFold(10, shuffle=True, random_state=seed).split(Xm):
        sc = StandardScaler().fit(Xm[tr]); m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xm[tr]), y[tr])
        pred[te] = m.predict(sc.transform(Xm[te]))
    return pred

def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

res = {"items": {}, "summary": {}}
conds = sys.argv[1:] or ["C-fac"]
truth = P[(P.model == "flash") & (P.condition == "C-fac")].pivot(index="escs_id", columns="item", values="truth")
items = list(truth.columns)
for cond in conds:
    for model in sorted(P.model.unique()):
        pred = P[(P.model == model) & (P.condition == cond)].pivot(index="escs_id", columns="item", values="prediction")
        r_h, r_m, dirc, sd_h, sd_m = [], [], [], [], []
        for it in items:
            df = pd.DataFrame({"y": truth[it], "p": pred[it]}).join(X[COLS]).dropna()
            if len(df) < 200: continue
            Xm = df[COLS].values
            fh = oof(Xm, df.y.values); fm = oof(Xm, df.p.values)
            r_h.append(r2(df.y.values, fh)); r_m.append(r2(df.p.values, fm))
            dirc.append(np.corrcoef(fh, fm)[0, 1] if np.std(fh) > 0 and np.std(fm) > 0 else np.nan)
            sd_h.append(df.y.std()); sd_m.append(df.p.std())
            res["items"].setdefault(it, {})["human_r2"] = r_h[-1]
            res["items"][it][f"{model}|{cond}"] = {"r2": r_m[-1], "direction_r": dirc[-1]}
        r_h, r_m, dirc = np.array(r_h), np.array(r_m), np.array(dirc)
        res["summary"][f"{model}|{cond}"] = {"n_items": len(r_m), "human_r2_mean": float(r_h.mean()), "human_r2_median": float(np.median(r_h)),
            "model_r2_mean": float(r_m.mean()), "model_r2_median": float(np.median(r_m)), "ratio_of_means": float(r_m.mean() / r_h.mean()),
            "direction_r_mean": float(np.nanmean(dirc)), "items_model_gt_human": int((r_m > r_h).sum()),
            "sd_human_mean": float(np.mean(sd_h)), "sd_model_mean": float(np.mean(sd_m))}
        s = res["summary"][f"{model}|{cond}"]
        print(f"{model:10s} {cond:10s} items {s['n_items']}  human R2 mean {s['human_r2_mean']:.3f} (med {s['human_r2_median']:.3f})  model R2 mean {s['model_r2_mean']:.3f} (med {s['model_r2_median']:.3f})  ratio {s['ratio_of_means']:.1f}  direction r {s['direction_r_mean']:.2f}  items model>human {s['items_model_gt_human']}  sd human {s['sd_human_mean']:.2f} model {s['sd_model_mean']:.2f}", flush=True)
json.dump(res, open(os.path.join(OUT, "trait_coupling.json"), "w"), indent=1)
