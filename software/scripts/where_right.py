#!/usr/bin/env python3
"""Where are the models right?
Per media item (ESCS, facet condition): how well each model orders people on that item (Spearman across people, prediction vs truth),
against how much real signal personality carries for that item (sqrt of adjusted R^2 of the people's real ratings on the 35 scores),
and against the human-trained ridge's per-item accuracy (the original disjoint ridge). Spearman across the 109 items, with a bootstrap
over items. -> analysis/where_right.json"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
PKG = "<DATA_ROOT>"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis")
P = pd.read_csv(f"{PKG}/03_OCTOBER_EXPANSION/offline_reproduction_candidate/data/core/MEDIA469_ITEM_PREDICTIONS.csv")
RG = pd.read_csv(f"{PKG}/03_OCTOBER_EXPANSION/offline_reproduction_candidate/data/core/MEDIA469_ECOSYSTEM/HUMAN_RIDGE_PREDICTIONS_LONG.csv")
X = pd.read_csv(f"{PKG}/05_ORIGINAL_INPUTS_AND_DEPENDENCIES/escs_data/ipip_neo_scored.csv").set_index("id"); COLS = [c for c in X.columns if c != "id"]
sub = P[P.condition == "C-fac"]
truth = sub[sub.model == "flash"].pivot(index="escs_id", columns="item", values="truth"); items = list(truth.columns)
def adj(Xm, y):
    A = np.column_stack([np.ones(len(y)), Xm]); b, *_ = np.linalg.lstsq(A, y, rcond=None); f = A @ b
    r2 = 1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2); n, p = len(y), Xm.shape[1]; return 1 - (1 - r2) * (n - 1) / (n - p - 1)
signal = {}
for it in items:
    df = truth[[it]].join(X[COLS]).dropna(); signal[it] = np.sqrt(max(adj(df[COLS].values, df[it].values), 0))
rp = RG.pivot(index="escs_id", columns="item", values="prediction"); rp = rp.loc[[i for i in rp.index if i in truth.index]]
ridge_acc = {it: spearmanr(rp[it], truth.loc[rp.index, it], nan_policy="omit").correlation for it in items if it in rp}
res = {"items": {}, "summary": {}}; rng = np.random.default_rng(1)
s = np.array([signal[i] for i in items]); r = np.array([ridge_acc.get(i, np.nan) for i in items])
for m in sorted(sub.model.unique()):
    pr = sub[sub.model == m].pivot(index="escs_id", columns="item", values="prediction")
    a = np.array([spearmanr(pr[i], truth.loc[pr.index, i], nan_policy="omit").correlation for i in items])
    rho_sig = spearmanr(a, s).correlation; rho_ridge = spearmanr(a, r, nan_policy="omit").correlation
    bs = []
    for _ in range(2000):
        k = rng.integers(0, len(items), len(items)); bs.append(spearmanr(a[k], s[k]).correlation)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    top = a[s >= np.median(s)].mean(); bot = a[s < np.median(s)].mean()
    res["summary"][m] = {"rho_item_accuracy_vs_trait_signal": float(rho_sig), "ci95": [float(lo), float(hi)], "rho_vs_ridge_item_accuracy": float(rho_ridge),
                         "mean_item_accuracy_high_signal_half": float(top), "mean_item_accuracy_low_signal_half": float(bot)}
    print(f"{m:10s} item accuracy vs trait signal rho {rho_sig:.2f} [{lo:.2f}, {hi:.2f}]  vs ridge {rho_ridge:.2f}   accuracy where traits matter {top:.3f}  where they don't {bot:.3f}")
res["ridge_accuracy_high_low"] = [float(np.nanmean(r[s >= np.median(s)])), float(np.nanmean(r[s < np.median(s)]))]
# the ridge's own item accuracy against the trait signal, same bootstrap (added Oct 11 after review: an independent reviewer recomputed .79)
ok = ~np.isnan(r); bs = []
for _ in range(2000):
    k = rng.integers(0, ok.sum(), ok.sum()); bs.append(spearmanr(r[ok][k], s[ok][k]).correlation)
res["ridge_rho_item_accuracy_vs_trait_signal"] = {"rho": float(spearmanr(r[ok], s[ok]).correlation), "ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "n_items": int(ok.sum())}
print("ridge item accuracy vs trait signal", res["ridge_rho_item_accuracy_vs_trait_signal"])
print("ridge accuracy where traits matter / don't", np.round(res["ridge_accuracy_high_low"], 3))
json.dump(res, open(os.path.join(OUT, "where_right.json"), "w"), indent=1)
