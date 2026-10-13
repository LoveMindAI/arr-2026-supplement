#!/usr/bin/env python3
"""Harmonies couples.
198 people in 99 couples rated 23 music genres; partners rated independently and were simulated independently from their own
five BFI-44 means. Distinctive taste = rating minus the population norm (the original disjoint reference), per genre.
1. Real partner concordance: within each couple, Spearman correlation of the two partners' distinctive profiles; mean over couples,
   against the same statistic for every cross-couple pair (strangers in the same sample). Bootstrap over couples.
2. Predicted partner concordance: same statistic computed on each model x format's predictions (prediction minus that model x
   format's own mean profile), so it reflects only what the model derives from each partner's personality.
3. Partner as predictor: distinctive accuracy (within-person Spearman of distinctive profiles) of the partner's REAL ratings as a
   prediction of the person, next to each model's distinctive accuracy and the human-trained ridge (the original).
-> analysis/couples.json"""
import json, os, itertools
import numpy as np
from scipy.stats import spearmanr
PKG = "<DATA_ROOT>/03_OCTOBER_EXPANSION/offline_reproduction_candidate/data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis")
E = np.load(f"{PKG}/expanded_predictions.npz"); R = np.load(f"{PKG}/expanded_reference.npz")
ids = R["harmonies__genres__ids"]; grp = R["harmonies__genres__groups"]; Y = R["harmonies__genres__truth"]; N = R["harmonies__genres__population"]
assert list(E["harmonies__genres__ids"]) == list(ids)
D = Y - N
couples = {}
for i, g in enumerate(grp): couples.setdefault(g, []).append(i)
pairs = [tuple(v) for v in couples.values() if len(v) == 2]
rho = lambda a, b: (lambda m: spearmanr(a[m], b[m]).correlation if m.sum() >= 8 else np.nan)(~np.isnan(a) & ~np.isnan(b))
rng = np.random.default_rng(20261009)
def concord(M):
    within = np.array([rho(M[a], M[b]) for a, b in pairs])
    allp = [(a, b) for a, b in itertools.combinations(range(len(M)), 2) if grp[a] != grp[b]]
    sub = [allp[k] for k in rng.choice(len(allp), 4000, replace=False)]
    between = np.nanmean([rho(M[a], M[b]) for a, b in sub])
    boots = [np.nanmean(within[rng.choice(len(within), len(within))]) for _ in range(2000)]
    return {"partners": float(np.nanmean(within)), "strangers": float(between), "ci95_partners": [float(x) for x in np.percentile(boots, [2.5, 97.5])], "n_couples": len(pairs)}
res = {"real": concord(D), "models": {}, "partner_as_predictor": {}}
print("real distinctive taste: partners", round(res["real"]["partners"], 3), res["real"]["ci95_partners"], "strangers", round(res["real"]["strangers"], 3), "couples", len(pairs))
keys = [k for k in E.keys() if k.startswith("harmonies__genres__") and k.split("__")[-1] in ("numeric", "structured", "portrait")]
for k in sorted(keys):
    Pm = E[k]; M = Pm - np.nanmean(Pm, 0)
    c = concord(M); res["models"][k.replace("harmonies__genres__", "")] = c
    acc = np.nanmean([rho(Pm[i] - N[i], D[i]) for i in range(len(Y))])
    c["distinctive_accuracy"] = float(acc)
    print(f"  {k.replace('harmonies__genres__',''):22s} predicted partners {c['partners']:.3f} {np.round(c['ci95_partners'],3)}  strangers {c['strangers']:.3f}   model distinctive accuracy {acc:.3f}")
# partner's real taste as a predictor of the person
pa = []
for a, b in pairs:
    pa += [rho(D[b], D[a]), rho(D[a], D[b])]
ridge = R["harmonies__genres__ridge"]
res["partner_as_predictor"] = {"partner_real_ratings": float(np.nanmean(pa)), "ridge_from_own_personality": float(np.nanmean([rho(ridge[i] - N[i], D[i]) for i in range(len(Y))]))}
print("distinctive accuracy: partner's real ratings", round(res["partner_as_predictor"]["partner_real_ratings"], 3), " human-trained ridge from own personality", round(res["partner_as_predictor"]["ridge_from_own_personality"], 3))
json.dump(res, open(os.path.join(OUT, "couples.json"), "w"), indent=1)
