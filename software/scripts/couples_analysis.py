#!/usr/bin/env python3
"""Analysis of the Harmonies couple conditions (written before results were read, Oct 9).
Conditions per model: numeric (September baseline, own personality only), rel, partner_p, partner_t, other_t.
Distinctive = rating minus the population norm (the original couple-grouped disjoint reference). All intervals: 2,000 whole-couple bootstraps.
 1. Distinctive accuracy per condition (mean within-person Spearman of distinctive profiles).
 2. Paired contrasts per model: partner_t - other_t (does the partner's evidence bind to the person, beyond any real person's ratings);
    partner_t - numeric; partner_p - rel; rel - numeric.
 3. partner_t vs the partner's real ratings used directly as the prediction (the .29 reference), paired per person.
 4. Predicted partner concordance (each model x condition's own mean profile removed) for numeric, rel, partner_p, vs real partners.
 5. Copying: within-person Spearman between the prediction and the ratings shown in the prompt (distinctive), for partner_t and other_t,
    next to the real resemblance of partners (.29). Copying the stranger's ratings would be a failure to reason about whose they are.
-> analysis/couples_conditions.json"""
import json, os, glob
import numpy as np
from scipy.stats import spearmanr
PKG = "<DATA_ROOT>/03_OCTOBER_EXPANSION/offline_reproduction_candidate/data"
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "analysis")
E = np.load(f"{PKG}/expanded_predictions.npz"); R = np.load(f"{PKG}/expanded_reference.npz")
ids = list(R["harmonies__genres__ids"]); grp = np.array(R["harmonies__genres__groups"]); keys = list(R["harmonies__genres__keys"])
Y = R["harmonies__genres__truth"]; N = R["harmonies__genres__population"]; D = Y - N; n = len(ids)
pos = {p: i for i, p in enumerate(ids)}
partner = {i: [j for j in range(n) if grp[j] == grp[i] and j != i][0] for i in range(n)}
rho = lambda a, b: (lambda m: spearmanr(a[m], b[m]).correlation if m.sum() >= 8 else np.nan)(~np.isnan(a) & ~np.isnan(b))
couples = np.unique(grp); rng = np.random.default_rng(20261009)
B = [rng.choice(couples, len(couples)) for _ in range(2000)]
rows_of = {c: np.where(grp == c)[0] for c in couples}
def ci(per):                                       # per-person values -> mean and whole-couple bootstrap CI
    per = np.asarray(per, float); est = np.nanmean(per)
    bs = [np.nanmean(np.concatenate([per[rows_of[c]] for c in s])) for s in B]
    return {"mean": float(est), "ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])]}

P = {}
for m in ["sol", "glm", "flash"]:
    P[(m, "numeric")] = E[f"harmonies__genres__{m}__numeric"]
    for c in ["rel", "partner_p", "partner_t", "other_t"]:
        A = np.full((n, len(keys)), np.nan); shown = np.full((n, len(keys)), np.nan)
        for f in glob.glob(os.path.join(OUT, "couples_run", "responses", f"*__{c}__{m}.json")):
            r = json.load(open(f))
            if not r.get("valid"): continue
            i = pos[r["person"]]; A[i] = [r["parsed"][k] for k in keys]
        P[(m, c)] = A
res = {"n_people": n, "n_couples": len(couples), "accuracy": {}, "contrasts": {}, "vs_partner_alone": {}, "concordance": {}, "copying": {}}
acc = {}
for (m, c), A in P.items():
    acc[(m, c)] = np.array([rho(A[i] - N[i], D[i]) for i in range(n)])
    res["accuracy"][f"{m}|{c}"] = ci(acc[(m, c)]); res["accuracy"][f"{m}|{c}"]["n_valid"] = int(np.sum(~np.isnan(A).all(1)))
partner_alone = np.array([rho(D[partner[i]], D[i]) for i in range(n)])
res["partner_alone"] = ci(partner_alone)
for m in ["sol", "glm", "flash"]:
    for a, b in [("partner_t", "other_t"), ("partner_t", "numeric"), ("partner_p", "rel"), ("rel", "numeric")]:
        res["contrasts"][f"{m}|{a}-{b}"] = ci(acc[(m, a)] - acc[(m, b)])
    res["vs_partner_alone"][f"{m}|partner_t-partner_alone"] = ci(acc[(m, "partner_t")] - partner_alone)
    for c in ["numeric", "rel", "partner_p"]:
        A = P[(m, c)]; M = A - np.nanmean(A, 0)
        res["concordance"][f"{m}|{c}"] = ci([rho(M[i], M[partner[i]]) for i in range(n)])
    donors = {}
    for f in glob.glob(os.path.join(OUT, "couples_run", "responses", f"*__other_t__{m}.json")):
        r = json.load(open(f)); donors[pos[r["person"]]] = pos[r["donor"]] if r.get("donor") in pos else None
    A = P[(m, "partner_t")]; res["copying"][f"{m}|partner_t"] = ci([rho(A[i] - N[i], D[partner[i]]) for i in range(n)])
    A = P[(m, "other_t")]; res["copying"][f"{m}|other_t (copies the stranger)"] = ci([rho(A[i] - N[i], D[donors[i]]) if donors.get(i) is not None else np.nan for i in range(n)])
res["real_partner_concordance"] = ci([rho(D[i], D[partner[i]]) for i in range(n)])
json.dump(res, open(os.path.join(OUT, "couples_conditions.json"), "w"), indent=1)
f = lambda d: f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
print("real partner concordance", f(res["real_partner_concordance"]), "| partner's real ratings as predictor", f(res["partner_alone"]))
for k, v in res["accuracy"].items(): print("accuracy", k, f(v), "n", v["n_valid"])
for k, v in res["contrasts"].items(): print("contrast", k, f(v))
for k, v in res["vs_partner_alone"].items(): print("vs partner alone", k, f(v))
for k, v in res["concordance"].items(): print("predicted partner concordance", k, f(v))
for k, v in res["copying"].items(): print("copying", k, f(v))
