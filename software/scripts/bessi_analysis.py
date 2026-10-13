#!/usr/bin/env python3
"""BESSI college cohort analysis (written before any prediction was read, Oct 9). Both directions:
  bfi2_to_bessi: five BFI-2 domain means -> 192 BESSI how-well items;  bessi_to_bfi2: five BESSI domain scores -> 60 BFI-2 items.
Evaluation people = 200 (bessi_run/PEOPLE.json); reference = the other 113 complete cases (norms and ridge only, never evaluated).
Predictors: Sol, GLM, Flash x numeric / structured / portrait; a no-person baseline per model; a ridge trained on the reference
people (same five source scores -> each item; alpha by leave-one-out within the reference); the population norm itself (= 0 distinctive).
Measures (same yardsticks as the other Panorama cohorts):
 1. Distinctive accuracy: mean within-person Spearman of (prediction - norm) vs (truth - norm), norm = reference item means.
    95% CI: bootstrap over people (people are independent here). Also the wrong-person value (prediction for i scored against every
    other j) and the binding gap = own - wrong.
 2. Raw fit: mean absolute error; within-person Pearson of raw profiles.
 3. Amplitude and direction (amplitude_cohorts.py statistic, five source domains as predictors, delete-half jackknife CI).
 4. Domain level: predicted items scored into the target instrument's domains (BESSI weighted facet means; BFI-2 keyed means);
    across-person correlation with the true target domains and SD ratio (predicted / true), next to the ridge.
 5. Readback: across-person Spearman of Luna's continuous readback vs the source domains (does the personality survive the writing).
-> analysis/bessi_results.json"""
import json, os
import numpy as np
from scipy.stats import rankdata, spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "bessi_run"); OUT = os.path.join(HERE, "..", "analysis")
CATS = "<INSTRUMENT_CATALOGS>"
BFI2 = json.load(open(f"{CATS}/BFI2_CATALOG.json")); BESSI = json.load(open(f"{CATS}/BESSI_CATALOG.json"))
people = json.load(open(f"{ROOT}/PEOPLE.json")); truth = json.load(open(f"{ROOT}/PRIVATE_TRUTH.json")); ref = json.load(open(f"{ROOT}/REFERENCE_IDS.json"))
TG = json.load(open(f"{ROOT}/TARGETS.json"))
rng = np.random.default_rng(20261009)
B2 = ["E", "A", "C", "N", "O"]; BS = ["SM", "SE", "CO", "ER", "IN"]
BS_COL = {"SM": "BESSI_SelfManagementSkills", "SE": "BESSI_SocialEngagementSkills", "CO": "BESSI_CooperationSkills", "ER": "BESSI_EmotionalResilienceSkills", "IN": "BESSI_InnovationSkills"}
DIRS = {"bfi2_to_bessi": ("bfi2", "bessi", B2, BS), "bessi_to_bfi2": ("bessi", "bfi2", BS, B2)}
ROUTES = ["sol", "glm", "flash"]; CONDS = ["numeric", "structured", "portrait"]

def score_domains(M, inst, items):
    ix = {k: j for j, k in enumerate(items)}
    if inst == "bessi":
        fac = {f: np.nanmean(M[:, [ix[i] for i in s["item_ids"]]], 1) for f, s in BESSI["facets"].items()}
        return np.column_stack([np.mean([fac[f] for f in BESSI["domains"][BS_COL[d]]["facets"]], 0) for d in BS])
    out = []
    for d in B2:
        its = [i for i, it in BFI2["items"].items() if BFI2["facets"][it["facet"]]["domain"] == d]
        out.append(np.nanmean(np.column_stack([M[:, ix[i]] if BFI2["items"][i]["key"] == 1 else 6 - M[:, ix[i]] for i in its]), 1))
    return np.column_stack(out)

def rowrank(A):
    R = np.full(A.shape, np.nan)
    for i in range(len(A)):
        m = ~np.isnan(A[i])
        if m.sum() >= 8: R[i, m] = rankdata(A[i, m])
    return R

def distinctive(P, Y, N):
    """own (per person) and wrong-person (per person: mean over others) Spearman of distinctive profiles."""
    A = rowrank(P - N); B = rowrank(Y - N)
    A = (A - np.nanmean(A, 1, keepdims=True)); B = (B - np.nanmean(B, 1, keepdims=True))
    A = A / np.sqrt(np.nansum(A ** 2, 1, keepdims=True)); B = B / np.sqrt(np.nansum(B ** 2, 1, keepdims=True))
    C = np.nan_to_num(A) @ np.nan_to_num(B).T            # complete rows only (predictions are complete when present)
    ok = ~np.isnan(P).all(1)
    own = np.where(ok, np.diag(C), np.nan)
    wrong = np.array([np.mean(np.delete(C[i], i)) if ok[i] else np.nan for i in range(len(C))])
    return own, wrong

def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    bs = [np.mean(rng.choice(v, len(v))) for _ in range(2000)]
    return {"mean": float(np.mean(v)), "ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "n": int(len(v))}

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
    return (np.sqrt(tm / th) if th > 0 else np.nan), (float(np.mean(dirs)) if dirs else np.nan)

def amp_ci(X, Y, P):
    rows = np.where(~np.isnan(P).all(1))[0]; est, dr = amp(X, Y, P, rows); halves = []
    for _ in range(300): halves.append(amp(X, Y, P, rng.choice(rows, len(rows) // 2, replace=False))[0])
    se = float(np.nanstd(halves)); return {"amplitude_ratio": float(est), "ci95": [float(est - 1.96 * se), float(est + 1.96 * se)], "direction_r": dr, "n_undefined_halves": int(np.sum(np.isnan(halves)))}

def ridge_fit_predict(Xr, Yr, Xe, grid=(0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000)):
    mu, sd = Xr.mean(0), Xr.std(0); Zr = (Xr - mu) / sd; Ze = (Xe - mu) / sd
    A = np.column_stack([np.ones(len(Zr)), Zr]); best = None
    for lam in grid:
        Pen = lam * np.eye(A.shape[1]); Pen[0, 0] = 0
        H = A @ np.linalg.solve(A.T @ A + Pen, A.T); F = H @ Yr
        loo = np.sum(((Yr - F) / (1 - np.diag(H))[:, None]) ** 2)
        if best is None or loo < best[0]: best = (loo, lam)
    lam = best[1]; Pen = lam * np.eye(A.shape[1]); Pen[0, 0] = 0
    W = np.linalg.solve(A.T @ A + Pen, A.T @ Yr)
    return np.column_stack([np.ones(len(Ze)), Ze]) @ W, lam

res = {"n_eval": len(people), "n_reference": len(ref), "directions": {}}
for d, (src, tgt, sdom, tdom) in DIRS.items():
    items = list(TG[tgt][0]["items"]); ids = [p["id"] for p in people]
    Y = np.array([[truth[i][tgt][k] for k in items] for i in ids], float); Yr = np.array([[truth[i][tgt][k] for k in items] for i in ref], float)
    X = np.array([[truth[i][f"{src}_domains"][k] for k in sdom] for i in ids]); Xr = np.array([[truth[i][f"{src}_domains"][k] for k in sdom] for i in ref])
    N = Yr.mean(0); Dt = score_domains(Y, tgt, items)
    preds = {}
    for r in ROUTES:
        for c in CONDS:
            M = np.full(Y.shape, np.nan)
            for k, i in enumerate(ids):
                f = os.path.join(ROOT, "predictions", f"{i}__{d}__{c}__{r}.json")
                if os.path.exists(f): a = json.load(open(f))["answers"]; M[k] = [a[t] for t in items]
            preds[f"{r}|{c}"] = M
        f = os.path.join(ROOT, "predictions", f"no_person__{d}__no_person__{r}.json")
        if os.path.exists(f): a = json.load(open(f))["answers"]; preds[f"{r}|no_person"] = np.tile([a[t] for t in items], (len(ids), 1)).astype(float)
    Pr, lam = ridge_fit_predict(Xr, Yr, X); preds["ridge|reference"] = Pr
    out = {"n_items": len(items), "ridge_alpha": lam, "predictors": {}}
    for k, P in preds.items():
        if np.isnan(P).all(): continue
        own, wrong = distinctive(P, Y, N); ok = ~np.isnan(P).all(1)
        e = {"n_people": int(ok.sum()), "distinctive": ci(own), "wrong_person": ci(wrong), "binding_gap": ci(own - wrong),
             "mae": float(np.nanmean(np.abs(P[ok] - Y[ok]))),
             "raw_profile_r": float(np.nanmean([np.corrcoef(P[i], Y[i])[0, 1] for i in np.where(ok)[0] if np.std(P[i]) > 0]))}
        if not k.endswith("no_person"):
            e["amplitude"] = amp_ci(X, Y, P)
            Dp = score_domains(P[ok], tgt, items)
            e["domains"] = {t: {"r_with_true": float(np.corrcoef(Dp[:, j], Dt[ok, j])[0, 1]), "sd_ratio": float(np.std(Dp[:, j]) / np.std(Dt[ok, j])),
                                "mean_pred": float(np.mean(Dp[:, j])), "mean_true": float(np.mean(Dt[ok, j]))} for j, t in enumerate(tdom)}
        out["predictors"][k] = e
        a = e.get("amplitude", {})
        print(f"{d:14s} {k:20s} n={e['n_people']:3d} distinctive {e['distinctive']['mean']:+.3f} [{e['distinctive']['ci95'][0]:+.3f},{e['distinctive']['ci95'][1]:+.3f}]"
              f"  wrong {e['wrong_person']['mean']:+.3f}  gap {e['binding_gap']['mean']:+.3f}  MAE {e['mae']:.2f}  raw r {e['raw_profile_r']:.2f}"
              + (f"  amp x{a['amplitude_ratio']:.2f} [{a['ci95'][0]:.2f},{a['ci95'][1]:.2f}] dir {a['direction_r']:.2f}" if a else ""), flush=True)
    # readback (Luna) vs source domains
    rb = {}
    for k, i in enumerate(ids):
        f = os.path.join(ROOT, "readbacks", f"{i}__{d}.json")
        if os.path.exists(f): rb[k] = json.load(open(f))["domains"]
    if rb:
        ks = sorted(rb); out["readback"] = {t: float(spearmanr([rb[k][t] for k in ks], X[ks, j]).correlation) for j, t in enumerate(sdom)}
        out["readback"]["n"] = len(ks); print(d, "readback spearman", {t: round(v, 3) for t, v in out["readback"].items()}, flush=True)
    # human cross-instrument correlations among the evaluation people (what personality really carries to the target)
    out["human_cross_r"] = {s: {t: float(np.corrcoef(X[:, a], Dt[:, b])[0, 1]) for b, t in enumerate(tdom)} for a, s in enumerate(sdom)}
    res["directions"][d] = out
json.dump(res, open(os.path.join(OUT, "bessi_results.json"), "w"), indent=1)
print("wrote analysis/bessi_results.json")
