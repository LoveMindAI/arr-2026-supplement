import json, numpy as np
from scipy.stats import spearmanr
GROUPS = {"Twin (150)": ["twin_sept", "twin_oct"], "Harmonies (148)": ["harmonies_sept"], "Indonesia (100)": ["indonesia_oct"],
          "College BFI-2 (200)": ["college_bfi2"], "College skills (200)": ["college_bessi"]}
def metrics(rows, key):
    rows = [r for r in rows if r.get(key)]
    doms = sorted(rows[0]["truth"].keys())
    rho = {d: spearmanr([r[key][d] for r in rows], [r["truth"][d] for r in rows]).correlation for d in doms}
    mae = float(np.mean([abs(r[key][d] - r["truth"][d]) for r in rows for d in doms]))
    return float(np.mean(list(rho.values()))), mae, len(rows), rho
def table(T, key):
    out = {}
    for g, sets in GROUPS.items():
        for mode in ("continuous", "questionnaire", "readback"):
            rows = [t for t in T if t["set"] in sets and t["mode"] == mode]
            if rows and any(r.get(key) for r in rows): out[f"{g} {mode}"] = metrics(rows, key)
    return out
