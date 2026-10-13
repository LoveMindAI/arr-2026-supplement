"""Shared scoring for the extension experiments: the paper's definitions (scripts/trait_amplitude.py; app 'Amplitude').
amplitude = sqrt(sum_j T_model_j / sum_j T_people_j), T_j = adjusted R^2 (OLS on the 35 trait scores) x variance across people,
floored at 0; direction = mean over items of corr(fitted model component, fitted people component);
distinctive accuracy = per person Spearman across items of (pred - norm) vs (truth - norm), norm = item mean among other
participants (disjoint_norm); flatness = mean over items of SD_people(pred) / SD_people(truth). Delete-half jackknife CIs."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
PKG = "<DATA_ROOT>"
X = pd.read_csv(f"{PKG}/05_ORIGINAL_INPUTS_AND_DEPENDENCIES/escs_data/ipip_neo_scored.csv").set_index("id")
COLS = [c for c in X.columns if c != "id"]
RID = pd.read_csv(f"{PKG}/03_OCTOBER_EXPANSION/offline_reproduction_candidate/data/core/MEDIA469_ECOSYSTEM/HUMAN_RIDGE_PREDICTIONS_LONG.csv")
TRUTH = RID.pivot(index="escs_id", columns="item", values="truth")
NORM = RID.groupby("item").disjoint_norm.first()
RIDGE = RID.pivot(index="escs_id", columns="item", values="prediction")
def _fit(Xm, y):
    A = np.column_stack([np.ones(len(y)), Xm]); b, *_ = np.linalg.lstsq(A, y, rcond=None); f = A @ b
    ss = np.sum((y - y.mean()) ** 2)
    if ss == 0: return 0.0, f
    n, p = len(y), Xm.shape[1]; r2 = 1 - np.sum((y - f) ** 2) / ss
    return (1 - (1 - r2) * (n - 1) / (n - p - 1)) * np.var(y, ddof=1), f
def amplitude(pred, items=None, idx=None):
    """pred: DataFrame people x items (escs_id index)."""
    items = items or [c for c in pred.columns if c in TRUTH.columns]
    Th = Tm = 0.0; dirs = []
    people = pred.index if idx is None else [i for i in idx if i in pred.index]
    for it in items:
        df = pd.DataFrame({"y": TRUTH.loc[people, it], "p": pred.loc[people, it]}).join(X[COLS]).dropna()
        if len(df) < 100: continue
        th, fh = _fit(df[COLS].values, df.y.values); tm, fm = _fit(df[COLS].values, df.p.values)
        Th += max(th, 0); Tm += max(tm, 0)
        if np.std(fm) > 0 and np.std(fh) > 0: dirs.append(np.corrcoef(fh, fm)[0, 1])
    if Th <= 0: return float("nan"), float("nan")
    return float(np.sqrt(Tm / Th)), float(np.nanmean(dirs)) if dirs else float("nan")
def jackknife_amp(pred, items=None, B=200, seed=0):
    rng = np.random.default_rng(seed); ppl = np.array(pred.index); est = amplitude(pred, items)[0]
    reps = [amplitude(pred, items, rng.choice(ppl, len(ppl) // 2, replace=False))[0] for _ in range(B)]
    # delete-half jackknife: variance = (n_half/(n - n_half)) * var(reps) with n_half = n/2 -> var(reps)
    se = float(np.std(reps, ddof=1)); return est, est - 1.96 * se, est + 1.96 * se
def distinctive(pred, items_by_person=None):
    out = {}
    for pid in pred.index:
        its = items_by_person.get(pid) if items_by_person else list(pred.columns)
        its = [i for i in its if i in TRUTH.columns and pd.notna(TRUTH.loc[pid, i]) and pd.notna(pred.loc[pid, i])] if pid in TRUTH.index else []
        if len(its) < 10: continue
        a = pred.loc[pid, its].values - NORM[its].values; b = TRUTH.loc[pid, its].values - NORM[its].values
        if np.std(a) == 0 or np.std(b) == 0: out[pid] = 0.0; continue
        out[pid] = spearmanr(a, b).correlation
    return pd.Series(out)
def boot_mean(s, B=2000, seed=0):
    rng = np.random.default_rng(seed); v = s.dropna().values
    m = [rng.choice(v, len(v)).mean() for _ in range(B)]; return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def flatness(pred, items=None):
    items = items or [c for c in pred.columns if c in TRUTH.columns]
    r = [pred[it].std() / TRUTH.loc[pred.index, it].std() for it in items if TRUTH.loc[pred.index, it].std() > 0]
    return float(np.nanmean(r))
