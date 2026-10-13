import sys, glob, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import common as C
def r2share(pred):
    out = []
    for it in pred.columns:
        df = pd.DataFrame({"p": pred[it]}).join(C.X[C.COLS]).dropna()
        if len(df) < 100 or df.p.std() == 0: continue
        t, _ = C._fit(df[C.COLS].values, df.p.values); out.append(t / np.var(df.p.values, ddof=1))
    return float(np.mean(out))
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "out_local/*.jsonl")):
    d = pd.read_json(f, lines=True); pr = d.pivot(index="pid", columns="item", values="ev").astype(float)
    est, lo, hi = C.jackknife_amp(pr, B=100)
    print(f"{os.path.basename(f)[:-6].replace('allenai_',''):34s} record: mass {d['mass'].mean():.3f} amp {est:.2f} [{lo:.2f},{hi:.2f}] dir {C.amplitude(pr)[1]:.2f} dist {C.boot_mean(C.distinctive(pr))[0]:+.3f} flat {C.flatness(pr):.2f} share {r2share(pr):.3f} argmax-top {d.argmax.value_counts().idxmax()} ({d.argmax.value_counts(normalize=True).max():.0%})", flush=True)
