#!/usr/bin/env python3
"""Collect the extension experiments' AGGREGATE results into one public file (no per-person rows). -> EXTENSIONS_RESULTS.json"""
import json, glob, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
def share(pred):
    out = []
    for it in pred.columns:
        df = pd.DataFrame({"p": pred[it]}).join(C.X[C.COLS]).dropna()
        if len(df) < 100 or df.p.std() == 0: continue
        t, _ = C._fit(df[C.COLS].values, df.p.values); out.append(t / np.var(df.p.values, ddof=1))
    return float(np.mean(out))
def summarize(pr, B=100):
    est, lo, hi = C.jackknife_amp(pr, B=B)
    return {"amplitude": est, "amplitude_ci95": [lo, hi], "direction": C.amplitude(pr)[1], "distinctive": C.boot_mean(C.distinctive(pr)),
            "flatness": C.flatness(pr), "personality_share": share(pr)}
R = {"reference": {"people_personality_share": share(C.TRUTH.loc[[i for i in C.TRUTH.index if i in C.X.index]]),
                   "ridge_amplitude_direction": list(C.amplitude(C.RIDGE))}}
R["A_odd_tastes"] = json.load(open(f"{HERE}/expA/RESULTS_A.json"))
B = json.load(open(f"{HERE}/expB/RESULTS_B.json"))
for f in sorted(glob.glob(f"{HERE}/expB/out_local/*.jsonl")):
    tag = os.path.basename(f)[:-6]; d = pd.read_json(f, lines=True)
    for fmt in ("raw", "chat"):
        B[f"{tag}|{fmt}"]["personality_share"] = share(d[d.fmt == fmt].pivot(index="pid", columns="item", values="ev").astype(float))
R["B_olmo_ladder_instruction_prompt"] = B
rec = {}
for f in sorted(glob.glob(f"{HERE}/expB2/out_local/*.jsonl")) + sorted(glob.glob(f"{HERE}/expB/out_record_local/*.jsonl")):
    d = pd.read_json(f, lines=True); rec[os.path.basename(f)[:-6]] = dict(summarize(d.pivot(index="pid", columns="item", values="ev").astype(float)), mass=float(d["mass"].mean()))
R["B2_olmo_survey_record_prompt"] = rec
C3 = json.load(open(f"{HERE}/expC_haiku_reader/RESULTS_C3.json"))
def load(f):
    best = {}
    for l in open(f):
        x = json.loads(l); k = (x["set"], x["pid"], x["mode"])
        if x["haiku"] or k not in best: best[k] = x
    return best
Hh, Mm = load(f"{HERE}/expC_haiku_reader/haiku.jsonl"), load(f"{HERE}/expC_haiku_reader/mistral.jsonl")
rows = [dict(h, mistral=Mm[k]["haiku"]) for k, h in Hh.items()]
bias = {}
for name, sets, mode in [("college_bfi2", ["college_bfi2"], "readback"), ("college_skills", ["college_bessi"], "readback"),
                         ("twin_harmonies_direct", ["twin_sept", "twin_oct", "harmonies_sept"], "continuous"), ("indonesia", ["indonesia_oct"], "readback")]:
    rr = [r for r in rows if r["set"] in sets and r["mode"] == mode]; ds = sorted(rr[0]["truth"])
    bias[name] = {key: {d: float(np.mean([r[key][d] - r["truth"][d] for r in rr])) for d in ds} for key in ("luna", "haiku", "mistral")}
R["C_three_readers"] = {"rank_r_mae_n": C3, "bias_estimate_minus_truth": bias,
                        "readers": {"luna": "openai/gpt-5.6-luna (the paper's reader; saved answers)", "haiku": "anthropic/claude-haiku-5.5", "mistral": "mistralai/mistral-large-4-0"}}
json.dump(R, open(f"{HERE}/EXTENSIONS_RESULTS.json", "w"), indent=1)
print("wrote EXTENSIONS_RESULTS.json; people share", round(R["reference"]["people_personality_share"], 3), "| record runs", {k: round(v["amplitude"], 2) for k, v in rec.items()})
