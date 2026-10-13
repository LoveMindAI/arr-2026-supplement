#!/usr/bin/env python3
"""Exp B analysis: per OLMo checkpoint and wrapper (raw / chat), the paper's amplitude (35 trait scores), direction,
distinctive accuracy and flatness, from expected ratings (EV over digits 1-7) and from argmax. Also the format check: mean
probability mass on the digits 1-7. Reading rule (fixed before any result): post-training 'paints the mannequin'
if amplitude at a post-trained stage sits above the base's jackknife interval; pretraining does if the base is already above 1
beyond its interval. -> expB/RESULTS_B.json, printed table."""
import json, os, sys, glob
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as C
HERE = os.path.dirname(os.path.abspath(__file__))
rows = {}
for f in sorted(glob.glob(f"{HERE}/out_local/*.jsonl")):
    tag = os.path.basename(f)[:-6]
    d = pd.read_json(f, lines=True)
    for fmt in ("raw", "chat"):
        x = d[d.fmt == fmt]
        if x.empty: continue
        rec = {"mass": float(x["mass"].mean()), "n_people": int(x.pid.nunique())}
        for val in ("ev", "argmax"):
            pr = x.pivot(index="pid", columns="item", values=val).astype(float)
            est, lo, hi = C.jackknife_amp(pr, B=int(os.environ.get("JK", 100)))
            rec[val] = {"amp": est, "amp_ci": [lo, hi], "dir": C.amplitude(pr)[1],
                        "distinctive": C.boot_mean(C.distinctive(pr)), "flat": C.flatness(pr)}
        rows[f"{tag}|{fmt}"] = rec
        e = rec["ev"]; print(f"{tag:42s} {fmt:4s} mass {rec['mass']:.2f}  amp {e['amp']:.2f} [{e['amp_ci'][0]:.2f},{e['amp_ci'][1]:.2f}]  dir {e['dir']:.2f}  dist {e['distinctive'][0]:+.3f}  flat {e['flat']:.2f}   | argmax amp {rec['argmax']['amp']:.2f}", flush=True)
json.dump(rows, open(f"{HERE}/RESULTS_B.json", "w"), indent=1)
