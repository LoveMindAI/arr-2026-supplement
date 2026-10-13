#!/usr/bin/env python3
"""Extension C: further automated readers for the readback (the paper's reader is GPT-5.6-Luna).
Collect every saved Luna readback that feeds the paper: request body (verbatim messages), Luna's parsed answer, truth.
Sets: Sept Twin(100)+Harmonies(148) continuous & questionnaire; Oct Twin(50) continuous & questionnaire; Oct Indonesia readback;
college BFI-2 and skill readbacks. -> tasks.jsonl (private: portraits of real people; nothing ships)"""
import json, os, glob, ast
S = "<DATA_ROOT_SEPT>/external_execution_20260916"
O = "<DATA_ROOT>/03_OCTOBER_EXPANSION/expansion_execution/external_additions"
B = "<PROJECT_ROOT>/bessi_run"
def dom(x): return x if isinstance(x, dict) else ast.literal_eval(x)
def luna(path):
    if not os.path.exists(path): return None
    d = json.load(open(path)); p = d.get("parsed")
    if d.get("valid") not in (True, "True") or p is None: return None
    return {k: float(v) for k, v in dom(p).items()}
T = []
sept = {p["id"]: p for p in json.load(open(f"{S}/PEOPLE.json"))}
for pid, p in sept.items():
    for mode in ("continuous", "questionnaire"):
        rq = f"{S}/requests/{pid}__{mode}__attempt1.json"
        if os.path.exists(rq): T.append({"set": f"{p['cohort']}_sept", "pid": pid, "mode": mode, "req": rq, "luna": luna(f"{S}/calls/{pid}__{mode}.json"), "truth": dom(p["domains"])})
octt = json.load(open(f"{O}/twin/PEOPLE.json")); octt = octt if isinstance(octt, list) else list(octt.values())
for p in octt:
    for mode in ("continuous", "questionnaire"):
        rq = f"{O}/requests/{p['id']}__{mode}.json"
        if os.path.exists(rq): T.append({"set": "twin_oct", "pid": p["id"], "mode": mode, "req": rq, "luna": luna(f"{O}/calls/{p['id']}__{mode}.json"), "truth": dom(p["domains"])})
ind = json.load(open(f"{O}/indonesia/PEOPLE.json")); ind = ind if isinstance(ind, list) else list(ind.values())
for p in ind:
    for sub in ("", "final_completion/", "technical_completion/"):
        rq = f"{O}/{sub}requests/{p['id']}__readback.json"
        if os.path.exists(rq):
            lp = next((x for x in (f"{O}/{sub}calls/{p['id']}__readback.json", f"{O}/calls/{p['id']}__readback.json") if os.path.exists(x)), "")
            T.append({"set": "indonesia_oct", "pid": p["id"], "mode": "readback", "req": rq, "luna": luna(lp) if lp else None, "truth": dom(p["domains"])}); break
truth = json.load(open(f"{B}/PRIVATE_TRUTH.json"))
for rq in sorted(glob.glob(f"{B}/requests/*__readback__attempt1.json")):
    pid, d = os.path.basename(rq).split("__")[:2]; src = d.split("_to_")[0]
    rb = f"{B}/readbacks/{pid}__{d}.json"
    T.append({"set": f"college_{src}", "pid": pid, "mode": "readback", "req": rq,
              "luna": {k: float(v) for k, v in dom(json.load(open(rb))["domains"]).items()} if os.path.exists(rb) else None, "truth": truth[pid][f"{src}_domains"]})
with open("tasks.jsonl", "w") as f:
    for t in T: f.write(json.dumps(t) + "\n")
import collections
c = collections.Counter((t["set"], t["mode"]) for t in T); m = collections.Counter((t["set"], t["mode"]) for t in T if t["luna"])
for k in sorted(c): print(k, c[k], "luna valid", m[k])
