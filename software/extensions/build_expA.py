#!/usr/bin/env python3
"""Extension A: do a person's own odd tastes pull a model off the personality stereotype?
Each request = the person's ORIGINAL ESCS 35-trait-score prompt (order_campaign/private_prompts/<id>__original.txt), verbatim,
with one added block inserted after the trait lines (arms 1-4). Nothing else in the prompt changes.
Arms: A0 scores only | A1 +4 own countertypical tastes | A2 +4 own typical tastes | A3 +4 countertypical tastes of a WRONG person
(fixed circular map over sorted ids) | A4 +12 own countertypical tastes.
Countertypical = the person's items with the largest |real rating - ridge prediction|, where the ridge was trained on other
participants (MEDIA469_ECOSYSTEM/HUMAN_RIDGE_PREDICTIONS_LONG.csv); typical = smallest |residual|. Ties broken by item order.
Evaluation items per person = all items with a real rating, minus the union of every item shown to that person in ANY arm.
-> requests.jsonl (private: contains real ESCS answers; ESCS is CC0 but nothing here ships), selections.json"""
import json, os, sys
import pandas as pd
D = "<DATA_ROOT_SEPT>/order_campaign/private_prompts"
R = "<DATA_ROOT>/03_OCTOBER_EXPANSION/offline_reproduction_candidate/data/core/MEDIA469_ECOSYSTEM/HUMAN_RIDGE_PREDICTIONS_LONG.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
rid = pd.read_csv(R)
ids = sorted(int(f.split("__")[0]) for f in os.listdir(D) if f.endswith("__original.txt"))
rid = rid[rid.escs_id.isin(ids)].dropna(subset=["truth"])
first = open(f"{D}/{ids[0]}__original.txt").read().split("\n")
ITEMS = [l.split(":", 1)[0] for l in first if l.split(":", 1)[0].rstrip("0123456789") in ("music", "books", "movies", "tv") and ":" in l]
LAB = {l.split(":", 1)[0]: l.split(":", 1)[1].strip() for l in first if l.split(":", 1)[0] in ITEMS}
order = {it: k for k, it in enumerate(ITEMS)}
sel = {}
for pid, g in rid.groupby("escs_id"):
    g = g.assign(res=(g.truth - g.prediction).abs(), o=g.item.map(order)).sort_values(["res", "o"], ascending=[False, True])
    ct = g.item.tolist(); tv = dict(zip(g.item, g.truth.astype(int)))
    sel[pid] = {"ct4": ct[:4], "ct12": ct[:12], "typ4": list(reversed(ct))[:4], "truth": tv}
wrong = {p: ids[(k + 1) % len(ids)] for k, p in enumerate(ids)}
def block(pid_src, items, n):
    t = sel[pid_src]["truth"]
    return (f"\n\nIn addition, {n} of this person's own ratings from the list below are known (1-7 scale):\n" +
            "\n".join(f"- {it}: {LAB[it]} = {t[it]}" for it in items) + "\n")
reqs = []
for pid in ids:
    if pid not in sel: continue
    lines = open(f"{D}/{pid}__original.txt").read().split("\n")
    k = next(i for i, l in enumerate(lines) if l.startswith("Answer every question AS this person"))
    assert lines[k - 1] == "" and all(l.startswith("- ") for l in lines[2:k - 1]) and len(lines[2:k - 1]) == 35
    head, tail = "\n".join(lines[:k - 1]), "\n".join(lines[k - 1:])
    w = wrong[pid]
    arms = {"A0": "", "A1": block(pid, sel[pid]["ct4"], 4), "A2": block(pid, sel[pid]["typ4"], 4),
            "A3": block(w, sel[w]["ct4"], 4), "A4": block(pid, sel[pid]["ct12"], 12)}
    shown = set(sel[pid]["ct12"]) | set(sel[pid]["typ4"]) | set(sel[w]["ct4"])
    sel[pid]["eval_items"] = [it for it in ITEMS if it in sel[pid]["truth"] and it not in shown]
    sel[pid]["wrong"] = w
    for arm, b in arms.items():
        reqs.append({"pid": pid, "arm": arm, "prompt": head + b + tail})
with open(f"{HERE}/requests.jsonl", "w") as f:
    for r in reqs: f.write(json.dumps(r) + "\n")
json.dump({str(k): {kk: vv for kk, vv in v.items() if kk != "truth"} for k, v in sel.items()}, open(f"{HERE}/selections.json", "w"))
import statistics
print("people", len({r['pid'] for r in reqs}), "requests", len(reqs), "items", len(ITEMS),
      "eval items median", statistics.median(len(v["eval_items"]) for v in sel.values() if "eval_items" in v))
