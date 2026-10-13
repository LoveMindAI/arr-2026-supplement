#!/usr/bin/env python3
"""Replay each saved Luna readback request VERBATIM (same messages, same max_tokens) to anthropic/claude-haiku-5.5, reasoning off,
default temperature (as Luna). Parse like Luna's validators: {"domains":{...}} for continuous/readback; {"answers":{bfi44_xx:int}}
for the questionnaire, scored with the BFI-44 key (reverse-keyed items = 6 - x). Resumable. Usage: run_haiku.py [--limit N]"""
import json, os, sys, time, threading, urllib.request, argparse
from concurrent.futures import ThreadPoolExecutor
CAT = json.load(open("<DATA_ROOT_SEPT>/external_launch_20260916/bfi44_ocean_adapter/bfi44_catalog.json"))
KEY = os.environ["OPENROUTER_API_KEY"]; lock = threading.Lock()
def score_q(ans):
    out = {}
    for d in "ACENO":
        its = [i for i in CAT["items"] if i["domain"] == d]
        v = [(ans[i["id"]] if i["key"] == 1 else 6 - ans[i["id"]]) for i in its if isinstance(ans.get(i["id"]), (int, float))]
        if len(v) < len(its): return None
        out[d] = sum(v) / len(v)
    return out
def parse(text, mode, truth_keys):
    t = text or ""; s, e = t.find("{"), t.rfind("}")
    try: j = json.loads(t[s:e + 1])
    except Exception: return None
    if mode == "questionnaire": return score_q(j.get("answers") or {})
    d = j.get("domains") or {}
    try: d = {k: float(d[k]) for k in truth_keys}
    except Exception: return None
    return d
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--model", default="anthropic/claude-haiku-5.5"); ap.add_argument("--out", default="haiku.jsonl"); ap.add_argument("--conc", type=int, default=40); a = ap.parse_args(); global MODEL, OUT; MODEL, OUT = a.model, a.out
    T = [json.loads(l) for l in open("tasks.jsonl")]
    if a.limit: T = T[:: max(1, len(T) // a.limit)][:a.limit]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            x = json.loads(l)
            if x.get("haiku"): done.add((x["set"], x["pid"], x["mode"]))
    todo = [t for t in T if (t["set"], t["pid"], t["mode"]) not in done]; print("todo", len(todo), flush=True)
    tot = [0.0]
    def job(t):
        b = json.load(open(t["req"]))
        body = {"model": MODEL, "messages": b["messages"], "max_tokens": int(b.get("max_tokens", 2048)),
                "reasoning": {"enabled": False}, "usage": {"include": True}}
        res, err = None, None
        for k in range(4):
            try:
                req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
                res = json.load(urllib.request.urlopen(req, timeout=240)); break
            except Exception as e: err = str(e)[:200]; time.sleep(3 * (k + 1))
        txt = ((res or {}).get("choices") or [{}])[0].get("message", {}).get("content") if res else None
        h = parse(txt, t["mode"], list(t["truth"].keys())); c = float(((res or {}).get("usage") or {}).get("cost") or 0)
        rec = {k: t[k] for k in ("set", "pid", "mode", "truth", "luna")}; rec.update(haiku=h, cost=c, err=err, model=(res or {}).get("model"), provider=(res or {}).get("provider"), raw=(txt or "")[:400] if not h else None)
        with lock:
            tot[0] += c
            with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
    with ThreadPoolExecutor(a.conc) as ex: list(ex.map(job, todo))
    print("spent $", round(tot[0], 4))
if __name__ == "__main__": main()
