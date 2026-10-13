#!/usr/bin/env python3
"""Run Exp A requests on OpenRouter. Usage: run_expA.py MODEL_KEY [--pilot N] [--conc 24]
Writes out/<model_key>.jsonl (one line per pid x arm; resumable: ok lines are skipped). Cost from OpenRouter usage accounting."""
import json, os, re, sys, time, argparse, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {  # same settings for every arm
    "flash":   dict(model="google/gemini-3-flash-preview", temperature=0.0, max_tokens=3000, reasoning={"effort": "none"}),
    "qwen37":  dict(model="qwen/qwen3.7-plus", temperature=0.0, max_tokens=6000, reasoning={"effort": "none"}),
    "minimax": dict(model="minimax/minimax-m3", temperature=0.0, max_tokens=8000),
}
KEY = os.environ["OPENROUTER_API_KEY"]
lock = threading.Lock()
def call(cfg, prompt):
    body = dict(cfg, messages=[{"role": "user", "content": prompt}], usage={"include": True})
    for attempt in range(4):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            j = json.load(urllib.request.urlopen(req, timeout=300))
            if "error" in j: raise RuntimeError(str(j["error"])[:200])
            return j
        except Exception as e:
            err = str(e)[:200]; time.sleep(3 * (attempt + 1))
    return {"error": err}
def parse(t):
    t = t or ""; s, e = t.find("{"), t.rfind("}")
    try: d = json.loads(t[s:e + 1])
    except Exception: return None
    return {k: int(v) for k, v in d.items() if str(v).strip().lstrip("-").isdigit() and 1 <= int(v) <= 7}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--pilot", type=int, default=0); ap.add_argument("--conc", type=int, default=24)
    a = ap.parse_args(); cfg = MODELS[a.model]
    reqs = [json.loads(l) for l in open(f"{HERE}/requests.jsonl")]
    if a.pilot:
        keep = sorted({r["pid"] for r in reqs})[:: max(1, len({r["pid"] for r in reqs}) // a.pilot)][:a.pilot]; reqs = [r for r in reqs if r["pid"] in keep]
    os.makedirs(f"{HERE}/out", exist_ok=True); out = f"{HERE}/out/{a.model}.jsonl"
    done = set()
    if os.path.exists(out):
        for l in open(out):
            x = json.loads(l)
            if x.get("ok"): done.add((x["pid"], x["arm"]))
    todo = [r for r in reqs if (r["pid"], r["arm"]) not in done]
    print(a.model, "todo", len(todo), "done", len(done), flush=True)
    tot = [0.0]
    def job(r):
        j = call(cfg, r["prompt"])
        txt = ((j.get("choices") or [{}])[0].get("message") or {}).get("content") if "choices" in j else None
        pred = parse(txt); cost = float((j.get("usage") or {}).get("cost") or 0)
        rec = {"pid": r["pid"], "arm": r["arm"], "model": cfg["model"], "ok": bool(pred and len(pred) >= 100), "n": len(pred or {}),
               "pred": pred, "cost": cost, "err": j.get("error"), "usage": j.get("usage"), "finish": ((j.get("choices") or [{}])[0]).get("finish_reason")}
        with lock:
            tot[0] += cost
            with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
    with ThreadPoolExecutor(a.conc) as ex: list(ex.map(job, todo))
    print(a.model, "spent $", round(tot[0], 4), flush=True)
if __name__ == "__main__": main()
