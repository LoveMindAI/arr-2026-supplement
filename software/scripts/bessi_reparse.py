#!/usr/bin/env python3
"""Recover prediction calls whose saved raw response failed only because integers were quoted ("3"). Re-reads the retained
response file, applies bessi_run.py's validator rule (exact key set; each value an int, or an exact integer string, in range),
and if valid writes calls/<key>.json and predictions/<key>.json flagged reparsed_from_attempt + coerced_string_ints. No new API
call; values are unchanged. Uses the latest invalid attempt per key. Prints what it recovered."""
import glob, json, os, re
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bessi_run")
T = json.load(open(os.path.join(ROOT, "TARGETS.json")))
allowed = {t: {k: set(int(s) for s in q["scale"]) for q in T[t] for k in q["items"]} for t in T}
MODEL = {"sol": "openai/gpt-5.6-sol", "glm": "z-ai/glm-5.2", "flash": "google/gemini-3.5-flash-lite"}
def jp(text):
    text = text.strip()
    if text.startswith("```"):
        m = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text); text = m[1]
    return json.loads(text)
rec = 0; seen = {}
for f in sorted(glob.glob(os.path.join(ROOT, "attempts", "*.json"))):
    a = json.load(open(f))
    if a.get("valid") or "Non-integer" not in a.get("error", ""): continue
    k = a["key"]
    if os.path.exists(os.path.join(ROOT, "calls", k + ".json")): continue
    seen[k] = (f, a)                       # later attempt overwrites earlier
for k, (f, a) in seen.items():
    rf = f.replace(os.sep + "attempts" + os.sep, os.sep + "responses" + os.sep)
    data = json.load(open(rf)); text = data["choices"][0]["message"]["content"]
    pid, d, cond, route = k.split("__"); tgt = "bessi" if d == "bfi2_to_bessi" else "bfi2"; al = allowed[tgt]
    try:
        x = jp(text); assert isinstance(x, dict) and set(x) == set(al)
        x = {kk: (int(v) if isinstance(v, str) and re.fullmatch(r"[1-9]", v.strip()) else v) for kk, v in x.items()}
        assert all(type(v) == int and v in al[kk] for kk, v in x.items())
    except Exception as e:
        print("NOT recoverable:", k, str(e)[:80]); continue
    out = dict(a); out.update(valid=True, text=text, parsed=x, usage=data.get("usage"), delivered_model=data.get("model"), provider=data.get("provider"),
                              reparsed_from_attempt=a["attempt"], coerced_string_ints=True)
    json.dump(out, open(os.path.join(ROOT, "calls", k + ".json"), "w"), ensure_ascii=False, indent=2)
    json.dump({"id": None if pid == "no_person" else pid, "direction": d, "condition": cond, "route": route, "model": MODEL[route], "answers": x, "key": k,
               "prompt_sha256": a["prompt_sha256"], "coerced_string_ints": True, "reparsed_from_attempt": a["attempt"]},
              open(os.path.join(ROOT, "predictions", k + ".json"), "w"), ensure_ascii=False, indent=2)
    rec += 1; print("recovered", k, "from attempt", a["attempt"])
print("recovered", rec)
