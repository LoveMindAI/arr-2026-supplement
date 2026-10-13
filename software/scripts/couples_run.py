#!/usr/bin/env python3
"""Harmonies couple conditions. Builds each new prompt from the EXACT saved September numeric request for
that person and model (byte-identical personality block, genre question and request settings), inserting one block immediately
before "Answer the following as this person.":
  rel        "This person is in a romantic relationship."
  partner_p  + "Their partner's BFI-44 domain means, on the same scale:" + the partner's five means
  partner_t  + "Their partner answered the same question as follows (1 = strongly dislike, 7 = strongly like):" + partner's ratings
  other_t    identical wording to partner_t, but the ratings of a member of a different couple in the same fold (fixed seed)
Request bodies (model, provider pinning, reasoning, temperature, max_tokens, response_format) are copied from the saved requests.
Saves requests/ and responses/ under analysis/couples_run/; resumable; up to 2 retries for invalid JSON.
Usage: couples_run.py [--n 5] [--conc 16]"""
import argparse, json, os, random, re, sys, time, glob, hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np, requests
SEPT = "<DATA_ROOT>/04_SEPTEMBER_ARCHIVE/external_execution_20260916"
PKG = "<DATA_ROOT>/03_OCTOBER_EXPANSION"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "couples_run")
MARK = "Answer the following as this person."
MODELS = ["sol", "glm", "flash"]
CONDS = ["rel", "partner_p", "partner_t", "other_t"]
DOMS = [("extraversion", "E"), ("agreeableness", "A"), ("conscientiousness", "C"), ("neuroticism", "N"), ("openness", "O")]

R = np.load(f"{PKG}/offline_reproduction_candidate/data/expanded_reference.npz")
ids = list(R["harmonies__genres__ids"]); grp = list(R["harmonies__genres__groups"]); fold = list(R["harmonies__genres__fold"])
Y = R["harmonies__genres__truth"]; keys = list(R["harmonies__genres__keys"])
people = {p["id"]: p for p in json.load(open(f"{PKG}/expansion_execution/external_additions/analysis/twin_harmonies/PEOPLE.json"))}
partner = {}
for i, g in enumerate(grp):
    others = [j for j, h in enumerate(grp) if h == g and j != i]
    partner[i] = others[0] if others else None
rng = random.Random(20261009)
donor = {}
for i in range(len(ids)):                                   # a member of a different couple in the same fold
    pool = [j for j in range(len(ids)) if fold[j] == fold[i] and grp[j] != grp[i]]
    donor[i] = rng.choice(pool)

def ratings_block(j):
    d = {k: int(Y[j, n]) for n, k in enumerate(keys) if not np.isnan(Y[j, n])}
    return json.dumps(d, ensure_ascii=False)

def block(cond, i):
    b = "This person is in a romantic relationship."
    if cond == "partner_p":
        p = people[ids[partner[i]]]["domains"]
        b += "\nTheir partner's BFI-44 domain means, on the same scale:\n" + "\n".join(f"- {name}: {p[k]:.3f}" for name, k in DOMS)
    elif cond == "partner_t":
        b += "\nTheir partner answered the same question as follows (1 = strongly dislike, 7 = strongly like):\n" + ratings_block(partner[i])
    elif cond == "other_t":
        b += "\nTheir partner answered the same question as follows (1 = strongly dislike, 7 = strongly like):\n" + ratings_block(donor[i])
    return b + "\n\n"

def build(i, model, cond):
    base = json.load(open(f"{SEPT}/requests/{ids[i]}__numeric__harmonies__genres__{model}__attempt1.json"))
    t = base["messages"][0]["content"]; assert t.count(MARK) == 1
    body = dict(base); body["messages"] = [{"role": "user", "content": t.replace(MARK, block(cond, i) + MARK)}]
    return body

def parse(text):
    try:
        d = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        out = {k: int(d[k]) for k in keys if k in d and str(d[k]).lstrip("-").isdigit() and 1 <= int(d[k]) <= 7}
        return out if len(out) == len(keys) else None
    except Exception:
        return None

def call(job):
    i, model, cond = job; key = f"{ids[i]}__{cond}__{model}"; rf = os.path.join(OUT, "responses", key + ".json")
    if os.path.exists(rf) and json.load(open(rf)).get("valid"): return key, "cached", 0.0
    body = build(i, model, cond); json.dump(body, open(os.path.join(OUT, "requests", key + ".json"), "w"), ensure_ascii=False)
    cost = 0.0
    for attempt in range(3):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=180,
                              headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"}, json=body)
            j = r.json(); text = j["choices"][0]["message"]["content"] or ""; u = j.get("usage", {}); cost += float(u.get("cost") or 0)
            parsed = parse(text)
            rec = {"key": key, "person": ids[i], "cond": cond, "model": model, "attempt": attempt + 1, "valid": parsed is not None, "parsed": parsed,
                   "text": text, "cost_usd": cost, "delivered_model": j.get("model"), "provider": j.get("provider"), "usage": u,
                   "partner": ids[partner[i]] if partner[i] is not None else None, "donor": ids[donor[i]] if cond == "other_t" else None,
                   "prompt_sha256": hashlib.sha256(body["messages"][0]["content"].encode()).hexdigest()}
            json.dump(rec, open(rf, "w"), ensure_ascii=False)
            if parsed is not None: return key, "ok", cost
        except Exception as e:
            json.dump({"key": key, "valid": False, "error": str(e)[:300], "attempt": attempt + 1}, open(rf, "w")); time.sleep(3)
    return key, "INVALID", cost

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=0); ap.add_argument("--conc", type=int, default=16); a = ap.parse_args()
    for d in ("requests", "responses"): os.makedirs(os.path.join(OUT, d), exist_ok=True)
    idx = [i for i in range(len(ids)) if partner[i] is not None]
    if a.n: idx = idx[: a.n]
    jobs = [(i, m, c) for i in idx for m in MODELS for c in CONDS]
    tot = 0.0; bad = []
    with ThreadPoolExecutor(a.conc) as ex:
        for k, status, cost in ex.map(call, jobs):
            tot += cost
            if status == "INVALID": bad.append(k)
    print(f"jobs {len(jobs)}  spent ${tot:.3f}  invalid {len(bad)} {bad[:5]}")
    print("DONE")

if __name__ == "__main__":
    main()
