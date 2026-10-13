#!/usr/bin/env python3
"""Exp B2: same information as the original ESCS prompt (the 35 trait lines verbatim), framed as a survey record instead of
instructions. Each item scored independently: record + '<label>: ' -> next-token probabilities over the digits 1-7 (OLMo-3
tokenizes the space before a digit as its own token, so the prefix ends with ': '). Usage: score_record.py HF_REPO OUT.jsonl"""
import json, math, sys
from vllm import LLM, SamplingParams
repo, out = sys.argv[1], sys.argv[2]
R = [json.loads(l) for l in open("records.jsonl")]; ITEMS = json.load(open("items.json")); LAB = json.load(open("labels.json"))
llm = LLM(repo, dtype="bfloat16", enable_prefix_caching=True, max_model_len=4608, gpu_memory_utilization=0.90, seed=0)
sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
reqs, keys = [], []
for r in R:
    for it in ITEMS:
        reqs.append(r["record"] + LAB[it] + ": "); keys.append((r["pid"], it))
res = llm.generate(reqs, sp)
with open(out, "w") as f:
    for (pid, it), x in zip(keys, res):
        lp = x.outputs[0].logprobs[0]; mass = {}
        for tid, y in lp.items():
            t = (y.decoded_token or "").strip()
            if len(t) == 1 and t in "1234567": mass[int(t)] = mass.get(int(t), 0) + math.exp(y.logprob)
        tot = sum(mass.values()); ev = sum(k * v for k, v in mass.items()) / tot if tot > 0 else None
        f.write(json.dumps({"pid": pid, "item": it, "fmt": "record", "ev": ev, "argmax": max(mass, key=mass.get) if mass else None, "mass": tot}) + "\n")
print("DONE", repo, len(res), flush=True)
