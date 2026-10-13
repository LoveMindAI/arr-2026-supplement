#!/usr/bin/env python3
"""Exp B worker (runs on the GPU pod). For one checkpoint: every person's ORIGINAL ESCS prompt, verbatim, in two wrappers.
  raw : the prompt as a plain document, then '\n{"<item>": '
  chat: the prompt as the user turn in OLMo-3's instruct chat template (same template text for every stage, base included),
        then the assistant turn begins '{"<item>": '
Each item is scored independently: one next-token step, top-20 logprobs, probability mass over the tokens "1".."7".
Expected rating = sum k p_k / sum p_k; we also keep argmax and the captured mass (format check). (OLMo-3 tokenizes the space before a digit as its own token, so the prefix ends with ": " and the next token is the bare digit; verified Oct 12.) Prefix caching shares the
person's prompt across the 109 items. Usage: score_ladder.py HF_REPO OUT.jsonl"""
import json, math, sys, os
from vllm import LLM, SamplingParams
repo, out = sys.argv[1], sys.argv[2]
P = [json.loads(l) for l in open("prompts.jsonl")]; ITEMS = json.load(open("items.json"))
from transformers import AutoTokenizer
tmpl = AutoTokenizer.from_pretrained(os.environ.get("TMPL", "allenai/Olmo-3-7B-Instruct"))   # one chat template for all stages of a ladder
llm = LLM(repo, dtype="bfloat16", enable_prefix_caching=True, max_model_len=int(os.environ.get("MAXLEN", 8192)), gpu_memory_utilization=float(os.environ.get("GPU_UTIL", 0.90)), seed=0)
sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
reqs, keys = [], []
for p in P:
    chat = tmpl.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True)
    for it in ITEMS:
        reqs.append(p["prompt"] + '\n{"' + it + '": '); keys.append((p["pid"], it, "raw"))
        reqs.append(chat + '{"' + it + '": '); keys.append((p["pid"], it, "chat"))
res = llm.generate(reqs, sp)
with open(out, "w") as f:
    for (pid, it, fmt), r in zip(keys, res):
        lp = r.outputs[0].logprobs[0]
        mass = {}
        for tid, x in lp.items():
            t = (x.decoded_token or "").strip()
            if t in "1234567" and len(t) == 1: mass[int(t)] = mass.get(int(t), 0) + math.exp(x.logprob)
        tot = sum(mass.values())
        ev = sum(k * v for k, v in mass.items()) / tot if tot > 0 else None
        am = max(mass, key=mass.get) if mass else None
        f.write(json.dumps({"pid": pid, "item": it, "fmt": fmt, "ev": ev, "argmax": am, "mass": tot}) + "\n")
print("DONE", repo, len(res), flush=True)
