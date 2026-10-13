#!/usr/bin/env python3
"""Build the inputs for Extension B (OLMo-3 stages) from each ESCS person's ORIGINAL 35-trait-score prompt (the same prompts the
paper's ESCS predictions used; one file per person, <id>__original.txt, under PROMPTS). Writes, next to the scoring scripts:
  prompts.jsonl  {pid, prompt}       the original prompt, unchanged (instruction framing; score_ladder.py)
  items.json     the 109 item ids, in prompt order
  records.jsonl  {pid, record}       the same 35 trait lines as a survey record, no instructions (score_record.py)
  labels.json    item id -> item label as printed in the prompt
Then run on a GPU host: score_ladder.py <hf_repo> out/<tag>.jsonl for each stage (LADDER below), score_record.py <hf_repo> ...,
and analyze_B.py / analyze_B2.py locally. Settings used: 7B stages default (TMPL=allenai/Olmo-3-7B-Instruct, MAXLEN=8192,
GPU_UTIL=0.90); 32B stages TMPL=allenai/Olmo-3.1-32B-Instruct MAXLEN=4608 GPU_UTIL=0.92; vLLM 0.31.0 on one 80 GB H100."""
import json, os, sys
PROMPTS = os.environ.get("PROMPTS", "<DATA_ROOT_SEPT>/order_campaign/private_prompts")
LADDER_7B = ["allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct-SFT", "allenai/Olmo-3-7B-Instruct-DPO", "allenai/Olmo-3-7B-Instruct"]
LADDER_32B = ["allenai/Olmo-3-1125-32B", "allenai/Olmo-3.1-32B-Instruct-SFT", "allenai/Olmo-3.1-32B-Instruct-DPO", "allenai/Olmo-3.1-32B-Instruct"]
RECORD_RUNS = ["allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct-SFT", "allenai/Olmo-3-7B-Instruct", "allenai/Olmo-3-1125-32B", "allenai/Olmo-3.1-32B-Instruct"]
ids = sorted(int(f.split("__")[0]) for f in os.listdir(PROMPTS) if f.endswith("__original.txt"))
first = open(f"{PROMPTS}/{ids[0]}__original.txt").read().split("\n")
items = [l.split(":", 1)[0] for l in first if ":" in l and l.split(":", 1)[0].rstrip("0123456789") in ("music", "books", "movies", "tv")]
labels = {l.split(":", 1)[0]: l.split(":", 1)[1].strip() for l in first if ":" in l and l.split(":", 1)[0] in items}
with open("prompts.jsonl", "w") as fp, open("records.jsonl", "w") as fr:
    for p in ids:
        txt = open(f"{PROMPTS}/{p}__original.txt").read(); fp.write(json.dumps({"pid": p, "prompt": txt}) + "\n")
        traits = [l for l in txt.split("\n") if l.startswith("- ")]; assert len(traits) == 35
        rec = ("Survey record\n\nPsychometric profile of respondent:\n" + "\n".join(traits) +
               "\n\nMedia enjoyment ratings given by this respondent (1 = strongly dislikes, 4 = neutral or mixed, 7 = strongly enjoys):\n")
        fr.write(json.dumps({"pid": p, "record": rec}) + "\n")
json.dump(items, open("items.json", "w")); json.dump(labels, open("labels.json", "w"))
with open("LADDER.txt", "w") as f: f.write("\n".join(LADDER_7B) + "\n")
with open("LADDER32.txt", "w") as f: f.write("\n".join(LADDER_32B) + "\n")
print(len(ids), "people;", len(items), "items")
