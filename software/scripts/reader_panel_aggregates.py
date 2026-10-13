#!/usr/bin/env python3
"""Aggregate-only summary of the human reader panels, for the public reproducibility packet. Reads the private domain-level summaries and keeps ONLY domain-level statistics:
the number of people, Spearman with the source scores, MAE, signed bias, the pooled 95% interval, and within-cohort point estimates.
Nothing per reader or per portrait is copied. The September panels are copied from their FACTS rows (sources: the panels' own reports).
-> analysis/reader_panels_aggregate.json"""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
H = "<DATA_ROOT>/02_HUMAN_VALIDATION_DESIGN/NATIVE_v2_20261009/native_blind/analysis_private"
D = json.load(open(f"{H}/DESCRIPTIVE_READBACK_SUMMARY_PRIVATE.json"))["domains"]
S = json.load(open(f"{H}/LUNA_SOURCE_COMPARISON_SUPPLEMENT_PRIVATE.json"))["domains"]
KEEP = ("n", "spearman_source", "mae_source", "signed_bias_estimate_minus_source")
out = {"held_out_panel": {"people": 14, "readers_per_portrait": 2, "domains": {}}}
for dom in ("A", "C", "E", "N", "O"):
    d = D[dom]
    out["held_out_panel"]["domains"][dom] = {
        "readers_vs_source": {k: S[dom]["human_mean"][k] for k in KEEP},
        "readers_vs_source_pooled_95ci": d["source"]["target_bootstrap"]["percentile_95_interval"],
        "readers_vs_source_within_cohort": {c: {"people": v["target_people"], "spearman": v["spearman"]} for c, v in d["within_cohort"].items()},
        "automated_direct_vs_source": {k: S[dom]["continuous"][k] for k in KEEP},
        "automated_questionnaire_vs_source": {k: S[dom]["questionnaire"][k] for k in KEEP}}
sp = [out["held_out_panel"]["domains"][d]["readers_vs_source"]["spearman_source"] for d in "ACENO"]
out["held_out_panel"]["readers_vs_source_mean_spearman"] = sum(sp) / len(sp)
F = json.load(open(os.path.join(ROOT, "paper_notes", "FACTS_v1.json")))
out["september_panels_from_facts"] = [r for r in F if r["section"].startswith("human readers")]
json.dump(out, open(os.path.join(ROOT, "analysis", "reader_panels_aggregate.json"), "w"), indent=1)
print(f"held-out panel mean Spearman {out['held_out_panel']['readers_vs_source_mean_spearman']:.4f}; september rows {len(out['september_panels_from_facts'])}")
