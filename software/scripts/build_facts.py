#!/usr/bin/env python3
"""Panorama facts file: every number the redraft may use, pulled from its source file, with the source beside it.
Rule: numbers go into prose only from this file. Re-run after any analysis changes.
Sources: the original generated v3 macros (results_macros.tex, "generated from the frozen analysis artifacts"), the original external-cohort
controls (CONTROLS_SUMMARY.json), the two human-reader reports, and this revision's Oct 9 analyses in analysis/.
-> paper_notes/FACTS_v1.md and FACTS_v1.json"""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
PKG = "<DATA_ROOT>"
MACROS = os.path.join(ROOT, "paper_notes", "v3_source", "results_macros.tex")   # the original v3 source zip (01_CURRENT_MANUSCRIPT), extracted Oct 11
B = "<DATA_ROOT_SEPT>"
F = []
def add(sec, label, val, src): F.append((sec, label, val, src))
A = lambda f: json.load(open(os.path.join(ROOT, "analysis", f)))
ci = lambda d, k="mean": f"{d[k]:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"

# ---- v3 manuscript macros (original; generated)
for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(MACROS).read()):
    add("v3 macros (original)", name, val, "01_CURRENT_MANUSCRIPT source zip: results_macros.tex")

# ---- readback, external cohorts (original, October expansion: Twin 200, Harmonies 198, Indonesia 150). The September
#      CONTROLS_SUMMARY.json (Twin 150) is superseded and NOT used.
for x in json.load(open(f"{PKG}/03_OCTOBER_EXPANSION/expansion_execution/external_additions/analysis/RESULTS.json"))["readbacks"]:
    add("readback (original, October)", f"{x['cohort']} / {x['mode']} / {x['subset']}", f"mean domain rank r {x['mean_domain_rank']:.3f}; MAE {x['mae']:.3f}; n {x['n']}",
        "03_OCTOBER_EXPANSION/expansion_execution/external_additions/analysis/RESULTS.json")

# ---- held-out human readback (original, Oct 10; NATIVE_v2 panel): 14 readers x 2 portraits, 14 held-out people (7 Twin, 7 Harmonies)
_HD = f"{PKG}/02_HUMAN_VALIDATION_DESIGN/NATIVE_v2_20261009/native_blind/analysis_private"
_LS = json.load(open(f"{_HD}/LUNA_SOURCE_COMPARISON_SUPPLEMENT_PRIVATE.json")); _DS = json.load(open(f"{_HD}/DESCRIPTIVE_READBACK_SUMMARY_PRIVATE.json"))
add("held-out human readback (original)", "people / readers / reader-portrait rows / estimates (numeric, unknown, unparsed)",
    f"{_DS['source_people']} / {_DS['readers']} / {_DS['reader_portrait_rows']} / {_DS['expected_ratings']} ({_DS['rating_status_counts']['numeric']}, {_DS['rating_status_counts']['unknown']}, {_DS['rating_status_counts']['unparsed']})",
    "02_HUMAN_VALIDATION_DESIGN/NATIVE_v2_20261009/native_blind/analysis_private/DESCRIPTIVE_READBACK_SUMMARY_PRIVATE.json")
for _m, _lab in [("human_mean", "human means"), ("continuous", "Luna direct"), ("questionnaire", "Luna questionnaire")]:
    _rs = {d: v[_m]["spearman_source"] for d, v in _LS["domains"].items()}; _ma = {d: v[_m]["mae_source"] for d, v in _LS["domains"].items()}; _bi = {d: v[_m]["signed_bias_estimate_minus_source"] for d, v in _LS["domains"].items()}
    add("held-out human readback (original)", f"{_lab} vs SOURCE: Spearman by domain; mean", ", ".join(f"{d} {r:.3f}" for d, r in _rs.items()) + f"; mean {sum(_rs.values()) / 5:.3f}", "LUNA_SOURCE_COMPARISON_SUPPLEMENT_PRIVATE.json")
    add("held-out human readback (original)", f"{_lab} vs SOURCE: MAE mean over domains; N signed bias", f"{sum(_ma.values()) / 5:.3f}; N {_bi['N']:+.3f}", "LUNA_SOURCE_COMPARISON_SUPPLEMENT_PRIVATE.json")
    add("held-out human readback (original)", f"{_lab} vs SOURCE: signed bias by domain (estimate minus source)", ", ".join(f"{d} {v:+.3f}" for d, v in _bi.items()), "LUNA_SOURCE_COMPARISON_SUPPLEMENT_PRIVATE.json")
for _d, _v in _DS["domains"].items():
    _b = _v["source"]["target_bootstrap"]["percentile_95_interval"]; _w = _v["within_cohort"]
    add("held-out human readback (original)", f"{_d}: human vs source 95% person-bootstrap; within-cohort Spearman (Twin, Harmonies)", f"[{_b[0]:.3f}, {_b[1]:.3f}]; {_w['twin']['spearman']:.3f}, {_w['harmonies']['spearman']:.3f}", "DESCRIPTIVE_READBACK_SUMMARY_PRIVATE.json")

# ---- raw (normative-inclusive) fit, ESCS media, computed here from the original item-prediction file (v3 text: "approximately 0.31 to 0.35")
import numpy as _np, pandas as _pd
from scipy.stats import spearmanr as _sp
_P = _pd.read_csv(f"{PKG}/03_OCTOBER_EXPANSION/offline_reproduction_candidate/data/core/MEDIA469_ITEM_PREDICTIONS.csv")
_g = _P[(_P.model == "flash") & (_P.condition == "C-fac")]
add("raw fit ESCS (revision, from the original predictions)", "population-average profile: mean within-person Spearman", f"{_np.nanmean([_sp(h.disjoint_norm, h.truth).correlation for _, h in _g.groupby('escs_id')]):.3f}", "MEDIA469_ITEM_PREDICTIONS.csv (disjoint_norm vs truth)")
_rr = {(m, c): _np.nanmean([_sp(h.prediction, h.truth).correlation for _, h in gg.groupby("escs_id")]) for (m, c), gg in _P.groupby(["model", "condition"])}
add("raw fit ESCS (revision, from the original predictions)", "8 models x 3 conditions: range of mean within-person Spearman", f"{min(_rr.values()):.3f} to {max(_rr.values()):.3f}", "MEDIA469_ITEM_PREDICTIONS.csv")

# ---- v3 hand-typed results sentences (original), verbatim with line numbers, for numbers that have no macro. the original analysis verifies independently.
_V3 = open(MACROS.replace("results_macros.tex", "compact_main.tex")).read().splitlines()
for _n in [149, 176, 181, 203, 216, 233, 745, 218, 223, 227, 229, 235, 239, 241, 599, 601, 669, 701, 703, 705, 722, 725, 730, 733, 736, 754, 755, 756, 763, 765, 768, 773, 775, 777]:
    add("v3 results text (original, verbatim)", f"compact_main.tex line {_n}", _V3[_n - 1].strip(), "01_CURRENT_MANUSCRIPT source zip: compact_main.tex")
_MR = open(f"{B}/overnight_internal_20260916/MORNING_READOUT.md").read()
add("similar-person matching ESCS (original)", "MORNING_READOUT.md first 1500 characters", _MR[:1500].replace("\n", " "), "overnight_internal_20260916/MORNING_READOUT.md")

# ---- numbers quoted from cited papers, verified by the lit room (source + page in its review files)
add("cited numbers (lit room)", "Vigl et al. 2026 couple similarity, item-standardized profiles", ".32 (Table 1, p. 6)", "lit/BIB_REVIEW_5_HARMONIES.md")

# ---- human readers (the original two studies)
r1 = open(f"{B}/next_internal_extensions_20260916/cloudresearch_repeated_readers/results_20260916/REPORT.md").read()
m = re.search(r"\| Humans: mean of two readers \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", r1)
add("human readers (original)", "ESCS portraits: Spearman / CCC / MAE (mean of two readers; 12 readers, 18 portraits)", "/".join(m.groups()), "cloudresearch_repeated_readers/results_20260916/REPORT.md")
r2 = open(f"{B}/prolific_bfi44_20260916/FINAL_READOUT.md").read()
m = re.search(r"\| combined \| (\d+) \| ([\d.]+) \| ([\d.]+) \|", r2)
add("human readers (original)", "BFI-44 development portraits: n portraits / Spearman / MAE", "/".join(m.groups()), "prolific_bfi44_20260916/FINAL_READOUT.md")

# ---- coupling: how much of taste personality explains (cross-validated R^2, 35 IPIP-NEO scores), people vs models (revision)
Tc = A("trait_coupling.json")["summary"]
for k, v in Tc.items():
    add("coupling ESCS (revision)", k, f"people CV R^2 mean {v['human_r2_mean']:.3f} (median {v['human_r2_median']:.3f}); model {v['model_r2_mean']:.3f} (median {v['model_r2_median']:.3f}); items model>people {v['items_model_gt_human']}/{v['n_items']}; direction r {v['direction_r_mean']:.2f}", "analysis/trait_coupling.json")

# ---- overall spread across people (compression) vs personality-driven spread (amplitude): ESCS from trait_coupling summary; external from the original appendix text
for k, v in Tc.items():
    add("overall spread ESCS (revision)", k, f"mean item SD across people, model/people = {v['sd_model_mean'] / v['sd_human_mean']:.2f} ({v['sd_model_mean']:.2f}/{v['sd_human_mean']:.2f})", "analysis/trait_coupling.json")
_NM = open(MACROS.replace("results_macros.tex", "new_methods_appendix.tex")).read()
_i = _NM.index("\\paragraph{Between-person dispersion.}")
add("overall spread external (original, verbatim)", "new_methods_appendix.tex: Between-person dispersion", _NM[_i:_i + 700].replace("\n", " "), "01_CURRENT_MANUSCRIPT source zip: new_methods_appendix.tex")

# ---- amplitude (revision)
T = A("trait_amplitude.json")
for k, v in T.items():
    add("amplitude ESCS 35 scores (revision)", k, f"x{v['amplitude_ratio_pooled']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]; direction r {v['direction_r']:.2f}; per-item median x{v['per_item_ratio_median']:.2f}", "analysis/trait_amplitude.json")
Ac = A("amplitude_cohorts.json")
for coh, d in Ac.items():
    for k, v in d.items():
        add("amplitude five domains (revision)", f"{coh} | {k}", f"x{v['amplitude_ratio']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]; direction r {v['direction_r']:.2f}", "analysis/amplitude_cohorts.json")

# ---- ridge amplitude: the reference point (revision, Oct 10; scripts/ridge_amplitude.py)
for k, v in A("ridge_amplitude.json").items():
    add("ridge amplitude (revision)", k, f"x{v['amplitude_ratio']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]; direction r {v['direction_r']:.2f}; n {v['n_people']}", "analysis/ridge_amplitude.json")

# ---- where right (revision)
W = A("where_right.json")
for k, v in W["summary"].items():
    add("where right (revision)", k, f"rho(item accuracy, trait signal) {v['rho_item_accuracy_vs_trait_signal']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]; rho vs ridge {v['rho_vs_ridge_item_accuracy']:.2f}; high/low half {v['mean_item_accuracy_high_signal_half']:.3f}/{v['mean_item_accuracy_low_signal_half']:.3f}", "analysis/where_right.json")
add("where right (revision)", "ridge item accuracy high/low half", f"{W['ridge_accuracy_high_low'][0]:.3f}/{W['ridge_accuracy_high_low'][1]:.3f}", "analysis/where_right.json")
_rr = W["ridge_rho_item_accuracy_vs_trait_signal"]
add("where right (revision)", "ridge rho(item accuracy, trait signal)", f"{_rr['rho']:.2f} [{_rr['ci95'][0]:.2f}, {_rr['ci95'][1]:.2f}]; items {_rr['n_items']}", "analysis/where_right.json")
# ---- repeatability: the fixed-output source behind v3's hand-typed .942-.955 / .902-.930
for r in json.load(open(f"{PKG}/03_OCTOBER_EXPANSION/offline_reproduction_candidate/expected/repeat.json"))["results"]:
    add("repeatability (original, fixed output)", r["model"], f"same person raw {r['raw_same_person']['mean']:.3f}; shared item profile removed: same person {r['shared_item_removed_same_person']['mean']:.3f}, other person {r['shared_item_removed_other_person']['mean']:.3f}; three-draw distinctive fidelity {r['human_fidelity_of_mean']['distinctive']['mean'] if isinstance(r['human_fidelity_of_mean']['distinctive'], dict) else r['human_fidelity_of_mean']['distinctive']:.3f}", "03_OCTOBER_EXPANSION/offline_reproduction_candidate/expected/repeat.json")

# ---- the ladder, recomputed from the original saved files (reproduces v3 tables exactly, Oct 9)
Ld = A("ladder.json")
for coh, v in Ld.items():
    cells = {k: d["mean"] for k, d in v["cells"].items()}
    add("ladder (revision, recomputed)", f"{coh}: LLM cells range", f"{min(cells.values()):+.3f} to {max(cells.values()):+.3f} (n {v['n_people']}, items {v['n_items']})", "analysis/ladder.json")
    for k, d in v["cells"].items():
        add("ladder (revision, recomputed)", f"{coh} / {k}", ci(d) + (f"; wrong-person {d['wrong_person']:+.3f}; gap {ci(d['gap'])}" if "gap" in d else ""), "analysis/ladder.json")
    for r in ["ridge", "earlier_self", "partner_real"]:
        if r in v: add("ladder (revision, recomputed)", f"{coh}: {r}", ci(v[r]), "analysis/ladder.json")

# ---- couples (revision; real concordance replicates the original COUPLE_RESULTS.json .290)
Cp = A("couples.json")
add("couples (revision)", "real partners' distinctive taste concordance / strangers", f"{Cp['real']['partners']:.3f} [{Cp['real']['ci95_partners'][0]:.3f}, {Cp['real']['ci95_partners'][1]:.3f}] / {Cp['real']['strangers']:.3f}; couples {Cp['real']['n_couples']}", "analysis/couples.json")
for k, v in Cp["models"].items():
    add("couples (revision)", f"model predicted-partner concordance | {k}", f"{v['partners']:.3f} [{v['ci95_partners'][0]:.3f}, {v['ci95_partners'][1]:.3f}]; strangers {v['strangers']:.3f}; distinctive accuracy {v['distinctive_accuracy']:.3f}", "analysis/couples.json")
add("couples (revision)", "partner's real ratings as predictor (raw json)", json.dumps(Cp["partner_as_predictor"])[:300], "analysis/couples.json")
Cc = A("couples_conditions.json")
add("couple conditions (revision)", "people / couples", f"{Cc['n_people']} / {Cc['n_couples']}", "analysis/couples_conditions.json")
add("couple conditions (revision)", "partner's real ratings used directly (distinctive accuracy)", ci(Cc["partner_alone"]), "analysis/couples_conditions.json")
add("couple conditions (revision)", "real partner concordance", ci(Cc["real_partner_concordance"]), "analysis/couples_conditions.json")
for sec in ["accuracy", "contrasts", "vs_partner_alone", "concordance", "copying"]:
    for k, v in Cc[sec].items(): add(f"couple conditions: {sec} (revision)", k, ci(v), "analysis/couples_conditions.json")

# ---- BESSI college cohort (revision), when present
p = os.path.join(ROOT, "analysis", "bessi_results.json"); st = os.path.join(ROOT, "bessi_run", "RUN_STATE.json")
if os.path.exists(p) and os.path.exists(st) and json.load(open(st)).get("status") == "complete":
    Bs = json.load(open(p))
    _SR = json.load(open(os.path.join(ROOT, "bessi_run", "SELECTION_RECEIPT.json")))
    add("BESSI (revision)", "complete cases / eval / reference / age range", f"{_SR['complete_cases']} / {_SR['eval_n']} / {_SR['reference_n']} / {_SR['age_range']}", "bessi_run/SELECTION_RECEIPT.json")
    add("BESSI (revision)", "evaluation / reference people", f"{Bs['n_eval']} / {Bs['n_reference']}", "analysis/bessi_results.json")
    for d, o in Bs["directions"].items():
        add("BESSI (revision)", f"{d}: items; ridge alpha", f"{o['n_items']}; {o['ridge_alpha']}", "analysis/bessi_results.json")
        if "readback" in o: add("BESSI (revision)", f"{d}: Luna readback Spearman by domain", json.dumps({k: round(v, 3) for k, v in o["readback"].items()}), "analysis/bessi_results.json")
        _src = "bfi2_domains" if d == "bfi2_to_bessi" else "bessi_domains"
        _rbs = [json.load(open(f)) for f in sorted(__import__("glob").glob(os.path.join(ROOT, "bessi_run", "readbacks", f"*__{d}.json")))]
        _TR = json.load(open(os.path.join(ROOT, "bessi_run", "PRIVATE_TRUTH.json")))
        if _rbs:
            _keys = list(_rbs[0]["domains"].keys())
            add("BESSI (revision)", f"{d}: Luna readback minus source, mean by domain", ", ".join(f"{k} {sum(r['domains'][k] - _TR[r['id']][_src][k] for r in _rbs) / len(_rbs):+.2f}" for k in _keys), "bessi_run/readbacks + PRIVATE_TRUTH.json")
        if "readback" in o:
            _rv = [v for k, v in o["readback"].items() if k != "n"]
            add("BESSI (revision)", f"{d}: Luna readback mean domain Spearman", f"{sum(_rv) / len(_rv):.3f}", "analysis/bessi_results.json")
        for k, e in o["predictors"].items():
            s = f"distinctive {ci(e['distinctive'])}; wrong-person {e['wrong_person']['mean']:+.3f}; gap {ci(e['binding_gap'])}; MAE {e['mae']:.3f}; raw profile r {e['raw_profile_r']:.3f}; n {e['n_people']}"
            if "amplitude" in e: a = e["amplitude"]; s += f"; amplitude x{a['amplitude_ratio']:.2f} [{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}], direction r {a['direction_r']:.2f}"
            add(f"BESSI {d} (revision)", k, s, "analysis/bessi_results.json")
            if "domains" in e:
                add(f"BESSI {d} domains (revision)", k, "; ".join(f"{t}: r {x['r_with_true']:.2f}, SD x{x['sd_ratio']:.2f}" for t, x in e["domains"].items()), "analysis/bessi_results.json")
        # flattering shift: predicted minus true domain mean, with negative emotionality (N) sign-flipped so that + = more desirable
        _cells = [k for k in o["predictors"] if not k.startswith("ridge") and not k.endswith("no_person")]
        for _form in ["numeric", "structured", "portrait"]:
            _sh = [(x["mean_pred"] - x["mean_true"]) * (-1 if t == "N" else 1) for k in _cells if k.endswith(_form) for t, x in o["predictors"][k]["domains"].items()]
            add(f"BESSI {d} (revision)", f"desirable-direction mean shift, {_form} (N flipped; avg over domains and models; range)", f"{sum(_sh)/len(_sh):+.2f} ({min(_sh):+.2f} to {max(_sh):+.2f})", "analysis/bessi_results.json")
        for _form in ["numeric", "portrait"]:
            _fc = [k for k in _cells if k.endswith(_form)]
            _per = {t: sum((o["predictors"][k]["domains"][t]["mean_pred"] - o["predictors"][k]["domains"][t]["mean_true"]) * (-1 if t == "N" else 1) for k in _fc) / len(_fc) for t in o["predictors"][_fc[0]]["domains"]}
            add(f"BESSI {d} (revision)", f"desirable-direction mean shift by domain, {_form} (avg over models)", ", ".join(f"{t} {v:+.2f}" for t, v in _per.items()), "analysis/bessi_results.json")
        _all = [x["r_with_true"] for k in _cells for x in o["predictors"][k]["domains"].values()]; _rid = [x["r_with_true"] for x in o["predictors"]["ridge|reference"]["domains"].values()]
        add(f"BESSI {d} (revision)", "domain-level r with true target domains: models range; ridge range", f"{min(_all):.2f} to {max(_all):.2f}; ridge {min(_rid):.2f} to {max(_rid):.2f}", "analysis/bessi_results.json")
        add(f"BESSI {d} (revision)", "human cross-instrument r (source domain x true target domain)", json.dumps({s_: {t: round(v, 2) for t, v in row.items()} for s_, row in o["human_cross_r"].items()}), "analysis/bessi_results.json")
else:
    add("BESSI (revision)", "status", "run not complete; re-run build_facts.py after bessi_analysis.py", "bessi_run/RUN_STATE.json")

# ---- totals (revision, Oct 10): evaluated people across the five cohorts. ESCS/Twin/Harmonies/Indonesia from ladder.json
#      (Twin policy and risk are the same 200 people, counted once; ESCS 467 = the 469 portrait people with all 109 media
#      answers, the original complete_truth filter); college from bessi_results.json.
_tot = {coh: v["n_people"] for coh, v in Ld.items() if not coh.startswith("Twin risk")}
if "Bs" in globals(): _tot["College (US)"] = Bs["n_eval"]
add("totals (revision)", "evaluated people across the cohorts (" + ", ".join(f"{k} {n}" for k, n in _tot.items()) + ")", f"{sum(_tot.values()):,}", "analysis/ladder.json + analysis/bessi_results.json")

out = os.path.join(ROOT, "paper_notes", "FACTS_v1.md"); secs = []
with open(out, "w") as f:
    f.write("# Panorama FACTS_v1: every number the redraft may use, with its source\nGenerated by scripts/build_facts.py. Do not edit by hand; re-run it.\n")
    for s, l, v, src in F:
        if s not in secs: secs.append(s); f.write(f"\n## {s}\n| fact | value | source |\n|---|---|---|\n")
        f.write(f"| {l.replace('|', ' / ')} | {str(v).replace('|', '/')} | `{src}` |\n")
json.dump([{"section": s, "fact": l, "value": v, "source": src} for s, l, v, src in F], open(out.replace(".md", ".json"), "w"), indent=1)
print(f"{len(F)} facts -> {out}")
