#!/usr/bin/env python3
"""Appendix tables generated from the analysis JSON (no hand-typed numbers): amplitude (ESCS 35 scores; five domains by cohort) and
the couple conditions. -> paper/tab_amplitude.tex, paper/tab_couples.tex"""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); P = os.path.join(ROOT, "paper")
A = lambda f: json.load(open(os.path.join(ROOT, "analysis", f)))
NAME = {"flash": "Gemini-3-Flash", "glm52": "GLM-5.2", "inkling": "Inkling", "mimopro": "MiMo-v2.5-pro", "minimaxm3": "MiniMax-M3",
        "qwen37": "Qwen3.7-plus", "sonnet5": "Claude-Sonnet-5", "terra": "GPT-5.6-Terra", "sol": "GPT-5.6-Sol", "glm": "GLM-5.2"}
NAME["flash_ext"] = "Gemini-3.5-Flash-Lite"
FORM = {"C-fac": "trait scores", "C-cardBF": "Big Five portrait", "C-cardFull": "full portrait", "numeric": "numbers", "structured": "sentences", "portrait": "portrait"}
T = A("trait_amplitude.json"); Ac = A("amplitude_cohorts.json"); RA = A("ridge_amplitude.json")   # ridge reference rows (Oct 10)
TRUNC = True    # lower bounds truncated at zero: an amplitude cannot be negative
lo_ = lambda x: max(x, 0.0) if TRUNC else x
fmt = lambda v: f"{v['amplitude_ratio']:.2f} [{lo_(v['ci95'][0]):.2f}, {v['ci95'][1]:.2f}] & {v['direction_r']:.2f}"
L = [r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}llrr@{}}\toprule", r"Model & Input & Amplitude [95\% CI] & Direction \\ \midrule"]
for k in sorted(T, key=lambda k: (k.split("|")[0], ["C-fac", "C-cardBF", "C-cardFull"].index(k.split("|")[1]))):
    m, c = k.split("|"); v = T[k]
    L.append(f"{NAME[m]} & {FORM[c]} & {v['amplitude_ratio_pooled']:.2f} [{lo_(v['ci95'][0]):.2f}, {v['ci95'][1]:.2f}] & {v['direction_r']:.2f} \\\\")
L += [r"\midrule", f"Ridge (other people) & trait scores & {fmt(RA['ESCS media (US) | 35 trait scores'])} \\\\"]
L += [r"\bottomrule\end{tabular}", r"\caption{Amplitude in ESCS with the 35 trait scores as predictors (109 media items, 467 people). 1 means the model spreads people apart by personality exactly as far as personality spreads their real answers; the ridge, fitted to 267 other participants' answers, is the reference. Intervals: delete-half jackknife over people; lower bounds truncated at zero.}", r"\label{tab:amp35}\end{table}", ""]
L += [r"\begin{table*}[p]\centering\scriptsize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llrr@{}}\toprule", r"Cohort & Model, input & Amplitude [95\% CI] & Direction \\ \midrule"]
for coh, d in Ac.items():
    first = True
    for k, v in d.items():
        if "__" in k: m, c = k.split("__"); lab = f"{NAME['flash_ext'] if m == 'flash' else NAME[m]}, {FORM[c]}"
        else: lab = NAME[k] + ", trait scores"
        L.append(f"{coh.replace('(DE)', '(German-speaking)') if first else ''} & {lab} & {fmt(v)} \\\\"); first = False
    _rv = RA.get('ESCS media (US) | five domains' if coh.startswith('ESCS') else coh)
    if _rv: L.append(f" & Ridge (other people) & {fmt(_rv)} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule\end{tabular}"
L += [r"\caption{Amplitude with the five domain means as predictors, every cohort. In ESCS the models received 35 trait scores; the five-domain regression is used here so that cohorts are comparable. Ridge rows: the human-trained ridge, fitted to other people's answers. Intervals: delete-half jackknife; lower bounds truncated at zero. The college cohort is in Table~\ref{tab:bessi}.}", r"\label{tab:amp5}\end{table*}"]
open(os.path.join(P, "tab_amplitude.tex"), "w").write("\n".join(L) + "\n")

Cc = A("couples_conditions.json")
CN = {"numeric": "own personality only", "rel": "+ ``in a relationship''", "partner_p": "+ partner's personality", "partner_t": "+ partner's ratings", "other_t": "+ a stranger's ratings, labeled as the partner's"}
L = [r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{2pt}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lrrr@{}}\toprule", r"Prompt adds & GPT-5.6-Sol & GLM-5.2 & Flash-Lite \\ \midrule"]
for c in ["numeric", "rel", "partner_p", "partner_t", "other_t"]:
    L.append(CN[c] + " & " + " & ".join(f"{Cc['accuracy'][f'{m}|{c}']['mean']:.3f}" for m in ["sol", "glm", "flash"]) + r" \\")
L.append(r"\midrule partner's ratings used directly & \multicolumn{3}{c}{" + f"{Cc['partner_alone']['mean']:.3f}" + r"} \\")
L.append(r"copying: shown partner's ratings & " + " & ".join(f"{Cc['copying'][f'{m}|partner_t']['mean']:.2f}" for m in ["sol", "glm", "flash"]) + r" \\")
L.append(r"copying: shown stranger's ratings & " + " & ".join(f"{Cc['copying'][f'{m}|other_t (copies the stranger)']['mean']:.2f}" for m in ["sol", "glm", "flash"]) + r" \\")
L += [r"\bottomrule\end{tabular}}", f"\\caption{{Harmonies couple conditions ({Cc['n_people']} people, {Cc['n_couples']} couples). Rows above the line: distinctive accuracy. Copying: within-person Spearman between the prediction's and the shown ratings' departures from the average; real partners resemble each other at {Cc['real_partner_concordance']['mean']:.2f}. Intervals in the replication files.}}", r"\label{tab:couples}\end{table}"]
open(os.path.join(P, "tab_couples.tex"), "w").write("\n".join(L) + "\n")
print("tables written")

# ---------------- BESSI college cohort, both directions
Bs = A("bessi_results.json")
NM = {"sol": "GPT-5.6-Sol", "glm": "GLM-5.2", "flash": "Flash-Lite", "ridge": "ridge (113 students)"}
DN = {"bfi2_to_bessi": "Personality (BFI-2) $\\rightarrow$ 192 skill items", "bessi_to_bfi2": "Skills (BESSI) $\\rightarrow$ 60 BFI-2 items"}
L = [r"\begin{table*}[tp]\centering\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llrrrrr@{}}\toprule",
     r"Direction & Predictor, input & Distinctive [95\% CI] & Wrong person & Amplitude [95\% CI] & Direction & MAE \\ \midrule"]
for d, o in Bs["directions"].items():
    first = True
    for k, e in o["predictors"].items():
        m, c = k.split("|"); lab = NM[m] + ("" if m == "ridge" else ", " + {"numeric": "numbers", "structured": "sentences", "portrait": "portrait", "no_person": "no person"}[c])
        amp = e.get("amplitude"); a = f"{amp['amplitude_ratio']:.2f} [{lo_(amp['ci95'][0]):.2f}, {amp['ci95'][1]:.2f}] & {amp['direction_r']:.2f}" if amp else "-- & --"
        L.append(f"{DN[d] if first else ''} & {lab} & {e['distinctive']['mean']:.3f} [{e['distinctive']['ci95'][0]:.3f}, {e['distinctive']['ci95'][1]:.3f}] & {e['wrong_person']['mean']:.3f} & {a} & {e['mae']:.2f} \\\\")
        first = False
    L.append(r"\midrule")
L[-1] = r"\bottomrule\end{tabular}"
L += [f"\\caption{{College cohort ({Bs['n_eval']} evaluated students; {Bs['n_reference']} reference students for averages and the ridge). Distinctive accuracy, wrong-person reassignment and amplitude as in \\S\\ref{{sec:data}}; amplitude uses the five source domains as predictors. Intervals: person bootstrap (distinctive), delete-half jackknife (amplitude; lower bounds truncated at zero).}}", r"\label{tab:bessi}\end{table*}"]
open(os.path.join(P, "tab_bessi.tex"), "w").write("\n".join(L) + "\n")
print("tab_bessi written")
