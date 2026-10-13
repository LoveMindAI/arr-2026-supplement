#!/usr/bin/env python3
"""Panorama figures (house style: serif, small, a finding sentence as each panel title).
  fig_teaser.pdf    Figure 1, a COMPOSITE example (labeled in the caption): two near-identical personalities -> near-identical predicted
                    tastes -> different real tastes. Values are illustrative, not participants'.
  fig_amplitude.pdf Figure 2: (a) ESCS, eight models, 35 trait scores, three input forms; (b) five domains, every cohort.
  fig_ladder.pdf    Figure 3: distinctive accuracy of every predictor in every cohort (analysis/ladder.json, couples_conditions.json).
Data: analysis/{trait_amplitude,amplitude_cohorts,ladder,couples_conditions}.json -> paper/figs/"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); FIG = os.path.join(ROOT, "paper", "figs")
A = lambda f: json.load(open(os.path.join(ROOT, "analysis", f)))
C = {"person": "#1f7a72", "person2": "#b7791f", "model": "#6b4fa0", "model_l": "#a68bd1", "ridge": "#6d6d6d", "partner": "#b7791f"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 7.5})
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
NAME = {"flash": "Gemini-3-Flash", "glm52": "GLM-5.2", "inkling": "Inkling", "mimopro": "MiMo-v2.5-pro", "minimaxm3": "MiniMax-M3",
        "qwen37": "Qwen3.7-plus", "sonnet5": "Claude-Sonnet-5", "terra": "GPT-5.6-Terra"}

# ---------------- Figure 1: composite teaser
fig, ax = plt.subplots(1, 3, figsize=(6.5, 1.9), gridspec_kw={"wspace": 0.55, "width_ratios": [0.8, 1, 1]})
dom = ["outgoing", "warm", "organized", "anxious", "curious"]
a_p = [2.6, 4.1, 3.4, 2.2, 4.4]; b_p = [2.7, 4.0, 3.5, 2.3, 4.3]               # illustrative
items = ["bluegrass", "cult films", "musicals", "poetry", "heavy metal", "science fiction"]
model_a = [3.9, 4.6, 4.4, 5.3, 2.6, 4.9]; model_b = [3.8, 4.5, 4.5, 5.2, 2.7, 4.8]
real_a = [6.5, 6.2, 2.5, 3.0, 3.5, 4.0]; real_b = [2.5, 2.0, 6.5, 6.4, 1.5, 3.5]
y = np.arange(len(dom))[::-1]
ax[0].plot(a_p, y, "o", color=C["person"], ms=4, label="Ruth"); ax[0].plot(b_p, y, "s", color=C["person2"], ms=3.6, label="Glen")
ax[0].set_yticks(y); ax[0].set_yticklabels(dom); ax[0].set_xlim(1, 5); ax[0].set_xlabel("self-description (1-5)")
ax[0].set_title("Two near-identical personalities", fontsize=7.8, loc="left"); ax[0].legend(frameon=False, fontsize=6.6, loc="lower left")
yi = np.arange(len(items))[::-1]
for k, (va, vb, t) in enumerate([(model_a, model_b, "One character, predicted for both"), (real_a, real_b, "What they actually like")]):
    a = ax[k + 1]; a.plot(va, yi, "o", color=C["person"], ms=4); a.plot(vb, yi, "s", color=C["person2"], ms=3.6)
    for j in range(len(items)): a.plot([va[j], vb[j]], [yi[j]] * 2, color="#bbb", lw=0.6, zorder=0)
    a.set_yticks(yi); a.set_yticklabels(items if k == 0 else [""] * len(items)); a.set_xlim(1, 7); a.set_xlabel("liking (1-7)")
    a.set_title(t, fontsize=7.8, loc="left")
for a in ax: clean(a)
fig.savefig(os.path.join(FIG, "fig_teaser.pdf"), bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 2: amplitude
T = A("trait_amplitude.json"); Ac = A("amplitude_cohorts.json"); RA = A("ridge_amplitude.json")   # ridge reference (Oct 10)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.45), gridspec_kw={"wspace": 1.0, "width_ratios": [1, 1.1]})
models = sorted({k.split("|")[0] for k in T}, key=lambda m: -T[f"{m}|C-fac"]["amplitude_ratio_pooled"])
yy = np.arange(1, len(models) + 1)[::-1]   # row 0 = the ridge reference
for j, (cond, lab, mk, col) in enumerate([("C-fac", "trait scores", "o", C["model"]), ("C-cardBF", "Big Five portrait", "^", C["model_l"]), ("C-cardFull", "full portrait", "v", C["model_l"])]):
    off = (j - 1) * 0.22
    for i, m in enumerate(models):
        d = T[f"{m}|{cond}"]; e = d["amplitude_ratio_pooled"]; lo, hi = d["ci95"]
        a1.plot([lo, hi], [yy[i] + off] * 2, color=col, lw=0.8); a1.plot(e, yy[i] + off, mk, color=col, ms=3.4, mfc=col if j == 0 else "white", label=lab if i == 0 else None)
_r = RA["ESCS media (US) | 35 trait scores"]; a1.plot(_r["ci95"], [0, 0], color=C["ridge"], lw=0.8); a1.plot(_r["amplitude_ratio"], 0, "D", color=C["ridge"], ms=3.2, label="ridge (other people)")
a1.axvline(1, color="#555", lw=0.6, ls="--"); a1.text(1.03, -0.85, "equal spread", fontsize=6.3, color="#555"); a1.set_ylim(-1.1, len(models) + 0.6)
a1.set_yticks(list(yy) + [0]); a1.set_yticklabels([NAME[m] for m in models] + ["Ridge, other people"]); a1.set_xlim(0.6, 2.4); a1.set_xlabel("personality-associated spread, predicted / observed")
a1.set_title("(a) ESCS: amplitude relative to observed answers", fontsize=7.8, loc="left")
a1.legend(frameon=False, fontsize=6.4, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
coh = [("ESCS media (US)", "ESCS media (US)"), ("Twin policy (US)", "Twin policy (US)"), ("Twin risk (US)", "Twin risk/benefit (US)"),
       ("Harmonies music (DE)", "Harmonies music (German-sp.)"), ("Indonesia stories (ID)", "Indonesia stories (ID)"),
       ("BESSI bfi2_to_bessi", "College: traits to skills (US)"), ("BESSI bessi_to_bfi2", "College: skills to traits (US)")]
_B = A("bessi_results.json")["directions"]
for _d in ["bfi2_to_bessi", "bessi_to_bfi2"]:
    Ac["BESSI " + _d] = {k: {"amplitude_ratio": v["amplitude"]["amplitude_ratio"], "direction_r": v["amplitude"]["direction_r"]}
                         for k, v in _B[_d]["predictors"].items() if "amplitude" in v and not k.startswith("ridge")}
yc = np.arange(len(coh))[::-1]
for i, (k, lab) in enumerate(coh):
    vals = Ac[k]; est = [v["amplitude_ratio"] for v in vals.values()]; dirs = [v["direction_r"] for v in vals.values()]
    a2.plot([min(est), max(est)], [yc[i]] * 2, color=C["model"], lw=2.2, alpha=0.35, solid_capstyle="round")
    a2.plot(est, [yc[i]] * len(est), "o", color=C["model"], ms=2.6)
    a2.text(3.55, yc[i], f"r {min(dirs):.2f}-{max(dirs):.2f}", va="center", fontsize=6.4, color="#444")
    _rv = RA.get("ESCS media (US) | five domains" if k.startswith("ESCS") else k)
    if k.startswith("BESSI"): _rv = _B[k.split(" ", 1)[1]]["predictors"]["ridge|reference"]["amplitude"]
    if _rv: a2.plot(_rv["amplitude_ratio"], yc[i] + 0.22, "D", color=C["ridge"], ms=2.8, label="ridge (other people)" if i == 0 else None)
a2.axvline(1, color="#555", lw=0.6, ls="--"); a2.set_yticks(yc); a2.set_yticklabels([l for _, l in coh]); a2.set_xlim(0.6, 3.5)
a2.text(3.55, -0.85, "direction r", fontsize=6.4, color="#444"); a2.set_ylim(-1.1, len(coh) - 0.4)
a2.set_xlabel("five-domain personality-associated spread, predicted / observed")
a2.set_title("(b) Five domains as predictors: above 1 in every cohort (Twin: wide intervals)", fontsize=7.8, loc="left")
for a in (a1, a2): clean(a)
fig.savefig(os.path.join(FIG, "fig_amplitude.pdf"), bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 3: the ladder
L = A("ladder.json"); Cc = A("couples_conditions.json")
rows = [("ESCS media (US)", "ESCS media\n(US, 8 models)"), ("Twin policy (US)", "Twin policy\n(US)"), ("Twin risk/benefit (US)", "Twin risk/benefit\n(US)"),
        ("Harmonies music (DE)", "Harmonies music\n(German-sp., couples)"), ("Indonesia stories (ID)", "Indonesia stories\n(ID)"),
        ("BESSI bfi2_to_bessi", "College: traits\nto skills (US)"), ("BESSI bessi_to_bfi2", "College: skills\nto traits (US)")]
_B = A("bessi_results.json")["directions"]
for _d in ["bfi2_to_bessi", "bessi_to_bfi2"]:
    L["BESSI " + _d] = {"cells": {k: v["distinctive"] for k, v in _B[_d]["predictors"].items() if not k.startswith("ridge") and not k.endswith("no_person")},
                        "ridge": _B[_d]["predictors"]["ridge|reference"]["distinctive"]}
fig, ax = plt.subplots(figsize=(6.5, 2.9)); yr = np.arange(len(rows))[::-1]
seen = set()
def lab(x):
    if x in seen: return None
    seen.add(x); return x
for i, (k, name) in enumerate(rows):
    v = L[k]; cells = [d["mean"] for d in v["cells"].values()]
    ax.plot([min(cells), max(cells)], [yr[i]] * 2, color=C["model"], lw=3, alpha=0.3, solid_capstyle="round")
    ax.plot(cells, [yr[i]] * len(cells), "o", color=C["model"], ms=2.4, label=lab("model given the personality (each cell)"))
    if "ridge" in v:
        d = v["ridge"]; ax.plot([d["ci95"][0], d["ci95"][1]], [yr[i] + 0.18] * 2, color=C["ridge"], lw=0.8); ax.plot(d["mean"], yr[i] + 0.18, "D", color=C["ridge"], ms=3.2, label=lab("ridge trained on other people"))
    if "earlier_self" in v:
        d = v["earlier_self"]; ax.plot([d["ci95"][0], d["ci95"][1]], [yr[i] - 0.18] * 2, color=C["person"], lw=0.8); ax.plot(d["mean"], yr[i] - 0.18, "s", color=C["person"], ms=3.4, label=lab("the person's own earlier answers"))
    if "partner_real" in v:
        d = v["partner_real"]; ax.plot([d["ci95"][0], d["ci95"][1]], [yr[i] - 0.18] * 2, color=C["partner"], lw=0.8); ax.plot(d["mean"], yr[i] - 0.18, "o", color=C["partner"], ms=3.6, label=lab("the partner's real answers"))
        pt = [Cc["accuracy"][f"{m}|partner_t"]["mean"] for m in ["sol", "glm", "flash"]]
        ax.plot(pt, [yr[i] - 0.36] * 3, "o", mfc="white", mec=C["model"], ms=3.4, label=lab("model also shown the partner's answers"))
ax.axvline(0, color="#555", lw=0.6); ax.text(0.004, -0.75, "population average = 0", fontsize=6.3, color="#555"); ax.set_ylim(-0.95, len(rows) - 0.55)
ax.set_yticks(yr); ax.set_yticklabels([n for _, n in rows]); ax.set_xlabel("distinctive accuracy (how a prediction departs from the average, vs how the person does)")
ax.set_title("Models mostly trail a ridge trained on other people; a person's own answers reach furthest", fontsize=7.8, loc="left")
ax.legend(frameon=False, fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3); clean(ax)
fig.savefig(os.path.join(FIG, "fig_ladder.pdf"), bbox_inches="tight"); plt.close(fig)
print("figures ->", FIG)

# ---------------- Figure 4: quantity and arrangement (the original data and selection logic, plot_quantity.py; redrawn single-column, house style)
from matplotlib.lines import Line2D
QO = "<DATA_ROOT>/03_OCTOBER_EXPANSION/expansion_execution/quantity"
qa = json.load(open(f"{QO}/ANALYSIS.json")); qb = json.load(open(f"{QO}/human_comparator/RESULTS.json"))
fig, axes = plt.subplots(1, 2, figsize=(3.25, 1.75), sharey=True, gridspec_kw={"wspace": 0.12})
QC = {"flash": C["model"], "glm52": C["model_l"]}; levels = [0, 4, 8, 16, 32, "all"]; labels = ["0", "4", "8", "16", "32", "all"]
for ax, d, title in zip(axes, ["taste_to_values", "values_to_taste"], ["values from tastes", "tastes from values"]):
    for m in ["flash", "glm52"]:
        for scrambled in [False, True]:
            vals = []
            for k in levels:
                cond = "source_none_0" if k == 0 else ("source_all_all" if k == "all" else "source_random_" + str(k))
                if scrambled:
                    if k == 0 or (k == 4 and d == "taste_to_values"): vals.append(np.nan); continue
                    cond = "scrambled_" + cond
                row = next(x for x in qa["summary"] if (x["direction"], x["model"], x["condition"]) == (d, m, cond)); vals.append(row["metrics"]["distinctive"]["mean"])
            ax.plot(range(6), vals, color=QC[m], ls="--" if scrambled else "-", marker=None if scrambled else "o", ms=2.4, lw=1.0)
    vals = [next(x for x in qb["rows"] if x["direction"] == d and x["quantity"] == k)["metrics"]["distinctive"]["mean"] for k in levels[1:]]
    ax.plot(range(1, 6), vals, color=C["ridge"], marker="D", mfc="white", ms=2.4, lw=0.8)
    ax.axhline(0, color="#555", lw=0.5); ax.set_title(title, fontsize=7, loc="left"); ax.set_xticks(range(6)); ax.set_xticklabels(labels, fontsize=6.3)
    ax.set_xlabel("answers supplied", fontsize=6.6); ax.set_ylim(-0.09, 0.30); ax.tick_params(labelsize=6.3); clean(ax)
axes[0].set_ylabel("distinctive accuracy", fontsize=6.6)
h = [Line2D([], [], color=QC["flash"], label="Gemini"), Line2D([], [], color=QC["glm52"], label="GLM"), Line2D([], [], color="#777", ls="--", label="scrambled"),
     Line2D([], [], color=C["ridge"], marker="D", mfc="white", ms=2.4, label="human-trained")]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=6.0, handlelength=1.6, columnspacing=0.8)
fig.savefig(os.path.join(FIG, "fig_quantity.pdf"), bbox_inches="tight"); plt.close(fig)
print("fig_quantity done")

# ---------------- Figure 1, single-column version (page 1, top right): same composite values as fig_teaser
fig, (b0, b1) = plt.subplots(1, 2, figsize=(3.25, 2.05), gridspec_kw={"width_ratios": [0.62, 1], "wspace": 0.95})
y = np.arange(len(dom))[::-1]
b0.plot(a_p, y, "o", color=C["person"], ms=3.2); b0.plot(b_p, y, "s", color=C["person2"], ms=2.8)
b0.set_yticks(y); b0.set_yticklabels(dom, fontsize=6.4); b0.set_xlim(1, 5); b0.set_xticks([1, 3, 5]); b0.tick_params(labelsize=6)
b0.set_xlabel("self-description", fontsize=6.4); b0.set_title("personality", fontsize=7, loc="left")
yi = np.arange(len(items))[::-1]
for j in range(len(items)):
    b1.plot([real_a[j], real_b[j]], [yi[j]] * 2, color="#ccc", lw=0.6, zorder=0)
b1.plot(real_a, yi, "o", color=C["person"], ms=3.2, label="Ruth, real")
b1.plot(real_b, yi, "s", color=C["person2"], ms=2.8, label="Glen, real")
b1.plot([(x + z) / 2 for x, z in zip(model_a, model_b)], yi, "D", mfc="white", mec=C["model"], mew=1.0, ms=3.6, label="model, for both")
b1.set_yticks(yi); b1.set_yticklabels(items, fontsize=6.4); b1.set_xlim(1, 7); b1.set_xticks([1, 4, 7]); b1.tick_params(labelsize=6)
b1.set_xlabel("liking", fontsize=6.4); b1.set_title("tastes", fontsize=7, loc="left")
b1.legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.2, -0.24), ncol=3, handletextpad=0.2, columnspacing=0.7)
for a in (b0, b1): clean(a)
fig.savefig(os.path.join(FIG, "fig_teaser_col.pdf"), bbox_inches="tight"); plt.close(fig)
print("fig_teaser_col done")
