# The Personality Panorama

Code, prompts and supplementary results for the paper. The ARR attachments contain the same material in two archives: `software.zip` and `data.zip`.

## Contents

- `software/scripts/` contains the analysis and table-generation code. `software/reproduction_core/` contains the earlier scoring and diagnostic scripts.
- `software/prompts/` contains the prediction prompt, portrait-writing prompts and continuous personality reader. The college-cohort prompts are in `software/prompts/college/`. The questionnaire reader and some earlier prompts are described in the paper's appendix but are not included here.
- `data/analysis/`, `data/cohorts/` and `data/quantity/` contain aggregate results. `data/reproduction_core/expected/` contains the saved outputs of the earlier analyses.
- `data/paper_tables/` contains the appendix tables and LaTeX result macros. `data/FACTS_v1.md` lists numerical results with their source files.

## Reproducing the appendix tables

The three appendix tables can be generated directly from the included results using Python 3, without installing additional packages.

Start in the directory containing `software/` and `data/`. If using the ARR attachments, extract each archive into a folder with that name. Then run:

```sh
mkdir -p table-check/scripts table-check/paper
cp software/scripts/tables.py table-check/scripts/
cp -R data/analysis table-check/
python3 table-check/scripts/tables.py
```

The tables are written to `table-check/paper/` as `tab_amplitude.tex`, `tab_couples.tex` and `tab_bessi.tex`. They should be identical to the copies in `data/paper_tables/`. To compare them:

```sh
diff -u data/paper_tables/tab_amplitude.tex table-check/paper/tab_amplitude.tex
diff -u data/paper_tables/tab_couples.tex table-check/paper/tab_couples.tex
diff -u data/paper_tables/tab_bessi.tex table-check/paper/tab_bessi.tex
```

No output from these comparisons means the files match.

## Reproducing the other analyses

The analysis scripts require participant-level questionnaire data, saved model predictions and preparation files that are not distributed here. Some earlier preparation scripts, run manifests and assignment files are also absent. Results for agreement between models, shrinking toward the average, item order, similar-person swaps and repeatability are included, but the files do not provide every step needed to recompute them.

Paths to these inputs are marked with `<DATA_ROOT>`, `<DATA_ROOT_SEPT>`, `<PROJECT_ROOT>`, `<LOCAL>` or `<INSTRUMENT_CATALOGS>`. Replace those placeholders with the locations of the required inputs before running a script. `software/requirements.txt` lists the analysis dependencies; model settings and scoring details are in the paper's appendix.

`software/scripts/build_facts.py` also reads the previous manuscript's source and private reader-panel summaries. The resulting numerical index is included as `data/FACTS_v1.md`, but its builder cannot run from the public files alone.

## Analyses by paper section

Script paths below are relative to `software/`; result paths are relative to `data/`.

| Paper section | Code | Results |
|---|---|---|
| §2 distinctive accuracy, wrong-person reassignment, ridge | `reproduction_core/check_scoring.py`, `reproduction_core/reviewer_diagnostics.py`; `scripts/ladder.py` | `reproduction_core/expected/core_RESULTS.json`; `analysis/ladder.json` |
| §3 automated readback | `reproduction_core/reproduce_extensions.py`; `scripts/bessi_analysis.py` (college) | `cohorts/external_additions_RESULTS.json`; `analysis/bessi_results.json` |
| §3 human reader panels | `scripts/reader_panel_aggregates.py` (reads private summaries) | `analysis/reader_panels_aggregate.json` |
| §4 stability and repeatability | `reproduction_core/reproduce_extensions.py` | `reproduction_core/expected/repeat.json` |
| §4 agreement between models, raw fit, shrinking toward the average | Aggregate results only | `paper_tables/results_macros.tex`, `reproduction_core/expected/core_RESULTS.json` |
| §4 amplitude and direction (Fig. 2; Tables amp35, amp5) | `scripts/trait_amplitude.py`, `scripts/amplitude_cohorts.py`, `scripts/ridge_amplitude.py`, `scripts/figs.py`, `scripts/tables.py` | `analysis/trait_amplitude.json`, `analysis/amplitude_cohorts.json`, `analysis/ridge_amplitude.json` |
| §4 overall spread | `scripts/trait_coupling.py` | `analysis/trait_coupling.json` |
| §5 item-level accuracy | `scripts/where_right.py` | `analysis/where_right.json` |
| §5 ladder (Fig. 3) | `scripts/ladder.py` | `analysis/ladder.json` |
| §5 similar-person swaps; item order (appendix) | Aggregate results only | `FACTS_v1.md`, `reproduction_core/expected/expanded_controls.json` |
| §5 answers in another domain; quantity and arrangement (Fig. 4) | `reproduction_core/reproduce_extensions.py`; `scripts/figs.py` | `quantity/ANALYSIS.json`, `quantity/human_comparator_RESULTS.json`, `reproduction_core/expected/quantity*.json` |
| §5.1 college cohort (Table bessi) | `scripts/bessi_prepare.py`, `scripts/bessi_run.py`, `scripts/bessi_reparse.py`, `scripts/bessi_analysis.py`, `scripts/tables.py` | `analysis/bessi_results.json` |
| §5.2 couples and couple conditions (Table couples) | `scripts/couples.py`, `scripts/couples_run.py`, `scripts/couples_analysis.py`, `scripts/tables.py` | `analysis/couples.json`, `analysis/couples_conditions.json` |
| Appendix controlled panel | `reproduction_core/reproduce_extensions.py` | `reproduction_core/expected/controlled.json` |

## Extension experiments

`extensions/EXTENSIONS.md` describes three further experiments, with their scripts in `software/extensions/` and aggregate results in `data/extensions/EXTENSIONS_RESULTS.json`: (A) adding a few of a person's own countertypical tastes to their trait scores; (B) the same ESCS task across the post-training stages of one open model family (OLMo-3, 7B and 32B: base, SFT, DPO, Instruct); (C) two further automated readers (Claude Haiku 5.5, Mistral Large 4) replaying every readback request unchanged. As for the main study, no per-person prompts, predictions or readbacks are released.

The supplementary tables for these experiments (`paper_tables/tab_exceptions_ready.tex`, `tab_olmo_ready.tex`, `tab_readers_ready.tex`) are rendered by `software/extensions/manuscript_ready/render_tables.py` from four aggregate files in `data/extensions/manuscript_ready/`: `EXPERIMENT_A_READY.json`, `OLMO_READY.json`, `READERS_READY.json` and `RECORD_32B_READY.json`. The `prepare_*.py` scripts beside it produce those aggregates from the frozen experiment outputs without new model calls. For experiment A they map an item label returned with appended text (`item:text`) back to its item, and score all five arms on each person's common set of unshown items, so every arm keeps all 467 people. These values supersede the experiment-A entries in `EXTENSIONS_RESULTS.json`. `paired_holm.py` adds paired $t$-tests on the same per-person differences, Holm-corrected across all 21 contrasts (`EXPERIMENT_A_PAIRED.json`), and `render_paired.py` renders `tab_exceptions_paired.tex` from it.

## Data sources

In keeping with the paper's ethics statement, we do not release participant records, including questionnaire answers, portraits, personality readbacks, individual predictions or identifiers. The original datasets can be obtained from their authors:

- **ESCS (Eugene–Springfield Community Sample)** — Goldberg & Saucier (2016), Harvard Dataverse, CC0. IPIP: [10.7910/DVN/UF52WY](https://doi.org/10.7910/DVN/UF52WY); SDV: [10.7910/DVN/LHHONE](https://doi.org/10.7910/DVN/LHHONE); PRS: [10.7910/DVN/XJ6MXH](https://doi.org/10.7910/DVN/XJ6MXH); PPQ: [10.7910/DVN/BTNABX](https://doi.org/10.7910/DVN/BTNABX).
- **Twin-2K-500** — Toubia et al. (2025), [10.1287/mksc.2025.0262](https://doi.org/10.1287/mksc.2025.0262), CC BY 4.0.
- **Harmonies of Taste, Study 2** — Vigl et al. (2026), [10.1177/19485506261472146](https://doi.org/10.1177/19485506261472146); [data and study materials](https://osf.io/83ruh/).
- **Indonesian cohort** — Dubourg et al. (2026), [10.31234/osf.io/v3ecq_v1](https://doi.org/10.31234/osf.io/v3ecq_v1); [Study 3 data](https://osf.io/w493y/).
- **College cohort (BESSI and BFI-2)** — Soto et al. (2022), Sewell et al. (2022); [data and study materials](https://osf.io/4zgyr/).

Instrument item catalogs are not included. Scripts that use item wording expect the appropriate catalog at `<INSTRUMENT_CATALOGS>`.
