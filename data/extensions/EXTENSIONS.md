# Extension experiments

Three further experiments on the ESCS and five-domain cohorts. Code is in `software/extensions/`; aggregate results are in `data/extensions/EXTENSIONS_RESULTS.json`. No per-person prompts, predictions, readbacks or selections are released (as for the main study). Scoring uses the paper's definitions (`extensions/common.py`), which reproduce the paper's saved ESCS values (ridge amplitude 0.777; Gemini-3-Flash distinctive accuracy .193; Qwen3.7-plus amplitude 1.92).

## A. A person's own countertypical tastes
**Question.** Does giving a model a few of the person's own tastes, alongside their 35 trait scores, move its predictions toward that person and off the personality stereotype?

**Design.** 467 ESCS participants. Each request is the person's original 35-trait-score prompt, unchanged, with one added block of known ratings inserted after the trait lines. Arms: A0 scores only; A1 four of the person's own countertypical tastes (largest |rating minus out-of-sample ridge prediction|); A2 four of their own typical tastes (smallest residual); A3 four countertypical tastes of another person (fixed circular map); A4 twelve own countertypical tastes. Models: Gemini-3-Flash, Qwen3.7-plus, MiniMax-M3 via OpenRouter. Each person is scored only on items shown in no arm (median 89 items). Reading rule, fixed before results: a model routes toward the person if A1 beats A0, A2 and A3 on distinctive accuracy (paired 95% intervals above zero) and lowers amplitude. Scripts: `build_expA.py`, `run_expA.py`, `analyze_A.py`.

| arm | distinctive accuracy (Flash / Qwen / MiniMax) | amplitude (people = 1) |
|---|---|---|
| A0 scores only | .206 / .167 / .138 | 1.88 / 1.92 / 1.64 |
| A1 + 4 own countertypical | .244 / .172 / .172 | 1.70 / 1.64 / 1.48 |
| A4 + 12 own countertypical | .288 / .201 / .205 | 1.57 / 1.46 / 1.40 |
| A2 + 4 own typical | .191 / .143 / .126 | 1.51 / 1.49 / 1.36 |
| A3 + 4 another person's countertypical | .153 / .127 / .113 | 1.64 / 1.64 / 1.50 |

A4 minus A0: +.082, +.035, +.066 (all intervals above zero). A3 minus A0: -.053, -.040, -.025 (all below zero). A1 meets the reading rule for Flash and MiniMax; for Qwen, A1 minus A0 is +.006 [-.003, .014].

## B. Where the exaggeration appears: OLMo-3 post-training stages
**Question.** At which stage of training does a model start tying taste to personality more strongly than people do?

**Design.** Open checkpoints from one family (AllenAI OLMo-3), each the parent of the next: 7B base, SFT, DPO, final Instruct; 32B base, 3.1 SFT, DPO, Instruct. Every ESCS person's original 35-trait-score prompt, unchanged. Each of the 109 ratings is read as the expected value of the next-token distribution over the digits 1-7 after `{"<item>": ` (OLMo-3 tokenizes the space before a digit as its own token), in two wrappers: the prompt as a plain document, and the prompt as a user turn in the instruct chat template. vLLM, temperature 0, top-20 log-probabilities; at least 95% of probability mass falls on the digits in every condition. A second framing presents the same 35 trait lines as a survey record, with no instructions. "Personality share" is the mean over items of the adjusted R^2 of a model's predictions on the 35 trait scores: the share of its own variation across people that personality explains (people: .135). Scripts: `score_ladder.py`, `run_ladder.sh`, `analyze_B.py`, `score_record.py`, `analyze_B2.py`, `export_aggregates.py`.

| checkpoint | amplitude, chat wrapper [95%] | amplitude, plain document | personality share (chat) | distinctive (chat) |
|---|---|---|---|---|
| 7B base | 0.15 [0.14, 0.17] | 0.13 | .319 | -.016 |
| 7B SFT | 0.49 [0.46, 0.53] | 0.25 | .706 | -.001 |
| 7B DPO | 0.63 [0.59, 0.67] | 0.37 | .751 | +.003 |
| 7B Instruct | 0.62 [0.58, 0.66] | 0.36 | .735 | +.003 |
| 32B base | 0.53 [0.49, 0.58] | 0.70 | .569 | +.016 |
| 32B SFT | 1.36 [1.27, 1.45] | 1.11 | .794 | +.050 |
| 32B DPO | 1.56 [1.46, 1.66] | 1.38 | .801 | +.052 |
| 32B Instruct | 1.62 [1.52, 1.73] | 1.43 | .802 | +.051 |

Survey-record framing (amplitude / personality share): 7B base 0.18 / .694; 7B SFT 0.18 / .493; 7B Instruct 0.27 / .608; 32B base 0.25 / .704; 32B Instruct 0.70 / .795. Direction stays between .25 and .41 in every condition. Reading: in every base model, personality already explains more of the (small) variation across people than it does in people's own answers (.17 to .81, against .135); each post-training stage raises how far personality spreads the predictions; the 32B models exceed people only when the prompt asks them to answer as the person.

## C. Additional automated readers
**Question.** Does the readback result depend on the paper's single automated reader (GPT-5.6-Luna)?

**Design.** Every readback request behind the paper's readback numbers (1,096 requests: Twin 150 and Harmonies 148, direct and questionnaire; Indonesia 100; college 200 personality and 200 skill portraits) was sent unchanged to Claude Haiku 5.5 and Mistral Large 4, reasoning off. Questionnaire answers are scored with the BFI-44 key. Scripts: `collect.py`, `run_reader.py`, `metrics.py`.

| set | Luna | Haiku | Mistral | mean absolute error (L / H / M) |
|---|---|---|---|---|
| Twin, direct | .957 | .939 | .959 | .368 / .274 / .290 |
| Twin, questionnaire | .925 | .925 | .902 | .530 / .296 / .568 |
| Harmonies, direct | .957 | .921 | .949 | .390 / .209 / .314 |
| Harmonies, questionnaire | .940 | .903 | .929 | .554 / .300 / .608 |
| Indonesia | .924 | .898 | .928 | .393 / .293 / .301 |
| College, personality | .966 | .941 | .963 | .382 / .234 / .309 |
| College, skills | .943 | .911 | .940 | .320 / .197 / .302 |

Values are mean rank correlations across people within each domain. Mean signed error (estimate minus score) for the college personality portraits: agreeableness +.28 (Luna), +.21 (Mistral), -.05 (Haiku); conscientiousness +.23, +.11, -.05; neuroticism -.29, -.26, -.08. Full bias tables are in the JSON.

## Rerunning
All three extensions read data that are not redistributed (see Data sources in the main README); paths are placeholders (`<DATA_ROOT>`, `<DATA_ROOT_SEPT>`, `<PROJECT_ROOT>`). Scoring for all three is `common.py`.
- **A.** `build_expA.py` (needs each person's original ESCS prompt and the out-of-sample ridge predictions) -> `run_expA.py MODEL` for `flash`, `qwen37`, `minimax` (OpenRouter; temperature 0; output limits 3,000 / 6,000 / 8,000 tokens) -> `analyze_A.py`.
- **B.** `build_ladder_inputs.py` writes `prompts.jsonl`, `items.json`, `records.jsonl`, `labels.json` and the checkpoint lists `LADDER.txt` (7B) and `LADDER32.txt` (32B). On one 80 GB GPU with vLLM 0.31.0: `run_ladder.sh` scores each listed checkpoint with `score_ladder.py` (7B defaults; for 32B set `TMPL=allenai/Olmo-3.1-32B-Instruct MAXLEN=4608 GPU_UTIL=0.92`), and `score_record.py <checkpoint> <out>` runs the survey-record framing. Then `analyze_B.py`, `analyze_B2.py` and `export_aggregates.py` locally.
- **C.** `collect.py` gathers each saved readback request with its scores; `run_reader.py --model anthropic/claude-haiku-5.5 --out haiku.jsonl` and `--model mistralai/mistral-large-4-0 --out mistral.jsonl` replay them unchanged; `metrics.py` computes the tables.
