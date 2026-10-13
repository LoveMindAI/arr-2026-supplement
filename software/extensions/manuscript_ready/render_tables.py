"""Render aggregate-only scientific tables for the independent manuscript."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
A = json.loads((ROOT/'analysis/EXPERIMENT_A_READY.json').read_text())
B = json.loads((ROOT/'analysis/OLMO_READY.json').read_text())['ladder']
def f(x, n=3):
    return f'{x:.{n}f}'.replace('0.', '.', 1) if abs(x) < 1 else f'{x:.{n}f}'
def interval(xs):
    return f'{f(xs[0])} [{f(xs[1])}, {f(xs[2])}]'
names = {'flash':'Gemini-3-Flash','qwen37':'Qwen3.7-plus','minimax':'MiniMax-M3'}
arms = {'A0':'None','A1':'4 own countertypical','A2':'4 own typical',
        'A3':'4 stranger countertypical','A4':'12 own countertypical'}
lines = [r'\begin{table*}[!t]', r'\centering\small',
    r'\begin{tabular}{@{}llrr@{}}', r'\toprule',
    r'Model & Added ratings & Distinctive accuracy [95\% CI] & Amplitude\\', r'\midrule']
for model in names:
    for arm,label in arms.items():
        rec=A[model][arm]
        lines.append(f'{names[model]} & {label} & {interval(rec["distinctive"])} & {rec["amplitude"]:.2f}'+r'\\')
    if model != 'minimax': lines.append(r'\midrule')
lines += [r'\bottomrule',r'\end{tabular}',
    r'\caption{Personal-evidence follow-up. All entries use 467 people and the same unshown-item mask across arms within each model. Accuracy intervals are participant bootstraps; amplitude entries are point estimates relative to observed human ratings.}',
    r'\label{tab:exceptions_full}',r'\end{table*}']
(ROOT/'paper/tab_exceptions_ready.tex').write_text('\n'.join(lines)+'\n')
stages = [('7B','Base','allenai_Olmo-3-1025-7B'),('7B','SFT','allenai_Olmo-3-7B-Instruct-SFT'),
    ('7B','DPO','allenai_Olmo-3-7B-Instruct-DPO'),('7B','Final','allenai_Olmo-3-7B-Instruct'),
    ('32B','Base','allenai_Olmo-3-1125-32B'),('32B','SFT','allenai_Olmo-3.1-32B-Instruct-SFT'),
    ('32B','DPO','allenai_Olmo-3.1-32B-Instruct-DPO'),('32B','Final','allenai_Olmo-3.1-32B-Instruct')]
lines = [r'\begin{table*}[!t]',r'\centering\small',
    r'\begin{tabular}{@{}lllrrr@{}}',r'\toprule',
    r'Size & Stage & Wrapper & Amplitude [95\% CI] & Direction & Distinctive [95\% CI]\\',r'\midrule']
for size,stage,tag in stages:
    for fmt in ['raw','chat']:
        r=B[f'{tag}|{fmt}']
        assert r['n_people']==467
        amp=f'{r["amp"]:.2f} [{r["amp_ci"][0]:.2f}, {r["amp_ci"][1]:.2f}]'
        lines.append(f'{size} & {stage} & {fmt} & {amp} & {r["dir"]:.2f} & {interval(r["distinctive"])}'+r'\\')
    if stage == 'Final' and size == '7B': lines.append(r'\midrule')
lines += [r'\bottomrule',r'\end{tabular}',
    r'\caption{OLMo released-checkpoint comparison. Each cell evaluates 467 profiles using expected ratings from next-token digit probabilities. Amplitude intervals use 100 random half-samples; distinctive-accuracy intervals use 2,000 participant bootstrap samples. Wrapper and estimator are fixed across stages within each size.}',
    r'\label{tab:olmo_ready}',r'\end{table*}']
(ROOT/'paper/tab_olmo_ready.tex').write_text('\n'.join(lines)+'\n')
readers=json.loads((ROOT/'analysis/READERS_READY.json').read_text())
labels={'Twin (150) continuous':'Twin, direct', 'Twin (150) questionnaire':'Twin, questionnaire',
        'Harmonies (148) continuous':'Harmonies, direct', 'Harmonies (148) questionnaire':'Harmonies, questionnaire',
        'Indonesia (100) readback':'Indonesia, direct', 'College BFI-2 (200) readback':'College personality',
        'College skills (200) readback':'College skills'}
lines=[r'\begin{table*}[!t]',r'\centering\small',r'\begin{tabular}{@{}lrrrrrrr@{}}',r'\toprule',
       r' & & \multicolumn{3}{c}{Mean rank correlation} & \multicolumn{3}{c}{Mean absolute error}\\',
       r'\cmidrule(lr){3-5}\cmidrule(l){6-8}',
       r'Panel & People & Luna & Haiku & Mistral & Luna & Haiku & Mistral\\',r'\midrule']
for name,r in readers.items():
    values=[f(r[reader][metric]) for metric in ['mean_rank_r','mae'] for reader in ['luna','haiku','mistral']]
    lines.append(f'{labels[name]} & {r["n"]} & '+' & '.join(values)+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',
    r'\caption{Matched automated readback against measured source scores for GPT-5.6-Luna, Claude Haiku 5.5 and Mistral Large 4. Rank correlations are averaged across five domains; mean absolute error is on the 1--5 scale. Each row uses the same portraits and task for all three readers. Indonesia uses the 100-person October subset of the original 150-person cohort.}',
    r'\label{tab:readers_ready}',r'\end{table*}']
(ROOT/'paper/tab_readers_ready.tex').write_text('\n'.join(lines)+'\n')
print('Rendered three aggregate tables.')
