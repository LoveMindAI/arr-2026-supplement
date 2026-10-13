"""Render the paired-contrast table for the personal-evidence follow-up from EXPERIMENT_A_PAIRED.json."""
from pathlib import Path
import json
HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'EXPERIMENT_A_PAIRED.json').read_text())
C = P['contrasts']
def f(x):
    s = f'{abs(x):.3f}'.lstrip('0')
    return ('$-$' if x < 0 else '+') + s if round(x, 3) != 0 else s
def cell(v):
    lo, hi = v['ci95']
    star = '' if v['p_holm'] < .05 else '$^{\\dagger}$'
    return f"{f(v['mean'])} [{f(lo)}, {f(hi)}]{star}"
rows = [('A1-A0', '4 own countertypical $-$ none'), ('A4-A0', '12 own countertypical $-$ none'),
        ('A4-A1', '12 own $-$ 4 own countertypical'), ('A1-A2', '4 own countertypical $-$ 4 own typical'),
        ('A1-A3', '4 own countertypical $-$ 4 stranger'), ('A2-A0', '4 own typical $-$ none'),
        ('A3-A0', '4 stranger countertypical $-$ none')]
L = [r'\begin{table*}[!t]', r'\centering\small', r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
     r'Paired contrast & Gemini-3-Flash & Qwen3.7-plus & MiniMax-M3\\', r'\midrule']
for k, lab in rows:
    L.append(lab + ' & ' + ' & '.join(cell(C[m][k]) for m in ('flash', 'qwen37', 'minimax')) + r'\\')
worst = max(v['p_holm'] for m in C for v in C[m].values() if v['p_holm'] < .05)
L += [r'\bottomrule', r'\end{tabular}',
      r'\caption{Paired differences in distinctive accuracy for the personal-evidence follow-up, computed within person on the same unshown-item mask (467 people per model). Brackets are paired 95\%% bootstrap intervals (2,000 resamples). The %d contrasts (seven per model) were tested with paired $t$-tests and Holm-corrected together; all survive (adjusted $p \le %s$) except the one marked $^{\dagger}$.}' % (P['family_size'], f'{worst:.3f}'.lstrip('0')),
      r'\label{tab:exceptions_paired}', r'\end{table*}']
(HERE / 'tab_exceptions_paired.tex').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
