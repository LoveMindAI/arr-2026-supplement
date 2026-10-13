#!/usr/bin/env python3
"""Every number in the main text must come from the facts file (house rule). This lists each number in the main-text sections
(00-08, LaTeX comments stripped) that does not appear in paper_notes/FACTS_v1.md, with its sentence, for a human to trace.
Matching is on the printed string with and without a leading zero (".957" ~ "0.957"), and on percentages, so a flag means
"not found as printed", not necessarily wrong (e.g. a rounded figure, a count, or a section number). Exit 0 always.
Usage: python3 scripts/check_numbers.py"""
import glob, os, re
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
facts = open(os.path.join(ROOT, "paper_notes", "FACTS_v1.md")).read()
facts_nums = set(re.findall(r"-?\d[\d,]*\.?\d*|-?\.\d+", facts))
norm = lambda x: x.replace(",", "").lstrip("+")
fset = {norm(x) for x in facts_nums} | {norm("0" + x) if x.startswith(".") else norm(x) for x in facts_nums} | {norm(x.replace("0.", ".", 1)) for x in facts_nums if x.startswith("0.")}
# also rounded forms of every fact value to 2 and 3 decimals
for x in list(facts_nums):
    try:
        v = float(norm(x))
        for d in (1, 2, 3):
            s = f"{v:.{d}f}"; fset |= {s, s.replace("0.", ".", 1) if s.startswith("0.") else s, s.replace("-0.", "-.", 1)}
    except ValueError: pass
MACROS = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(os.path.join(ROOT, "paper", "results_macros.tex")).read()))
SKIP = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"}   # list markers, scale points, RQ numbers: read by eye
flags = 0
for f in sorted(glob.glob(os.path.join(ROOT, "paper", "sections", "0[0-8]*.tex"))):
    txt = "\n".join(re.sub(r"(?<!\\)%.*", "", l) for l in open(f).read().splitlines())   # only unescaped % starts a comment (the original analysis/a reviewer, Oct 11)
    for name, val in MACROS.items(): txt = txt.replace("\\" + name + "{}", val).replace("\\" + name + " ", val + " ")
    txt = re.sub(r"\\(ref|label|cite[pt]?|includegraphics|input|section|paragraph)\*?(\[[^\]]*\])?\{[^}]*\}", "", txt)
    for sent in re.split(r"(?<=[.;])\s+(?=[A-Z\\])", txt):
        for m in re.finditer(r"(?<![\w.])(-?\d[\d,]*(?:\.\d+)?|-?\.\d+)(?![\w])", sent):
            n = norm(m.group(1)).rstrip(".")
            if n in SKIP or n in fset: continue
            flags += 1; print(f"{os.path.basename(f)}: {m.group(1)!r}  in: {sent.strip()[:160]}")
print(f"{flags} numbers not found in FACTS_v1.md as printed")
