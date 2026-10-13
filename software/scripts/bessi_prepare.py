#!/usr/bin/env python3
"""Prepare the BESSI college cohort for Panorama, both directions.

Source: OSF 4zgyr, College_student_sample.sav (final BESSI v1.0, 192 how-well items, plus the 60-item BFI-2), SHA-checked on download.
Directions:
  bfi2_to_bessi  the person's five BFI-2 domain means  -> answer all 192 BESSI items
  bessi_to_bfi2  the person's five BESSI domain scores -> answer all 60 BFI-2 items
Same procedure as the earlier Panorama cohorts (original, Sep 16 and Oct): Gemma draft with the frozen revision-1 writer, Gemma edit with
the frozen revision-3 editor and the source scores, terminology repair, Luna continuous readback, predictions by Sol, GLM and Flash in
numeric / structured / portrait formats with the same wrapper, plus a no-person baseline.
The writer, editor and reader are adapted from the frozen text by checked one-for-one replacements (each old string must occur exactly
once); the adapted templates are saved under bessi_run/templates/ so the delivered wording can be read.
Selection: complete cases on all 252 items; 200 evaluation people = first 200 SHA-256 ranks of the row position in the released file
(fixed salt; the file's Case number restarts in each of three batches);
the rest are the disjoint reference for norms and the ridge. Outcomes are not inspected.
Usage: PYTHONPATH=<scratch>/pyrs python3 scripts/bessi_prepare.py"""
import hashlib, json, re
from pathlib import Path
import numpy as np
import pyreadstat

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "bessi_osf" / "raw"
OUT = HERE / "bessi_run"
CATS = Path("<INSTRUMENT_CATALOGS>")
SEPT = Path("<DATA_ROOT>/04_SEPTEMBER_ARCHIVE")
T = SEPT / "bfi44_calibration_rounds_20260916"; AD = SEPT / "external_launch_20260916" / "bfi44_ocean_adapter"
SALT = "panorama-bessi-college-20261009"; N_EVAL = 200

def sha(x): return hashlib.sha256(x if isinstance(x, bytes) else x.encode()).hexdigest()
def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + "\n"); tmp.replace(p)
def adapt(text, pairs, label):
    for old, new in pairs:
        n = text.count(old)
        if n != 1: raise ValueError(f"{label}: expected exactly one occurrence of {old[:70]!r}, found {n}")
        text = text.replace(old, new)
    return text

# ---------------- instruments ----------------
BFI2 = json.loads((CATS / "BFI2_CATALOG.json").read_text()); BESSI = json.loads((CATS / "BESSI_CATALOG.json").read_text())
BFI2_ANCHORS = "1 = disagree strongly; 2 = disagree a little; 3 = neutral, no opinion; 4 = agree a little; 5 = agree strongly."
BESSI_ANCHORS = "1 = not at all well; 2 = not very well; 3 = pretty well; 4 = very well; 5 = extremely well."
# Official instructions, verbatim: BESSI v1.0 self-report form (OSF docx); BFI-2 block of the OSF survey PDF.
BESSI_INSTRUCTIONS = ("Here is a list of activities or things you could do. For each one, please select a response to indicate how well you can do "
    "that thing. For example, how well can you follow the instructions for an assignment? Note that how well you can do something may be "
    "different from how often you do it, or how much you like to do it. For each activity, you should rate how well you can do that thing.")
BFI2_INSTRUCTIONS = ("Here are a number of characteristics that may or may not apply to you. For example, do you agree that you are someone who likes "
    "to spend time with others? Please select a response next to each statement to indicate the extent to which you agree or disagree with "
    "that statement.\nI am someone who . . .")

B2_KEYS = [("E", "extraversion"), ("A", "agreeableness"), ("C", "conscientiousness"), ("N", "negative emotionality"), ("O", "open-mindedness")]
BS_COL = {"SM": "BESSI_SelfManagementSkills", "SE": "BESSI_SocialEngagementSkills", "CO": "BESSI_CooperationSkills",
          "ER": "BESSI_EmotionalResilienceSkills", "IN": "BESSI_InnovationSkills"}
BS_KEYS = [("SM", "self-management skills"), ("SE", "social engagement skills"), ("CO", "cooperation skills"),
           ("ER", "emotional resilience skills"), ("IN", "innovation skills")]

# Frozen BFI-44 meanings (bfi44_catalog.json) renamed to the official BFI-2 domain labels; content unchanged.
D44 = json.loads((AD / "bfi44_catalog.json").read_text())["definitions"]
BFI2_DEF = {"E": D44["E"], "A": D44["A"], "C": D44["C"],
            "N": D44["N"].replace("Neuroticism concerns", "Negative Emotionality concerns", 1),
            "O": D44["O"].replace("Openness concerns", "Open-Mindedness concerns", 1)}
assert BFI2_DEF["N"] != D44["N"] and BFI2_DEF["O"] != D44["O"]
# BESSI meanings written from the instrument's own facet names (core facets, then the cross-facets that count once).
BESSI_DEF = {
 "SM": "Self-management skills concern how well a person can manage their time, keep things organized, stay consistent, manage tasks, attend to details, follow rules, handle responsibilities, pursue goals and make decisions, and to a lesser degree regulate their energy, act ethically, control impulses and process information. Lower scores mean the person reports doing these things less well when they are needed.",
 "SE": "Social engagement skills concern how well a person can lead others, persuade, express their thoughts and feelings, and carry a conversation, and to a lesser degree regulate their energy. Lower scores mean the person reports doing these things less well when they are needed.",
 "CO": "Cooperation skills concern how well a person can take others' perspectives, extend trust, show warmth and work as part of a team, and to a lesser degree act ethically. Lower scores mean the person reports doing these things less well when they are needed.",
 "ER": "Emotional resilience skills concern how well a person can regulate stress, stay optimistic, manage anger and keep their confidence steady, and to a lesser degree control impulses. Higher scores indicate greater reported capacity to steady their own emotions when needed. Describe everyday emotional capacities without diagnosing the person.",
 "IN": "Innovation skills concern how well a person can think abstractly, come up with creative ideas, engage with art, and understand and get along with people from other cultures, and to a lesser degree process information. Lower scores mean the person reports doing these things less well when they are needed. Convey the degree of these capacities without assigning particular cultural tastes or accomplishments.",
}
LABELS_WRITER = ("openness, open-mindedness, conscientiousness, extraversion, introversion, agreeableness, neuroticism, negative emotionality, "
                 "self-management, social engagement, cooperation skills, emotional resilience, innovation skills, Big Five, BFI-2, or BESSI")
LABELS_REPAIR = ("openness, open-mindedness, conscientiousness, extraversion, extroversion, introversion, agreeableness, neuroticism, negative "
                 "emotionality, self-management, social engagement, cooperation skills, emotional resilience, innovation skills, Big Five, BFI, "
                 "BFI-2, BESSI, percentile, psychometric, questionnaire")

# ---------------- writer (revision 1), editor (revision 3), repair, reader (C3) adaptations ----------------
R1 = (T / "revision1_WRITER.txt").read_text(); R3 = (T / "revision3_WRITER.txt").read_text()
REP = (T / "terminology_repair" / "REPAIR_PROMPT.txt").read_text(); C3 = (AD / "C3_BFI44_CONTINUOUS_SYSTEM.txt").read_text()
old_meanings_r1 = "\n".join(l for l in R1.splitlines() if l.startswith("- ") and " concerns " in l)
old_meanings_r3 = next(l for l in R3.splitlines() if l.startswith("Extraversion concerns"))
old_meanings_c3 = "\n".join(l for l in C3.splitlines() if l.startswith("- ") and " concerns " in l)
OLD_LABELS_R1 = "Do not use the labels openness, conscientiousness, extraversion, introversion, agreeableness, neuroticism, or Big Five."
OLD_LABELS_R3 = "The final portrait must not contain the words openness, conscientiousness, extraversion, introversion, agreeableness, neuroticism, Big Five, BFI, percentile, psychometric, or questionnaire."
OLD_LABELS_REP = "openness, conscientiousness, extraversion, extroversion, introversion, agreeableness, neuroticism, Big Five, BFI, BFI-44, percentile, psychometric, questionnaire"
REPAIR = adapt(REP, [(OLD_LABELS_REP, LABELS_REPAIR)], "repair")

def bfi2_templates():
    order = ["E", "A", "C", "N", "O"]
    w = adapt(R1, [
        ("translating private BFI-44 personality measurements", "translating private BFI-2 personality measurements"),
        ("BFI-44 DOMAIN MEANINGS", "BFI-2 DOMAIN MEANINGS"),
        (old_meanings_r1, "\n".join("- " + BFI2_DEF[d] for d in order)),
        (OLD_LABELS_R1, f"Do not use the labels {LABELS_WRITER}."),
    ], "bfi2 writer")
    e = adapt(R3, [
        ("supplied BFI-44 measurements", "supplied BFI-2 measurements"),
        ("Neuroticism concerns susceptibility to worry, tension, low mood and emotional upset;", "Negative emotionality concerns susceptibility to worry, tension, low mood and emotional upset;"),
        ("Openness concerns imagination, curiosity,", "Open-mindedness concerns imagination, curiosity,"),
        (OLD_LABELS_R3, f"The final portrait must not contain the words {LABELS_WRITER.replace(', or BESSI', ', BESSI, BFI')}, percentile, psychometric, or questionnaire."),
    ], "bfi2 editor")
    r = adapt(C3, [
        ('Response scale: {"1": "Disagree strongly", "2": "Disagree a little", "3": "Neither agree nor disagree", "4": "Agree a little", "5": "Agree strongly"}',
         'Response scale: ' + json.dumps({str(i + 1): a for i, a in enumerate(BFI2["anchors"])}, ensure_ascii=False)),
        ("BFI-44 domain meanings", "BFI-2 domain meanings"),
        (old_meanings_c3, "\n".join("- " + BFI2_DEF[d] for d in ["A", "C", "E", "N", "O"])),
        ("mean keyed agreement on the BFI-44", "mean keyed agreement on the BFI-2"),
    ], "bfi2 reader")
    return w, e, r

def bessi_templates():
    order = ["SM", "SE", "CO", "ER", "IN"]
    w = adapt(R1, [
        ("translating private BFI-44 personality measurements", "translating private BESSI skill measurements"),
        ("The five measurements are means of questionnaire responses on a 1–5 agreement scale.",
         "The five measurements are weighted means of the person's own ratings, on a 1–5 scale, of how well they can do particular things when needed."),
        ("Responses have already been keyed so larger values indicate more of the named tendency; do not reverse them again. Read the degree literally. Values near 2 and 4 express a modest lean, not the strength of the endpoints. Values close to 3 express a slight lean or an intermediate tendency.",
         "Larger values indicate greater reported capacity; no response is reversed. Read the degree literally against these anchors. A value near 3 means the person reports doing these things pretty well; values near 4 mean very well and values near 2 not very well; only values near 5 or 1 reach the strength of the endpoints."),
        ("A mean is not a percentile, a percentage of occasions, or an objective measure of ability or worth.",
         "These are reported capacities when needed, not how often the person does these things or how much they like doing them. A mean is not a percentile, a percentage of occasions, or an objective measure of ability or worth."),
        ("BFI-44 DOMAIN MEANINGS", "BESSI DOMAIN MEANINGS"),
        (old_meanings_r1, "\n".join("- " + BESSI_DEF[d] for d in order)),
        ("- Give each tendency a behavioral expression grounded in its measurement.",
         "- Give each capacity a behavioral expression grounded in its measurement: what the person can manage, and how well, when a situation calls for it. A capacity is available when needed, not something the person does constantly."),
        (OLD_LABELS_R1, f"Do not use the labels {LABELS_WRITER}."),
    ], "bessi writer")
    e = adapt(R3, [
        ("supplied BFI-44 measurements", "supplied BESSI skill measurements"),
        ("These are keyed means on a 1–5 agreement scale:", "These are weighted means of the person's own how-well ratings on a 1–5 scale:"),
        ("Larger means indicate more of the named tendency. A value near 3 indicates a slight lean or an intermediate tendency; values near 2 and 4 are moderate rather than extreme.",
         "Larger means indicate greater reported capacity when needed, not frequency or liking. A value near 3 means pretty well; values near 4 mean very well and near 2 not very well, rather than the extremes."),
        (old_meanings_r3, " ".join(BESSI_DEF[d] for d in order)),
        (OLD_LABELS_R3, f"The final portrait must not contain the words {LABELS_WRITER.replace(', or BESSI', ', BESSI, BFI')}, percentile, psychometric, or questionnaire."),
    ], "bessi editor")
    r = adapt(C3, [
        ('Response scale: {"1": "Disagree strongly", "2": "Disagree a little", "3": "Neither agree nor disagree", "4": "Agree a little", "5": "Agree strongly"}',
         'Response scale: ' + json.dumps({str(i + 1): a for i, a in enumerate(BESSI["anchors"])}, ensure_ascii=False)),
        ("BFI-44 domain meanings", "BESSI domain meanings"),
        (old_meanings_c3, "\n".join("- " + BESSI_DEF[d] for d in order)),
        ("mean keyed agreement on the BFI-44, conditional on the portrait. Each domain mean averages its listed items, with reverse-keyed responses scored as 6 minus the original response.",
         "weighted domain score on the BESSI how-well form, conditional on the portrait. Each domain score averages its listed facets, each facet being the mean of its items; a facet with weight 2 counts twice. No BESSI item is reverse-keyed."),
        ("These forecasts are already keyed; do not reverse them afterward. ", ""),
    ], "bessi reader")
    return w, e, r

def reader_target(inst):
    if inst == "bfi2":
        return {d: {"name": BFI2["domains"][d]["name"], "items": [
            {"id": i, "text": it["text"], "orientation": "positive" if it["key"] == 1 else "reverse"}
            for i, it in BFI2["items"].items() if BFI2["facets"][it["facet"]]["domain"] == d]} for d, _ in B2_KEYS}
    out = {}
    for d, _ in BS_KEYS:
        dom = BESSI["domains"][BS_COL[d]]; seen = []
        for f in dom["facets"]:
            if f not in seen: seen.append(f)
        out[d] = {"name": dom["name"], "facets": [{"name": BESSI["facets"][f]["name"], "weight": dom["facets"].count(f),
                  "items": [BESSI["items"][i]["text"] for i in BESSI["facets"][f]["item_ids"]]} for f in seen]}
    return out

def main():
    df, _ = pyreadstat.read_sav(str(RAW / "College_student_sample.sav"))
    bes_items = [f"bessi_{i}" for i in range(1, 193)]; bfi_items = [f"bfi2_{i}" for i in range(1, 61)]
    assert set(bes_items) == set(BESSI["items"]) and set(bfi_items) == set(BFI2["items"])
    comp = df[df[bes_items + bfi_items].notna().all(axis=1)].copy()
    assert len(comp) == 313, len(comp)
    assert comp[bes_items + bfi_items].isin([1, 2, 3, 4, 5]).all().all()
    # scoring (verified earlier to reproduce the authors' columns exactly); re-assert here
    fac = {f: comp[s["item_ids"]].mean(axis=1) for f, s in BESSI["facets"].items()}
    bs = {d: np.mean([fac[f] for f in BESSI["domains"][BS_COL[d]]["facets"]], axis=0) for d, _ in BS_KEYS}
    for d, _ in BS_KEYS: assert np.nanmax(np.abs(bs[d] - comp[BS_COL[d]].values)) < 1e-9
    keyed = {i: (comp[i] if it["key"] == 1 else 6 - comp[i]) for i, it in BFI2["items"].items()}
    b2 = {}
    for d, _ in B2_KEYS:
        its = [i for i, it in BFI2["items"].items() if BFI2["facets"][it["facet"]]["domain"] == d]
        b2[d] = np.mean([keyed[i] for i in its], axis=0); assert np.nanmax(np.abs(b2[d] - comp[BFI2["domains"][d]["source_column"]].values)) < 1e-9
    # "Case" restarts in each of three administration batches and is not unique (duplicate-number pairs are unrelated people:
    # their BFI-2 profiles correlate ~0, like random pairs), so the stable row position in the released file is the identity.
    cases = [int(i) for i in comp.index]
    assert len(set(cases)) == len(cases)
    rank = sorted(range(len(comp)), key=lambda k: sha(f"{SALT}|{cases[k]}"))
    eval_rows = set(rank[:N_EVAL])

    # official BFI-2 item wording ("Is outgoing, sociable.") from the catalog, checked against the OSF survey PDF text
    pdf = (RAW / "sm.txt").read_text(); part1 = pdf[pdf.index("Part 1: Instructions"):pdf.index("Here are a number of activities")]
    lines = [l.strip() for l in part1.splitlines()]
    official = {}
    for k, l in enumerate(lines):
        m = re.match(r"^(\d+)\.\s+(.*)$", l)
        if m and 1 <= int(m.group(1)) <= 60:
            txt = m.group(2); j = k + 1
            while not txt.rstrip().endswith(".") and j < len(lines) and lines[j] and not re.match(r"^\d+\.", lines[j]): txt += " " + lines[j]; j += 1
            official[int(m.group(1))] = txt.replace("’", "'")
    bfi_text = {}
    for i in bfi_items:
        t = BFI2["items"][i]["text"]; assert t.startswith("I am someone who ")
        s = t[len("I am someone who "):]; s = s[0].upper() + s[1:]; bfi_text[i] = s
        n = int(i.split("_")[1])
        if official.get(n) is None or official[n].lower() != s.replace("’", "'").lower(): raise ValueError(f"BFI-2 wording mismatch {i}: {official.get(n)!r} vs {s!r}")
        if official[n] != s: print(f"catalog case typo {i}: using official {official[n]!r} (catalog {s!r})")
        bfi_text[i] = official[n]
    targets = {
        "bessi": [{"question": BESSI_INSTRUCTIONS, "items": {i: BESSI["items"][i]["text"] for i in bes_items},
                   "scale": {str(k + 1): a for k, a in enumerate(BESSI["anchors"])}}],
        "bfi2": [{"question": BFI2_INSTRUCTIONS, "items": {i: bfi_text[i] for i in bfi_items},
                  "scale": {str(k + 1): a for k, a in enumerate(BFI2["anchors"])}}],
    }
    w2, e2, r2 = bfi2_templates(); ws, es, rs = bessi_templates()
    tdir = OUT / "templates"; tdir.mkdir(parents=True, exist_ok=True)
    for name, txt in [("WRITER_BFI2.txt", w2), ("EDITOR_BFI2.txt", e2), ("READER_BFI2.txt", r2), ("WRITER_BESSI.txt", ws),
                      ("EDITOR_BESSI.txt", es), ("READER_BESSI.txt", rs), ("REPAIR.txt", REPAIR)]:
        (tdir / name).write_text(txt)
    def packets(inst, dom):
        if inst == "bfi2":
            keys, anchors, defs = B2_KEYS, BFI2_ANCHORS, "\n".join(BFI2_DEF[d] for d in ["E", "A", "C", "N", "O"])
            shared = "BFI-2 domain means on a 1–5 agreement scale. Higher means more of the named tendency; negative emotionality increases with distress.\n" + anchors + "\n"
            numeric = shared + "\n".join(f"- {name}: {dom[d]:.3f}" for d, name in keys) + "\n\n" + defs
            structured = shared + "\n".join(f"For {name}, this person has a measured BFI-2 domain mean of {dom[d]:.3f} on the 1–5 scale." for d, name in keys) + "\n\n" + defs
            meas = "\n".join(f"- {name}: mean={dom[d]:.3f}" for d, name in keys); W, E = w2, e2
        else:
            keys, anchors, defs = BS_KEYS, BESSI_ANCHORS, "\n".join(BESSI_DEF[d] for d in ["SM", "SE", "CO", "ER", "IN"])
            shared = "BESSI domain scores on a 1–5 scale of how well the person can do things when needed. Higher means greater reported skill.\n" + anchors + "\n"
            numeric = shared + "\n".join(f"- {name}: {dom[d]:.3f}" for d, name in keys) + "\n\n" + defs
            structured = shared + "\n".join(f"For {name}, this person has a measured BESSI domain score of {dom[d]:.3f} on the 1–5 scale." for d, name in keys) + "\n\n" + defs
            meas = "\n".join(f"- {name}: mean={dom[d]:.3f}" for d, name in keys); W, E = ws, es
        writer = W.replace("{{RESPONSE_ANCHORS}}", anchors).replace("{{AGGREGATE_MEASUREMENTS}}", meas)
        editor = E.replace("{{RESPONSE_ANCHORS}}", anchors).replace("{{AGGREGATE_MEASUREMENTS}}", meas)
        assert "{{" not in writer and editor.count("{{DRAFT}}") == 1 and "{{" not in editor.replace("{{DRAFT}}", "")
        return {"numeric": numeric, "structured": structured, "writer": writer, "editor_template": editor}
    people, reference, truth = [], [], {}
    for k in range(len(comp)):
        row = comp.iloc[k]; pid = f"college_{cases[k]:03d}"
        d2 = {d: float(b2[d][k]) for d, _ in B2_KEYS}; ds = {d: float(bs[d][k]) for d, _ in BS_KEYS}
        truth[pid] = {"bessi": {i: int(row[i]) for i in bes_items}, "bfi2": {i: int(row[i]) for i in bfi_items},
                      "bfi2_domains": d2, "bessi_domains": ds, "split": "eval" if k in eval_rows else "reference"}
        if k in eval_rows:
            people.append({"id": pid, "bfi2_domains": d2, "bessi_domains": ds,
                           "bfi2_to_bessi": packets("bfi2", d2), "bessi_to_bfi2": packets("bessi", ds)})
        else:
            reference.append(pid)
    people.sort(key=lambda p: sha(f"{SALT}|{int(p['id'].split('_')[1])}"))
    save(OUT / "PEOPLE.json", people); save(OUT / "PRIVATE_TRUTH.json", truth); save(OUT / "REFERENCE_IDS.json", reference)
    save(OUT / "TARGETS.json", targets); save(OUT / "READER_TARGETS.json", {"bfi2": reader_target("bfi2"), "bessi": reader_target("bessi")})
    save(OUT / "SELECTION_RECEIPT.json", {
        "source": "OSF 4zgyr College_student_sample.sav (final BESSI v1.0 how-well form, 192 items; BFI-2, 60 items)",
        "source_sha256": sha((RAW / "College_student_sample.sav").read_bytes()),
        "complete_cases": len(comp), "eval_n": len(people), "reference_n": len(reference),
        "selection_rule": f"first {N_EVAL} SHA-256 ranks of the row position in the released file with salt {SALT}; complete cases only; no outcome or score inspected",
        "age_range": [float(comp["Age"].min()), float(comp["Age"].max())],
        "scoring_check": "BESSI facets/weighted domains and keyed BFI-2 domains reproduce the authors' columns (max abs diff < 1e-9)",
        "bfi2_wording_check": "BFI-2 item wording taken from the OSF survey PDF text; the catalog matches it up to letter case (three capitalization typos in the catalog: bfi2_55, 56, 60); apostrophes straightened",
    })
    print(json.dumps(json.loads((OUT / "SELECTION_RECEIPT.json").read_text()), indent=1))

if __name__ == "__main__":
    main()
