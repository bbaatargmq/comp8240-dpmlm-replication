"""Build every number/table/example used in the LaTeX report from the notebook outputs.

Usage:  python analysis/make_report_assets.py results/ report/generated/

Reads the CSVs written by dp_mlm_replication.ipynb (and, if present, results/llm/ written by
llm_evaluation.ipynb) and writes numbers.tex, tab_*.tex, ex_*.tex and copies the figures.
Also runs the pronoun-only leakage analysis (Section 4.2 of the report).
"""
import os, re, sys, shutil, difflib, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

RES, GEN = sys.argv[1], sys.argv[2]
LLM = os.path.join(RES, "llm")
os.makedirs(GEN, exist_ok=True)
EPS = [10, 25, 50, 100, 250]
EPS_TEX = r"$\varepsilon$="
PH = r"\textcolor{red}{??}"

def tex(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"$<$", ">": r"$>$"}
    return "".join(rep.get(c, c) for c in str(s))
def f(x, d=2): return "--" if x is None or pd.isna(x) else f"{x:.{d}f}"
def num(n): return f"{int(n):,}".replace(",", "{,}")
def write(name, lines): open(os.path.join(GEN, name), "w").write("\n".join(lines) + "\n")
def rd(name, sub=RES):
    p = os.path.join(sub, name); return pd.read_csv(p) if os.path.exists(p) else None
macros = {}
def mac(k, v): macros[k] = str(v)
TAG = {10: "Ten", 25: "TwentyFive", 50: "Fifty", 100: "Hundred", 250: "TwoFifty", "original": "Orig"}

# ---------------- utility (SST-2, IMDb) ----------------
def changed(o, p):
    """share of original words not kept in order (difflib alignment, robust to shifted positions)"""
    a, b = str(o).lower().split(), str(p).lower().split()
    same = sum(m.size for m in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks())
    return 1 - same / max(len(a), 1)

util = {}
for ds, tag in [("SST-2", "sst2"), ("IMDb", "imdb")]:
    u = rd(f"{tag}_utility.csv"); pv = rd(f"{tag}_privatized.csv")
    u["epsilon"] = u["epsilon"].astype(str)
    chg = {str(e): 100 * np.mean([changed(o, p) for o, p in zip(g.original, g.privatized)]) for e, g in pv.groupby("epsilon")}
    chg["original"] = 0.0
    u["changed"] = u["epsilon"].map(chg)
    util[ds] = u.set_index("epsilon")
    key = "SST" if ds == "SST-2" else "IMDB"
    mac(f"N{key}", pv.id.nunique())
    mac(f"{key}Len", f"{np.mean([len(str(t).split()) for t in pv.drop_duplicates('id').original]):.1f}")
    for e in ["original"] + EPS:
        r = util[ds].loc[str(e)]
        mac(f"{key}Acc{TAG[e]}", f"{100 * r.Acc:.1f}"); mac(f"{key}Bleu{TAG[e]}", f"{r.BLEU / 100:.2f}")
        mac(f"{key}Cos{TAG[e]}", f"{r.CosSim:.2f}"); mac(f"{key}Chg{TAG[e]}", f"{r.changed:.0f}")
mac("IMDBCutLen", macros["IMDBLen"].split(".")[0])

def row(lbl, src, vals): return f"{lbl} & {src} & " + " & ".join(vals) + r" \\"
S = util["SST-2"]; cols = ["original"] + [str(e) for e in EPS]
PAPER = {"Acc": [68.50, 76.49, 84.10, 86.16, 86.05], "BLEU": [0.08, 0.11, 0.19, 0.22, 0.23], "CS": [0.17, 0.37, 0.61, 0.65, 0.67]}
L = [r"\begin{tabular}{llrrrrrr}", r"\toprule", r" & & Orig. & $\varepsilon$=10 & 25 & 50 & 100 & 250 \\", r"\midrule",
     row("BLEU", "paper", ["--"] + [f(v) for v in PAPER["BLEU"]]),
     row("", "ours", [f(S.loc[c].BLEU / 100) for c in cols]),
     row("CosSim", "paper", ["--"] + [f(v) for v in PAPER["CS"]]),
     row("", "ours", [f(S.loc[c].CosSim) for c in cols]),
     row(r"Acc (\%)", r"paper$^\dagger$", ["95.7"] + [f(v, 1) for v in PAPER["Acc"]]),
     row("", r"ours$^\ddagger$", [f(100 * S.loc[c].Acc, 1) for c in cols]),
     row(r"\% changed", "ours", [f(S.loc[c].changed, 0) for c in cols]), r"\bottomrule", r"\end{tabular}"]
write("tab_sst2.tex", L)
L = [r"\begin{tabular}{llrrrrrr}", r"\toprule", r" & & Orig. & 10 & 25 & 50 & 100 & 250 \\", r"\midrule"]
for metric, lbl, sc, d in [("BLEU", "BLEU", 0.01, 2), ("CosSim", "CosSim", 1, 2), ("Acc", r"Acc (\%)", 100, 1), ("changed", r"\% chg.", 1, 0)]:
    for ds in ["SST-2", "IMDb"]:
        L.append(row(lbl if ds == "SST-2" else "", ds, [f(sc * util[ds].loc[c][metric], d) for c in cols]))
write("tab_imdb.tex", L + [r"\bottomrule", r"\end{tabular}"])

# ---------------- examples ----------------
def ex_block(orig, rewrites, w=0.11):
    out = [rf"\begin{{tabular}}{{@{{}}p{{{w}\linewidth}}p{{{0.93 - w:.2f}\linewidth}}@{{}}}}", r"\toprule", f"Original & {tex(orig)} \\\\", r"\midrule"]
    return out + [f"{EPS_TEX}{e} & {tex(r)} \\\\" for e, r in rewrites] + [r"\bottomrule", r"\end{tabular}"]
demo = rd("demo_rewrites.csv")
for i, (o, g) in enumerate(demo.groupby("original", sort=False)):
    write(f"ex_demo{i}.tex", ex_block(o, zip(g.epsilon, g.rewritten), 0.16))
sp = rd("sst2_privatized.csv"); lens = sp.drop_duplicates("id").assign(n=lambda x: x.original.str.split().str.len())
i_ex = int(lens.iloc[(lens.n - 14).abs().argsort()].iloc[0].id)
g = sp[sp.id == i_ex]; write("ex_sst2.tex", ex_block(g.original.iloc[0], zip(g.epsilon, g.privatized), 0.16))

# ---------------- Reddit dataset ----------------
red = rd("reddit_selfdisclosure_dataset.csv"); fun = rd("reddit_funnel.csv")
fun.columns = ["step", "posts"]
L = [r"\begin{tabular}{lr}", r"\toprule", r"Step & Posts \\", r"\midrule"] + [f"{tex(s)} & {num(n)} \\\\" for s, n in zip(fun.step, fun.posts)]
write("tab_funnel.tex", L + [r"\bottomrule", r"\end{tabular}"])
names = ["FunParsed", "FunTag", "FunNoAge", "FunLen", "FunDedup", "FunBalanced"]
for k, n in zip(names, fun.posts): mac(k, num(n))
sm = open(os.path.join(RES, "summary.md")).read() if os.path.exists(os.path.join(RES, "summary.md")) else ""
m = re.search(r"Title-vs-body label agreement: ([0-9.]+) \(n=(\d+)\)", sm)
mac("AgreeRate", f"{100 * float(m[1]):.1f}" if m else PH); mac("AgreeN", num(m[2]) if m else PH)
mac("PctTagged", f"{100 * fun.posts[1] / fun.posts[0]:.1f}")
mac("AgeMedian", f"{red.age.median():g}"); mac("AgeMin", int(red.age.min())); mac("AgeMax", int(red.age.max()))
mac("PctMinor", f"{100 * (red.age < 18).mean():.1f}"); mac("NMinor", int((red.age < 18).sum()))
mac("RedditLen", f"{red.n_words.mean():.1f}")
vc = red.subreddit.value_counts(normalize=True)
mac("PctRelationships", f"{100 * vc.iloc[0]:.1f}"); mac("NSubreddits", red.subreddit.nunique())
mac("NSubLabels", red.sub_label.nunique()); mac("SubLabels", tex(", ".join(sorted(red.sub_label.unique()))))
mac("PctOther", f"{100 * (red.sub_label == 'other').mean():.1f}")
mac("NTest", 200); mac("NAdapt", 400); mac("NTrain", len(red) - 200)
L = [r"\begin{tabular}{@{}p{0.15\linewidth}p{0.81\linewidth}@{}}", r"\toprule"]
for (gd, b), grp in red.groupby(["gender", "age_bracket"]):
    r = grp.sort_values("n_words").iloc[len(grp) // 20]
    L += [f"\\textbf{{{gd}, {tex(b)}}} \\newline r/{tex(r.subreddit)} & {tex(r.text)} \\\\", r"\midrule"]
L[-1] = r"\bottomrule"; write("ex_reddit.tex", L + [r"\end{tabular}"])

# ---------------- privacy ----------------
att = rd("reddit_attack_results.csv"); rg = rd("reddit_relative_gain.csv")
tp = rd("reddit_test_privatized.csv")
test_text = tp[tp.epsilon == 10].original.tolist()
test = red.drop_duplicates("text").set_index("text").loc[test_text]
train = red[~red.text.isin(set(test_text))]
PR = ["he", "him", "his", "himself", "she", "her", "hers", "herself"]
def pfeat(texts):
    out = []
    for s in texts:
        w = re.findall(r"[a-z']+", str(s).lower()); out.append([w.count(p) for p in PR])
    return np.array(out)
pclf = LogisticRegression(max_iter=1000).fit(pfeat(train.text), train.gender)
pron = {"original": 100 * f1_score(test.gender, pclf.predict(pfeat(test_text)), average="macro")}
for e in EPS:
    pron[e] = 100 * f1_score(test.gender, pclf.predict(pfeat(tp[tp.epsilon == e].privatized)), average="macro")
otxt = " ".join(test_text).lower(); ptxt = " ".join(tp[tp.epsilon == 10].privatized.astype(str)).lower()
cnt = lambda ws, t: sum(len(re.findall(rf"\b{w}\b", t)) for w in ws)
PRON = ["he", "him", "his", "she", "her", "hers"]; REL = ["boyfriend", "girlfriend", "husband", "wife", "bf", "gf", "fiance", "fiancee"]
mac("PronKeep", f"{100 * cnt(PRON, ptxt) / cnt(PRON, otxt):.0f}"); mac("RelKeep", f"{100 * cnt(REL, ptxt) / cnt(REL, otxt):.0f}")
mac("PronN", cnt(PRON, otxt)); mac("RelN", cnt(REL, otxt))
for e in ["original"] + EPS: mac(f"Pron{TAG[e]}", f"{pron[e]:.1f}")
json.dump({str(k): v for k, v in pron.items()}, open(os.path.join(GEN, "pronoun_attack.json"), "w"), indent=1)

llm_att = rd("reddit_llm_attacker.csv", LLM)
L = [r"\begin{tabular}{llrrrrrr}", r"\toprule", r"Task & Attacker & Orig. & 10 & 25 & 50 & 100 & 250 \\", r"\midrule"]
for task, name in [("utility: subreddit", "Subreddit"), ("privacy: gender", "Gender"), ("privacy: age bracket", "Age bracket")]:
    for st in ["static", "adaptive"]:
        a = att[(att.task == task) & (att.setting == st)].set_index("epsilon")["F1"]
        L.append(f"{name if st == 'static' else ''} & {st} & " + " & ".join(f(a.get(c), 1) for c in cols) + r" \\")
    if task == "privacy: gender":
        L.append(" & pronouns only & " + " & ".join(f(pron[e if e == 'original' else int(e)], 1) for e in cols) + r" \\")
    if task.startswith("privacy") and llm_att is not None:
        b = llm_att[llm_att.attribute == task.split(": ")[1]].set_index("epsilon")["F1"]
        L.append(" & LLM zero-shot & " + " & ".join(f(b.get(c), 1) for c in cols) + r" \\")
    L.append(r"\midrule")
for st in ["static", "adaptive"]:
    for attr in ["gender", "age bracket"]:
        r = rg[(rg.setting == st) & (rg.attribute == attr)].set_index("epsilon")["RG"]
        L.append(f"RG {attr} & {st} & -- & " + " & ".join(f(r.get(e)) for e in EPS) + r" \\")
write("tab_reddit.tex", L + [r"\bottomrule", r"\end{tabular}"])
for task, k in [("utility: subreddit", "Sub"), ("privacy: gender", "Gen"), ("privacy: age bracket", "Age")]:
    for st, s in [("static", "S"), ("adaptive", "A")]:
        a = att[(att.task == task) & (att.setting == st)].set_index("epsilon")
        for e in cols: mac(f"F{k}{s}{TAG[e if e == 'original' else int(e)]}", f"{a.loc[e].F1:.1f}")
a = att[(att.task == "utility: subreddit") & (att.setting == "static")].set_index("epsilon")
mac("SubAccOrig", f"{a.loc['original'].Acc:.1f}")

# ---------------- LLM outputs (llm_evaluation.ipynb) ----------------
env = os.path.join(LLM, "env.json")
E = json.load(open(env)) if os.path.exists(env) else {}
for k in ["EnvGPU", "EnvTorch", "EnvTransformers", "EnvPython", "EnvDpmlm", "EnvSbert", "EnvSacrebleu", "ClipMin", "ClipMax", "Sensitivity", "JudgeModel"]:
    mac(k, tex(E[k]) if k in E else PH)
rq = rd("reddit_quality.csv", LLM)
for e in EPS:
    v = rq.set_index("epsilon").loc[e] if rq is not None else None
    mac(f"RedBleu{TAG[e]}", f(v.BLEU / 100) if v is not None else PH); mac(f"RedCos{TAG[e]}", f(v.CosSim) if v is not None else PH)
jt = rd("judge_summary.csv", LLM); js = rd("judge_stats.csv", LLM)
order = ["identical (control)", "eps=10", "eps=50", "eps=250", "mismatched (control)"]
L = [r"\begin{tabular}{llrrr}", r"\toprule", r"Data & Condition & Meaning & Fluency & CosSim \\", r"\midrule"]
for ds in ["SST-2", "Reddit"]:
    for c in order:
        r = jt[(jt.dataset == ds) & (jt.condition == c)].iloc[0] if jt is not None else None
        vals = [f(r.meaning), f(r.fluency), f(r.cossim)] if r is not None else ["?"] * 3
        L.append(f"{ds if c == order[0] else ''} & {tex(c).replace('eps=', EPS_TEX)} & " + " & ".join(vals) + r" \\")
        if r is not None:
            mac(f"J{'Sst' if ds == 'SST-2' else 'Red'}{re.sub('[^A-Za-z]', '', c.split('(')[0].title()) + (TAG[int(c[4:])] if c.startswith('eps') else '')}M", f(r.meaning))
            mac(f"J{'Sst' if ds == 'SST-2' else 'Red'}{re.sub('[^A-Za-z]', '', c.split('(')[0].title()) + (TAG[int(c[4:])] if c.startswith('eps') else '')}F", f(r.fluency))
    if ds == "SST-2": L.append(r"\midrule")
write("tab_judge.tex", L + [r"\bottomrule", r"\end{tabular}"])
for k in ["NJudge", "JudgeParse", "JudgeRho", "JudgeRhoN"]:
    v = js.set_index("stat").loc[k].value if js is not None else PH
    if js is not None and k in ("NJudge", "JudgeRhoN"): v = int(float(v))
    if js is not None and k == "JudgeParse": v = f"{float(v):g}"
    mac(k, v)
ag = rd("label_agreement.csv", LLM)
L = [r"\begin{tabular}{llrrr}", r"\toprule", r"Attribute & Pair & $n$ & $\kappa$ & Agree (\%) \\", r"\midrule"]
if ag is not None:
    for _, r in ag.iterrows():
        L.append(f"{tex(r.attribute)} & {tex(r.pair)} & {r.n} & {r.kappa:.2f} & {100 * r['raw agreement']:.0f} \\\\")
        nm = {"regex vs LLM": "RegLLM", "regex vs human": "RegHum", "LLM vs human": "LLMHum"}[r.pair] + ("G" if r.attribute == "gender" else "A")
        mac("Kappa" + nm, f"{r.kappa:.2f}"); mac("Raw" + nm, f"{100 * r['raw agreement']:.0f}")
    mac("NHuman", int(ag[ag.pair == "regex vs human"].n.max()) if (ag.pair == "regex vs human").any() else 0)
else:
    L.append(r"\multicolumn{5}{c}{\textcolor{red}{run llm\_evaluation.ipynb}} \\")
for k in ["KappaRegLLMG", "KappaRegLLMA", "KappaRegHumG", "KappaRegHumA", "KappaLLMHumG", "KappaLLMHumA", "NHuman"]:
    macros.setdefault(k, PH)
write("tab_agreement.tex", L + [r"\bottomrule", r"\end{tabular}"])
if llm_att is not None:
    for _, r in llm_att.iterrows():
        e = r.epsilon if r.epsilon == "original" else int(r.epsilon)
        mac(f"LLMAtt{'G' if r.attribute == 'gender' else 'A'}{TAG[e]}", f"{r.F1:.1f}")
mac("NLLMAtt", 100)
for a_ in "GA":
    for e in ["original", 10, 25, 50, 100, 250]: macros.setdefault(f"LLMAtt{a_}{TAG[e]}", PH)
rp = os.path.join(LLM, "ex_reddit_priv.csv")
if os.path.exists(rp):
    x = pd.read_csv(rp); write("ex_reddit_priv.tex", ex_block(x.original.iloc[0], zip(x.epsilon, x.privatized), 0.16))
else:
    x = tp[tp.original == test_text[0]]; write("ex_reddit_priv.tex", ex_block(x.original.iloc[0], [(e, p) for e, p in zip(x.epsilon, x.privatized) if e in (10, 50, 250)], 0.16))

with open(os.path.join(GEN, "numbers.tex"), "w") as fh:
    fh.write("% auto-generated by analysis/make_report_assets.py\n")
    for k, v in macros.items():
        if not k.isalpha(): raise ValueError(k)
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
for png in ["utility_sst2_imdb.png", "reddit_dataset_overview.png", "reddit_privacy_utility.png"]:
    if os.path.exists(os.path.join(RES, png)): shutil.copy(os.path.join(RES, png), os.path.join(GEN, png))
print(f"wrote {len(macros)} macros to {GEN}")
