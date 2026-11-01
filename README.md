# Replicating and extending DP-MLM (COMP8240 project)

Bayarbaatar Gonchigsuren (48728497), Macquarie University

Replication of Meisenbacher et al. (2024), *DP-MLM: Differentially Private Text Rewriting Using Masked Language
Models*, Findings of ACL 2024 ([paper](https://aclanthology.org/2024.findings-acl.554/),
[official code](https://github.com/sjmeis/DPMLM)), and extension to two new datasets:

* **IMDb** movie reviews (existing data, utility)
* **Reddit self-disclosure** (constructed): gender and age bracket mined from title tags such as `Me [24F]`,
  removed from the text, used as sensitive attributes for privacy attacks

## How to run

1. Open `dp_mlm_replication.ipynb` in Google Colab.
2. `Runtime → Change runtime type → T4 GPU`.
3. `Runtime → Run all`. With `QUICK = True` this takes about 45 minutes. Everything is written to
   `results/` and downloaded as `results.zip`.
4. Open `llm_evaluation.ipynb` (T4 GPU), upload `results.zip` in the first cell, `Run all` (about 15 minutes).
   It adds the LLM-as-judge, LLM annotator (Cohen's κ) and LLM attacker results and downloads `llm_results.zip`
   (unzip into `results/llm/`).
5. Build the report tables: `python analysis/make_report_assets.py results/ report/generated/`, then compile
   `report/main.tex` (pdflatex + bibtex).

No API keys are needed: all models (RoBERTa-base, MiniLM, DistilBERT SST-2, Qwen2.5-3B-Instruct) and datasets
(GLUE SST-2, `stanfordnlp/imdb`, `trl-lib/tldr`) are downloaded from Hugging Face.

| Notebook part | Content |
|---|---|
| 1 | Official DP-MLM on example sentences at ε ∈ {10, 25, 50, 100, 250} (+ PyTorch/NLTK compatibility patch) |
| 2 | Replication on GLUE SST-2: BLEU, SBERT cosine, static sentiment accuracy, % words changed |
| 3 | IMDb (existing new data) |
| 4 | Construction of the Reddit self-disclosure dataset (funnel, cleaning, labels, audit file) |
| 5 | Privacy: static + adaptive TF-IDF attackers, Relative Gain |
| 6 | LLM-as-judge (meaning, fluency, with controls), LLM annotator (Cohen's κ), zero-shot LLM attacker |
| 7 | Export of all numbers/tables/figures for the LaTeX report (`results/generated/`) |

### Human audit of the constructed labels (optional, recommended)
`results/reddit_audit_sample_100.csv` contains 100 posts **without** the extracted labels. Fill `human_gender`
(`M`/`F`/`?`) and `human_age_bracket` (`<=25`/`>25`/`?`) for the poster, upload the file back to `results/`
with the same name, and re-run the last cells of Part 6 and Part 7. Cohen's κ against the regex labels and the
LLM labels is then added to `results/label_agreement.csv`.

## Repository layout
```
dp_mlm_replication.ipynb   main notebook (DP-MLM rewriting, utility, dataset construction, attacks)
llm_evaluation.ipynb       LLM judge / annotator / attacker on the saved rewrites
analysis/                  make_report_assets.py: report tables + pronoun-leakage analysis
requirements.txt
report/                    LaTeX source of the final report (ACL template); generated/ holds notebook outputs
early_attempt/             first re-implementation from the paper (not used for results, see its README)
src/dpmlm/, LLMDP.py       copy of the authors' original code (github.com/sjmeis/DPMLM, see LICENSE) for reference;
                           the notebooks install the same code from PyPI (dpmlm 1.3.0)
```

## Data and ethics
The Reddit data come from the public TL;DR corpus (Völske et al., 2017; Stiennon et al., 2020). The constructed
dataset is not redistributed here; the notebook rebuilds it from the public source. Posts by users who state an
age under 18 should be excluded before any publication. See Section 5 of the report.
