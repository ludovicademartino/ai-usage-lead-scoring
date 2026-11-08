# Who uses AI at work? Lead scoring for a vendor of AI coding tools

Claudio Bendinelli · Ludovica De Martino · Cormac O'Sullivan — Université Paris-Saclay, M1 Hands-on Machine Learning 2026

A company that sells an AI coding assistant wants to know which developers to call first.
Using only what is visible on a public profile, we predict how much a developer uses AI tools
(**Daily / Occasional / No**) and rank a list of contacts.

**Data:** [Stack Overflow Developer Survey 2025](https://survey.stackoverflow.co/2025/) (licence ODbL 1.0).
The notebook downloads it (~140 MB) into `data/` the first time it runs.

## Contents of the notebook

1. **Preprocessing** — train/test split, leakage detection (45 AI-related columns removed),
   choice of observable features, cleaning of hidden missing values, `ColumnTransformer` pipeline
2. **Training** — baselines (incl. a hand-written "human rule"), comparison of 7 models,
   multiclass strategies, feature importance, ablation study, PCA/feature selection,
   personas with PCA + k-means + t-SNE, hyperparameter tuning, soft-voting ensemble, calibration
3. **Evaluation** — test set used once: confusion matrix, bootstrap confidence intervals,
   precision@k / lift, ROC-AUC, calibration, analysis of the worst errors and ethics

## Main results

| | Question | Answer |
|---|---|---|
| Q1 | Is a public profile enough? | Partly: test macro-F1 **0.467** [0.456; 0.479], against 0.383 for the human rule |
| Q2 | Value of the missing information | 0.473 → 0.550 with opinions; 0.757 with the AI questions (leakage) |
| Q3 | Types of developers | Web/JavaScript (most daily users), C/C++/systems (most "No"), .NET (in between) |

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook project_final_ML.ipynb   # about 30 minutes on a laptop
```
