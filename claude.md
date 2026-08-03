# CLAUDE.md — Bug-Risk Predictor for Code Commits
 
## Who this is for
Aditi is building this project to genuinely learn it, not just get a working repo — she needs
to be able to explain every design decision in a technical interview (this is for a Microsoft
CoreAI internship application). When implementing anything non-obvious, explain *why* in plain
language before or alongside the code, not just what.
 
Work in small phases (see Roadmap below). After each phase, stop, show output/results, and
wait for confirmation before moving to the next phase. Do not silently skip ahead.
 
## Project goal
Predict whether a code commit is likely to introduce a bug ("buggy" vs "clean"), using only
commit-level metadata (size of change, author experience, files touched, etc.) — not the code
diff itself. This replicates the "Just-In-Time (JIT) Defect Prediction" research approach
(Kamei et al.), applied to a real dataset of Apache open-source commits.
 
## Dataset — ApacheJIT
- Source: Keshavarz, H., & Nagappan, M. (2022). *ApacheJIT: A Large Dataset for Just-In-Time
  Defect Prediction*. Zenodo. https://doi.org/10.5281/zenodo.5907002
- License: CC-BY 4.0 — **must be credited in the README**, exactly as above.
- Files (fetch the authoritative list from the Zenodo REST API rather than hardcoding URLs,
  since file paths can change between versions):
  `https://zenodo.org/api/records/5907002/files`
  Expected files: `apachejit_train.csv` (balanced, 2003–2016), `apachejit_total.csv` (full,
  imbalanced, ~106k commits, ~26% buggy), `apachejit_test_large.csv` / `apachejit_test_small.csv`
  (unbalanced, last 3 years — use one of these as the held-out evaluation set).
- Target column: `buggy` (0/1).
- **Do not assume exact feature column names.** Load `apachejit_train.csv` first and print
  `.columns`, `.info()`, `.head()`, and the class balance of `buggy`. Confirm what's actually
  there before writing any feature-selection code. These are expected to be JIT "expert
  features" in the tradition of Kamei et al. (commit size, files/subsystems touched, developer
  experience, entropy of change, whether it's a fix, etc.) — verify against the real columns.
## Critical design decisions — do not deviate without discussing with Aditi first
1. **Do not re-split the data randomly.** Use the provided train file (2003–2016) for training
   and a provided test file (last 3 years) for evaluation. This is intentional time-based
   splitting to avoid data leakage — a random shuffle-split would leak future information into
   training. Preserve this structure.
2. **Do not report plain accuracy as the headline metric.** The real-world test set is
   imbalanced (~26% buggy), so accuracy is misleading. Report Precision, Recall, F1, and
   ROC-AUC for every model, plus a confusion matrix.
3. **Scale features for Logistic Regression and the PyTorch MLP** (StandardScaler, fit on
   train only, applied to test). Random Forest does not need scaling — don't scale unnecessarily
   for it, and explain why in a code comment (tree splits are scale-invariant).
## Tech stack
Python 3.10+, pandas, scikit-learn, matplotlib, and (Phase 5 only) PyTorch. Keep dependencies
minimal — no deep-learning framework needed until Phase 5.
 
## Repo structure to build
```
bug-risk-predictor/
├── README.md
├── requirements.txt
├── .gitignore              # must exclude data/ (large files, don't commit raw CSVs)
├── scripts/
│   └── download_data.py    # pulls CSVs from Zenodo via the REST API
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train_logreg.py
│   ├── train_forest.py
│   ├── train_mlp.py        # Phase 5, optional
│   └── evaluate.py         # shared metric/plotting functions, used by all train scripts
├── results/
│   ├── metrics.json        # precision/recall/f1/auc per model
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── feature_importance.png   # from Random Forest
└── CLAUDE.md
```
 
## Roadmap (work through in order, pause after each for confirmation)
1. **Scaffold + data**: create repo structure, write `download_data.py`, run it, load
   `apachejit_train.csv`, print columns/info/class balance. Stop and show this before continuing.
2. **Preprocessing**: build `preprocess.py` — handle any missing values, define the feature list
   from the *actual* verified columns, fit a StandardScaler on train.
3. **Logistic Regression baseline**: train with `class_weight='balanced'` if the train set isn't
   already balanced, evaluate on the held-out test file, save metrics + confusion matrix.
4. **Random Forest**: train, evaluate the same way, save a feature-importance plot. Compare
   against Logistic Regression in a simple table.
5. **[Optional, time-permitting] PyTorch MLP**: small feed-forward network (2–3 hidden layers,
   ReLU, dropout, sigmoid output), BCE loss, Adam optimizer, a real training loop with visible
   epoch/loss logging (not just a black-box `.fit()`). Evaluate identically to the other models.
6. **README**: problem statement in plain English, dataset citation, architecture explanation
   (include the three "why" points: balanced train / unbalanced test, time-based split, why
   multiple models), results table, how to reproduce, what you'd improve with more time.
## Conventions
- Every script should be runnable standalone (`python src/train_forest.py`) and also importable.
- Save all metrics to `results/metrics.json` (append, don't overwrite) so the final comparison
  table can be generated from one source of truth.
- Comment *why*, not just *what*, especially for the three design decisions above — these
  comments double as Aditi's interview prep.
- Keep commits small and descriptive (this repo will be linked on a resume — commit history is
  part of the impression it makes).