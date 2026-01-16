# Adult Census — MLflow Tracking & Model Registry

Hands-on project to learn **MLflow Tracking**, the **Model Registry**, and **robust model comparison**
through a clean, end-to-end **scikit-learn workflow**.

This repository is intentionally **pedagogical and opinionated**: everything is explicit, reproducible,
and designed to highlight MLflow’s strengths *(and limitations)* in a realistic notebook setup.

---

## Highlights (what you’ll learn)

- Build **clean sklearn pipelines** (preprocessing + model)
- Evaluate models with **stratified cross-validation**
  - 5 folds, `shuffle=True`, `random_state=42` (same splitter used everywhere)
- Track experiments with **MLflow Tracking** (params, metrics, tags, artifacts)
- Generate **Out-Of-Fold (OOF)** predicted probabilities for fair model comparison
- Plot **OOF ROC + Precision–Recall curves** (side-by-side) for 4 model families
- Select a **champion** by maximizing **mean CV PR-AUC** (Average Precision)
- Refit the champion on the train split, evaluate on a held-out test split
- Log and (optionally) **register the model** in the MLflow Model Registry

---

## Models evaluated

All models are implemented as **sklearn Pipelines**.

- **LogisticRegression**
  - `OneHotEncoder(handle_unknown="ignore")` (categorical)
  - `StandardScaler()` (numerical)

- **HistGradientBoostingClassifier**
  - `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` (categorical)

- **RandomForestClassifier**
  - `OneHotEncoder(handle_unknown="ignore")` (categorical)

- **XGBClassifier**
  - `OneHotEncoder(handle_unknown="ignore")` (categorical)

---

## What is tracked with MLflow

Each model gets its own CV run where we log:

- Cross-validation metrics **mean/std**:
  - ROC-AUC
  - PR-AUC (Average Precision)
  - F1
  - Recall
- Minimal, human-readable params (only what you changed vs defaults)
- Clear metadata (`model_key`, `model_family`, CV splitter settings)

The champion run logs:

- Final test metrics (ROC-AUC, PR-AUC, F1, Recall)
- A lightweight param summary (clean MLflow UI)
- A **full pipeline params JSON artifact** for reproducibility
- The serialized model artifact (sklearn flavor)

MLflow is configured to store everything locally under:

```
mlflow/
├── mlflow.db        # SQLite backend store
└── artifacts/       # runs artifacts (models, files)
```

---

## Model comparison (OOF curves)

We compute **Out-Of-Fold predicted probabilities** for each model using the **same CV splitter**, then plot:

- **ROC curves** (AUC)
- **Precision–Recall curves** (Average Precision)

Plots are saved under:

```
reports/
├── oof_roc_pr_curves.png
└── champion_params_full_<model_key>.json
```

> Note: OOF curves reflect cross-validated generalization, while the champion run reports
> metrics on the held-out test split.

---

## Champion selection strategy

We select the champion by maximizing:

- `cv_pr_auc_mean` (Average Precision mean over CV folds)

This is usually a better choice than accuracy for **imbalanced binary classification**.

After selection:

1. Fit champion on `data_train`
2. Evaluate on `data_test`
3. Log metrics + model artifact
4. (Optional) register the champion into the MLflow Registry

---

## Project structure

```
.
├── notebooks/
│   ├── 01_baseline_sklearn_pipeline.ipynb
│   └── 02_mlflow_model_registry.ipynb
│
├── data/
│   └── raw/                  # local dataset
│
├── mlflow/
│   ├── mlflow.db             # local MLflow backend (ignored)
│   └── artifacts/            # MLflow artifacts (ignored)
│
├── reports/                  # saved plots + JSON params artifacts
│
├── .gitignore
└── README.md
```

---

## Dataset

This project uses the **Adult Census Income** dataset.

Source:
https://github.com/pooja2512/Adult-Census-Income/blob/master/adult.csv

Notes:

- The original file `adult.csv` was renamed locally to `adult-census.csv`
- The column `education.num` is dropped for simplicity
- Target mapping used in this project:
  - `<=50K` → 0
  - `>50K`  → 1

---

## Run locally

### Install dependencies

```bash
pip install -U pandas numpy scikit-learn matplotlib mlflow xgboost
```

### Run notebooks

Open and run the notebooks from `notebooks/`.

### Inspect in MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/mlruns
```

Open:

- http://127.0.0.1:5000

---

## Key takeaways

MLflow does not infer anything.
If it’s not explicitly logged, it doesn’t exist.

Tracking is a discipline, not a feature.
Tooling only enforces what you decide to make explicit.

---

## Status

- [x] MLflow Tracking (CV runs + champion run)
- [x] OOF ROC/PR model comparison (saved in `reports/`)
- [x] Champion selection
- [x] Model registration (local registry)

Next steps: aliases (Champion/Challenger), deployment packaging, and CI-friendly evaluation.
