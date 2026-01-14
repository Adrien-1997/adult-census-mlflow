# Adult Census — MLflow Tracking & Registry

Hands-on project to learn **MLflow Tracking, Model Registry**, and **model comparison with Skore**
through a clean, end-to-end scikit-learn workflow.

This repository is intentionally **pedagogical and opinionated**: everything is explicit, reproducible,
and designed to surface MLflow’s real strengths *and* limitations.

---

## Scope

- Build **clean sklearn pipelines** (preprocessing + models)
- Train and compare **multiple model families**
- Track experiments with **MLflow Tracking** (params, metrics, runs, experiments)
- Select a **champion model** based on cross-validation
- Log and **register the final model** in the MLflow Model Registry
- Compare models visually using **Skore Comparison Reports**
- Keep a **clean repo layout** (no generated artifacts committed)

---

## Models evaluated

- **Logistic Regression**
  - OneHotEncoder (categorical)
  - StandardScaler (numerical)

- **HistGradientBoostingClassifier**
  - OrdinalEncoder (categorical, unknown = -1)

- **RandomForestClassifier**
  - OneHotEncoder (categorical)

- **XGBoostClassifier**
  - OneHotEncoder (categorical)
  - Explicit binary target encoding (0 / 1)

All models are trained using **sklearn Pipelines**.

---

## What is tracked with MLflow

- Cross-validation accuracy (mean & std)
- Model hyperparameters and preprocessing choices
- Model identity (`model_key`, `model_family`)
- Final test accuracy of the champion model
- Serialized model artifacts
- Registered model versions

MLflow is configured to store everything locally under:

/mlflow/
├── mlflow.db        # SQLite backend store
└── artifacts/       # Models, figures, files

---

## Model comparison (Skore)

Model comparison is performed using **Skore**:

- Unified ROC curve (all models on a single plot)
- Unified Precision–Recall curve
- Metrics summary table (CSV)

Outputs are generated locally under:

/reports/skore/

and then logged to MLflow as artifacts.

Skore is used **in complement to MLflow**, not as a replacement:
- MLflow → tracking, lineage, registry
- Skore → evaluation quality & comparison

---

## Key takeaways

MLflow does not infer anything.
If it is not explicitly logged, it does not exist.

Tracking is a discipline, not a feature.
Tooling only enforces what you decide to make explicit.

---

## Project structure

.
├── notebooks/
│   └── 01_baseline_sklearn_pipeline.ipynb
│
├── data/
│   └── raw/                  # ignored (local dataset)
│
├── mlflow/
│   ├── mlflow.db             # local MLflow backend
│   └── artifacts/            # MLflow artifacts (models, plots)
│
├── reports/
│   └── skore/                # Skore comparison outputs
│
├── .gitignore
└── README.md

---

## Dataset

This project uses the Adult Census Income dataset.

Source:
https://github.com/pooja2512/Adult-Census-Income/blob/master/adult.csv

Notes:
- The dataset is not included in this repository
- The original file adult.csv was renamed locally to adult-census.csv
- No modification was made to the data content
- Target mapping used in this project:
  - <=50K → 0
  - >50K  → 1

---

## Reproducibility notes

- MLflow artifacts and database are not versioned
- Dataset is local and ignored via .gitignore
- All paths are resolved relative to the repository root
- The notebook can be executed from /notebooks without breaking MLflow tracking

---

## Status

MLflow Tracking
Model comparison
Champion selection
Model registration

Next steps: staging / production transitions, aliases, CI-friendly evaluation
