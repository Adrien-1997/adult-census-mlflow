# Adult Census — MLflow Tracking

Hands-on training project to learn **MLflow Tracking** through a concrete
scikit-learn workflow.

## Scope
- Build clean sklearn pipelines (preprocessing + models)
- Train and compare multiple models on the Adult Census dataset
- Track experiments with MLflow (params, metrics, runs, experiments)
- Select a champion model based on cross-validation
- Log the final model as an MLflow artifact

## Models
- Logistic Regression (OneHotEncoder + StandardScaler)
- HistGradientBoostingClassifier (OrdinalEncoder)
- RandomForestClassifier (OneHotEncoder)
- XGBoostClassifier (OneHotEncoder, encoded target)

## What is tracked
- Cross-validation mean / std accuracy
- Model configuration (explicit logging)
- Final champion model artifact

## Key takeaway
> MLflow does not infer anything.  
> If it is not explicitly logged, it does not exist.

## Project structure
.
├── notebooks/
│   └── 01_baseline_sklearn_pipeline.ipynb
├── data/
│   └── raw/               # ignored (local dataset)
├── .gitignore
└── README.md

## Dataset
The dataset is based on the Adult Census Income dataset.

Source:
https://github.com/pooja2512/Adult-Census-Income/blob/master/adult.csv

Notes:
- The dataset is **not included** in this repository
- The original file `adult.csv` was renamed locally to `adult-census.csv`
- No modification was made to the data content

## Notes
- MLflow tracking is local (`mlruns/`, not versioned)
- Dataset is not included in the repository