# Model Selection and Tuning

Produced by `python -m src.train`. Every setting below is also logged in MLflow
(experiment `heart-disease`) as a nested child run, and the full tables are in
[`report/tuning/`](tuning/).

## Setup
- **Data:** `data/processed/heart_clean.csv`, 303 patients, 13 features.
- **Split:** 80% train (242) / 20% test (61), stratified so both keep the
  54/46 class ratio. The test set is not used until the final evaluation.
- **Preprocessing** (`src/features.py`, inside the pipeline so it is fitted
  only on training folds):
  - numeric (`age, trestbps, chol, thalach, oldpeak`): median imputer + StandardScaler
  - categorical (`sex, cp, fbs, restecg, exang, slope, ca, thal`): most-frequent
    imputer + OneHotEncoder(handle_unknown="ignore")
- **Tuning:** `GridSearchCV`, stratified 5-fold CV, scored on ROC-AUC.
  Accuracy, precision and recall are recorded for every fold as well.

## Grids searched

| Model | Parameter | Values | Why |
|---|---|---|---|
| Logistic Regression | `C` | 0.01, 0.1, 1, 10 | Regularisation strength: small C = simpler model, less overfitting |
| Random Forest | `n_estimators` | 100, 300 | More trees = more stable predictions |
| | `max_depth` | None, 5, 10 | Limits how complex each tree can get |
| | `min_samples_leaf` | 1, 5 | Stops trees from making rules from 1-2 patients |

4 + 12 = **16 settings**, each trained 5 times (80 fits).

## Tuning results (cross-validation, mean of 5 folds)

**Logistic Regression** ([full table](tuning/logistic_regression_grid_results.csv))

| C | ROC-AUC (± std) | Recall | Accuracy |
|---|---|---|---|
| **1.0** | **0.9025 ± 0.014** | 0.783 | 0.847 |
| 0.1 | 0.9025 ± 0.017 | 0.774 | 0.839 |
| 10.0 | 0.8995 ± 0.014 | 0.802 | 0.855 |
| 0.01 | 0.8893 ± 0.035 | 0.693 | 0.806 |

Too much regularisation (C=0.01) hurts recall badly; C between 0.1 and 10 is
a flat plateau, so the model is not sensitive to the exact value.

**Random Forest**: top 3 of 12 ([full table](tuning/random_forest_grid_results.csv))

| max_depth | min_samples_leaf | n_estimators | ROC-AUC (± std) | Recall |
|---|---|---|---|---|
| **None** | **5** | **300** | **0.9023 ± 0.033** | 0.774 |
| 10 | 5 | 300 | 0.9023 ± 0.033 | 0.774 |
| 5 | 5 | 300 | 0.9009 ± 0.034 | 0.774 |

All top settings use `min_samples_leaf=5`: forcing leaves to hold several
patients reduces overfitting on such a small dataset.

## Final comparison

| Model | Split | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | CV | 0.847 | 0.877 | 0.783 | 0.902 |
| | Test | 0.885 | 0.839 | 0.929 | 0.966 |
| Random Forest | CV | 0.810 | 0.810 | 0.774 | 0.902 |
| | Test | 0.852 | 0.806 | 0.893 | 0.948 |

## Decision: Logistic Regression

1. **Chosen on CV ROC-AUC, not test score.** Picking by test score would
   leak the test set into the decision. The two models are effectively tied
   (0.9025 vs 0.9023), so the other criteria decide.
2. **Higher recall.** Missing a sick patient (false negative) is worse than a
   false alarm, which only leads to more tests. Logistic Regression has
   higher recall in CV (0.783 vs 0.774) and on the test set (0.929, i.e. 26
   of 28 sick patients found).
3. **More stable.** Its fold-to-fold ROC-AUC spread is less than half that of
   the forest (± 0.014 vs ± 0.033).
4. **Simpler and explainable.** Each feature has one weight that can be
   explained to a clinician. A small dataset does not need a complex model.

Test scores are higher than CV scores because 61 patients is a small test set
and moves a lot with one or two patients; **~0.90 ROC-AUC** (the CV estimate)
is the expected performance on new patients.

Saved model: `models/model.joblib` (full pipeline), described by `models/metadata.json`.
