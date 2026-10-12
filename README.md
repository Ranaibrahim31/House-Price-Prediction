# House Price Prediction

Predicting residential house sale prices from 79 property features with scikit-learn, with an emphasis on **leak-free preprocessing, honest evaluation and error analysis** rather than only a headline score.

> **Status of results.** Every number below was produced by executing `notebooks/house_price_prediction.ipynb` from top to bottom with a fixed seed (42). The notebook runs in about 3-4 minutes on a single CPU core.

## Problem statement
Given the features of a house (size, quality, age, neighbourhood, basement/garage details, …), predict its `SalePrice` in USD. The labelled data has 1,460 houses; a separate file with 1,459 **unlabelled** houses is used only to produce a predictions file.

## Dataset
- `data/train.csv` — 1,460 rows × 81 columns (includes `SalePrice`).
- `data/test.csv` — 1,459 rows × 80 columns (no `SalePrice`, so **no test metrics can be computed**).
- The column names and sizes match the public Ames-housing data used in the Kaggle *House Prices: Advanced Regression Techniques* competition (sales between 2006 and 2010), but the origin of these particular files was not verified by this project. **Check the original source's terms before redistributing the data** (if in doubt, delete `data/` and ask users to download the files themselves).

## Workflow
1. Load and inspect data (shapes, duplicates, ids, train/test column consistency).
2. **Train/validation split (80/20, seed 42) before any target-based analysis.**
3. EDA, missing-value analysis, target analysis, correlation analysis and outlier investigation on the training part only.
4. Reproduce the original (first-version) notebook's numbers, then diagnose them.
5. Single `Pipeline` + `ColumnTransformer` preprocessing, fitted on training rows only.
6. Compare models with the same 5-fold CV folds and the same validation rows; test a log target, an outlier rule and feature engineering with pre-declared decision rules.
7. Small, purposeful hyperparameter search; model selection by cross-validation.
8. Error analysis, permutation importance, extrapolation check, predictions for the unlabelled test file.

## Highlights from the analysis
- `SalePrice` is right-skewed (skewness 1.74 → 0.12 after `log1p`), and a log target turned out to matter a lot for linear models.
- `OverallQual` (r ≈ 0.79) and `GrLivArea` (r ≈ 0.70) are the strongest numeric correlates of price; several predictors are strongly correlated with each other.
- Many `NaN`s in this dataset mean "feature absent" (e.g. no garage). This was verified against related columns (100 % agreement for garage, basement, pool, fireplace) and handled explicitly instead of imputing a median.
- 48 expensive houses flagged by the IQR rule are valid (larger, higher quality) and were **kept**. Only 2 very large, very cheap houses (Ids 524 and 1299) were excluded from model *fitting*, because that improved both CV (on the other rows) and validation for the two models tested.

![Actual vs predicted](images/06_actual_vs_predicted_residuals.png)

## Preprocessing
- `Id` dropped (a row label, not a feature); `MSSubClass` treated as a category.
- Categorical "absent" values (garage, basement, pool, fence, …) → explicit `"None"` category; numeric "absent" values (e.g. `GarageArea`, `TotalBsmtSF`) → 0.
- Genuinely unknown values → median (numeric, with a missing-indicator for `LotFrontage`/`GarageYrBlt`) or most frequent (categorical), learned from training rows only.
- One-hot encoding with `handle_unknown="ignore"`; `StandardScaler` only for linear models.
- 8 stateless engineered features (`TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`, `Has*` flags) — kept by a pre-declared rule, but their benefit was marginal (see the notebook).
- The old pre-processed CSVs were **not reused**: their scaler had been fitted on all 1,460 labelled rows (leakage into any later split) and they contain `Id` and the target next to the features. No replacement CSV is generated because preprocessing lives inside the pipeline.

## Models evaluated
Linear Regression (baseline), Ridge, Random Forest and HistGradientBoosting, each with raw and `log1p` targets (predictions are converted back to USD with `expm1` before any metric is computed), plus tuned versions of Ridge, Random Forest and HistGradientBoosting.

## Results
CV = 5-fold cross-validation inside the training part (suspect rows not scored); Val = the 292-house validation set, scored once per model. All metrics are in USD on the original price scale.

| Model | CV RMSE (USD) | Val MAE (USD) | Val RMSE (USD) | Val R² |
|---|---:|---:|---:|---:|
| Ridge(10) — log1p target | 19,827 ± 889 | 14,869 | 22,369 | 0.935 |
| Ridge (tuned) — log1p target | 19,827 ± 889 | 14,869 | 22,369 | 0.935 |
| HistGradBoost (tuned) — log1p target | 23,264 ± 2,770 | 15,369 | 26,958 | 0.905 |
| Ridge(10) — raw target | 23,754 ± 2,023 | 17,713 | 27,827 | 0.899 |
| HistGradBoost — log1p target | 24,499 ± 3,147 | 15,526 | 26,556 | 0.908 |
| HistGradBoost — raw target | 25,342 ± 3,379 | 15,841 | 27,019 | 0.905 |
| RandomForest(150) — raw target | 25,361 ± 2,282 | 16,933 | 28,728 | 0.892 |
| RandomForest(150) — log1p target | 26,051 ± 2,335 | 16,256 | 27,608 | 0.901 |
| RandomForest (tuned) — log1p target | 26,079 ± 2,452 | 16,555 | 28,193 | 0.896 |
| Linear — log1p target | 26,598 ± 6,360 | 15,357 | 23,310 | 0.929 |
| Linear — raw target | 32,646 ± 12,219 | 21,700 | 71,264 | 0.338 |

(`Ridge(10)` and `Ridge (tuned)` are identical because the search selected α = 10.)

![Model comparison](images/05_model_comparison.png)

## Final model
**Ridge regression (α = 10) on `log1p(SalePrice)`**, selected by the lowest cross-validated RMSE (≈ 19,827 USD) and confirmed by the validation set:

| Validation metric | Value |
|---|---:|
| MAE | 14,869 USD |
| RMSE | 22,369 USD |
| R² | 0.935 |

Typical error is about 6 % of the price (median absolute percentage error 6.4 %), but dollar errors are about twice as large for the most expensive quartile of houses.

## Important findings
- The original notebook's numbers (R² 0.4434 raw, 0.9316 log target) were **reproduced exactly**. The low raw-target R² came from one validation house predicted at about −839,000 USD by the unregularised linear model (RMSE 65,340 → 27,845 without that single row).
- Regularisation plus a log target mattered more than model complexity: tree ensembles did not beat Ridge on this small dataset.
- Permutation importance shows the model relies mainly on `OverallQual`, `GrLivArea`, `1stFlrSF`, `2ndFlrSF`, `TotalBsmtSF` and `YearBuilt`. This describes the fitted model; it is **not** evidence of causation.
- Extrapolation check: for test house Id 2550 (5,095 sq ft, beyond any training house) Ridge predicts ≈ 1.69 M USD while gradient boosting predicts ≈ 0.62 M USD. The predictions file keeps the raw model output; `outputs/prediction_warnings.csv` flags this and three other houses.

## Repository structure
```
House-Price-Prediction/
├── README.md
├── requirements.txt
├── notebooks/house_price_prediction.ipynb   # full workflow with executed outputs
├── data/train.csv, data/test.csv            # original files, unmodified
├── outputs/
│   ├── test_predictions.csv        # Id, SalePrice for the 1,459 unlabelled houses (same order as test.csv)
│   ├── validation_predictions.csv  # actual vs predicted on the validation houses
│   ├── model_comparison.csv        # CV and validation metrics of every model
│   ├── metrics_summary.json        # selected model, parameters, metrics, library versions
│   └── prediction_warnings.csv     # flagged test houses (outside training range / models disagree)
└── images/                         # figures generated by the notebook
```

## How to run
**Local**
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/house_price_prediction.ipynb  # then: Kernel > Restart & Run All
```
**Google Colab:** upload `train.csv` and `test.csv` (and the notebook) to the same Colab folder, open the notebook and choose *Runtime > Run all*. The notebook creates `outputs/` and `images/` next to itself.

## Limitations
- One 80/20 split (292 validation houses) and 5-fold CV on 1,168 houses: differences of 1-2k USD between top models are within noise. The validation set was reused for several decisions and to compare 11 models, so its metrics are slightly optimistic; there is no untouched labelled test set.
- `test.csv` has no `SalePrice`; the predictions file has **not** been evaluated.
- Data covers one market and one period (2006-2010); performance on other places or times is unknown. No prediction intervals are provided.
- A linear model can extrapolate unreliably outside the training range (see the Id 2550 example).
- The outlier exclusion rests on only two houses; feature engineering gave at most a small benefit.
- Tested only with the library versions listed in `requirements.txt` (executed with a minimal script runner in this environment; running it once in Jupyter/Colab is recommended before publishing).

## Author's contribution
- **Original work (first version of the notebook):** data loading and inspection, SalePrice and missing-value analysis, correlation analysis, IQR outlier calculation, the 80/20 split with `random_state=42`, a scikit-learn `ColumnTransformer` + `Pipeline` with median/`"None"` imputation, scaling and one-hot encoding, a `LinearRegression` baseline with MAE/RMSE/R², actual-vs-predicted and residual plots, and the `log1p` target experiment.
- **Audit and extension (developed with AI assistance, documented in the notebook):** leakage and validity checks, the explanation of the original R² of 0.44, absence-aware missing-value handling, Ridge / Random Forest / gradient-boosting comparison with cross-validation, rule-based outlier and feature-engineering experiments, tuning, error analysis, permutation importance, extrapolation check and the predictions files.

*(Edit this section so it reflects exactly what you did yourself.)*
