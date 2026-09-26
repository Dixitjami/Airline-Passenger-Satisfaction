# Airline Passenger Satisfaction Prediction

End-to-end ML project predicting `satisfaction` (satisfied vs neutral/dissatisfied) from Kaggle Airline Passenger Satisfaction data.

## Dataset
- `dataset/train.csv` — 103904 x 25, `dataset/test.csv` — 25976 x 25
- Target `satisfaction`: 56.7% neutral/dissatisfied, 43.3% satisfied (mild imbalance)
- Only missing: `Arrival Delay in Minutes` 310 (0.30%); 0 duplicates
- 1894 children Age 7-9 (1.8%) retained as valid passengers; delays heavily skewed with genuine long-tail outliers

## Project Structure
```
src/
  logger.py              # timestamped logs in logs/
  exception.py           # CustomException with file:line
  preprocessing.py       # clean_data, encode_target, AirlineSatisfactionPreprocessor, preprocess_data
  train.py               # 5 models (XGBoost optional)
  train_baseline.py      # Baseline Logistic Regression training
  components/data_ingestion.py  # DataIngestion with logging
notebooks/
  01_EDA.ipynb               # RAW + cleaned EDA (34 cells)
  02_data_cleaning.ipynb     # cleaning demo via src/preprocessing
models/
  baseline_logistic_regression.joblib  # Fitted baseline pipeline
reports/
  baseline_metrics.json      # Validation metrics
  figures/                   # EDA plots
```

## What is Done So Far
1. **Scaffold & Git** — `requirements.txt`, `.gitignore` (ignores `logs/`), branch `feature/logging-exception`
2. **Logging & Exception (Unit 2)** — `CustomException`, timestamped `logs/*.log`, `DataIngestion` with logging
3. **Preprocessing** — drops `Unnamed: 0`/`id`, dedupes, median-imputes Arrival Delay (train median reused for test, no leakage), logs children/outliers, int casts, target map 1/0, `ColumnTransformer` pipeline
4. **EDA** — `01_EDA.ipynb` raw (missing, duplicates, target, numeric/categorical/service ratings, outliers, correlation, feature vs target) + §17 cleaned-data comparison (overlaid hists, boxplots, cleaned heatmap)
5. **Cleaning notebook** — `02_data_cleaning.ipynb` demonstrates `clean_data`/`preprocess_data`
6. **Baseline Model** — Logistic Regression trained with leakage-safe pipeline, stratified 80/20 split, `class_weight="balanced"`

## How to Run
```bash
pip install -r requirements.txt  # pandas, sklearn, matplotlib, seaborn, etc.
# always run from project root
python -c "import src.logger; import src.components.data_ingestion as d; print(d.DataIngestion().load_raw_data()[0].shape)"  # check logs/*.log
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_data_cleaning.ipynb

# Baseline Logistic Regression (baseline model)
python src/train_baseline.py

# Full model comparison (all 5 models)
python src/train.py        # or python -m src.train
```

## Baseline Results (Logistic Regression)
Validation split: 20% stratified | Target mapping: 0=neutral or dissatisfied, 1=satisfied | Positive class: 1 (satisfied)

| Metric | Value |
|--------|-------|
| Accuracy | 0.8690 |
| Precision | 0.8396 |
| Recall | 0.8625 |
| F1-score | 0.8509 |
| ROC-AUC | 0.9281 |

Confusion Matrix (validation):
```
[[10292  1484]    # 0: neutral/dissatisfied
 [ 1238  7767]]   # 1: satisfied
```

Artifacts saved:
- Model: `models/baseline_logistic_regression.joblib`
- Metrics: `reports/baseline_metrics.json`

## Next Steps
- Feature engineering
- Train and compare Decision Tree, Random Forest, Gradient Boosting, XGBoost
- Model evaluation dashboard
- Hyperparameter tuning

## Notes
- Raw CSVs never modified; `dataset/` is gitignored/untracked by design
- Outliers and children retained; no scaling/encoding in EDA
- Test set (`dataset/test.csv`) kept completely separate for final evaluation only