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
  logger.py          # timestamped logs in logs/
  exception.py       # CustomException with file:line
  preprocessing.py   # clean_data, encode_target, AirlineSatisfactionPreprocessor, preprocess_data
  train.py           # 5 models (XGBoost optional)
  components/data_ingestion.py  # DataIngestion with logging
notebooks/
  01_EDA.ipynb               # RAW + cleaned EDA (34 cells)
  02_data_cleaning.ipynb     # cleaning demo via src/preprocessing
```

## What is Done So Far
1. **Scaffold & Git** — `requirements.txt`, `.gitignore` (ignores `logs/`), branch `feature/logging-exception`
2. **Logging & Exception (Unit 2)** — `CustomException`, timestamped `logs/*.log`, `DataIngestion` with logging
3. **Preprocessing** — drops `Unnamed: 0`/`id`, dedupes, median-imputes Arrival Delay (train median reused for test, no leakage), logs children/outliers, int casts, target map 1/0, `ColumnTransformer` pipeline
4. **EDA** — `01_EDA.ipynb` raw (missing, duplicates, target, numeric/categorical/service ratings, outliers, correlation, feature vs target) + §17 cleaned-data comparison (overlaid hists, boxplots, cleaned heatmap)
5. **Cleaning notebook** — `02_data_cleaning.ipynb` demonstrates `clean_data`/`preprocess_data`

## How to Run
```bash
pip install -r requirements.txt  # pandas, sklearn, matplotlib, seaborn, etc.
# always run from project root
python -c "import src.logger; import src.components.data_ingestion as d; print(d.DataIngestion().load_raw_data()[0].shape)"  # check logs/*.log
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_data_cleaning.ipynb
python src/train.py        # or python -m src.train
```

## Next Steps
- Feature engineering, model training/evaluation, dashboard

## Notes
- Raw CSVs never modified; `dataset/` is gitignored/untracked by design
- Outliers and children retained; no scaling/encoding in EDA
