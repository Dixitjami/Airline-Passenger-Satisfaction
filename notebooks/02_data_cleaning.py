# 02_data_cleaning.py - Data Cleaning for Airline Passenger Satisfaction
# Converted from 02_data_cleaning.ipynb - standalone script

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import glob

# Setup path for src imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import src modules
from src.preprocessing import clean_data, preprocess_data, load_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Load raw data
df_train_raw, df_test_raw = load_data()
print(f'raw train {df_train_raw.shape} test {df_test_raw.shape}')
print(f'Age<10 in raw train: {(df_train_raw["Age"] < 10).sum()}')

# Check missing and duplicates
print('\nMissing before train:')
print(df_train_raw.isnull().sum().to_string())
print(f'dups train: {df_train_raw.duplicated().sum()} test {df_test_raw.duplicated().sum()}')
print(df_train_raw.describe().T.to_string())

# Clean train data
print('\nCleaning train data...')
train_median = df_train_raw['Arrival Delay in Minutes'].median()
df_train_clean = clean_data(df_train_raw, arrival_median=train_median)

# Clean test data (reuse train median to avoid leakage)
df_test_clean = clean_data(df_test_raw, arrival_median=train_median)

print(f'\nClean train: {df_train_clean.shape}')
print(f'Clean test: {df_test_clean.shape}')
print(f'Missing after train: {int(df_train_clean.isnull().sum().sum())} test: {int(df_test_clean.isnull().sum().sum())}')
print(f'Children retained train: {int((df_train_clean["Age"] < 10).sum())}')
print(df_train_clean.head().to_string())

# Outlier check
print('\nOutlier analysis:')
for c in ['Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']:
    q1, q3 = df_train_clean[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    out = int(((df_train_clean[c] < q1 - 1.5*iqr) | (df_train_clean[c] > q3 + 1.5*iqr)).sum())
    print(f'{c}: outliers {out} ({out/len(df_train_clean)*100:.2f}%) - retained')

# Full pipeline check
X_train, y_train, X_test, y_test = preprocess_data(df_train_raw, df_test_raw)
print(f'\nX_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'X_test: {X_test.shape}')
print(f'Target distribution: {y_train.value_counts().to_dict()}')
print('Columns dropped: Unnamed: 0, id - target encoded 1/0 - raw CSVs unchanged')

# Show recent log
print('\nRecent log entries:')
lf = sorted(glob.glob(str(ROOT/'logs/*.log')))[-1]
print(lf)
with open(lf, encoding='utf-8') as f:
    print(f.read()[-2500:])

print('\nData cleaning complete!')