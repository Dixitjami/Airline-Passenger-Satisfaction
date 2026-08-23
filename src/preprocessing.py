"""
Data loading, cleaning, and preprocessing utilities for the
Airline Passenger Satisfaction Prediction project.

All functions are written to be reusable in notebooks, the Streamlit
dashboard, and the CLI training/evaluation scripts.
"""

import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns that are identifiers or row indices — never used as features.
ID_COLUMNS = ["Unnamed: 0", "id"]

# The target variable.
TARGET_COLUMN = "satisfaction"

# Categorical features (object / text columns).
CATEGORICAL_COLUMNS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

# Ordinal service-rating columns (scale 0–5, except Baggage Handling which is 1–5).
SERVICE_RATING_COLUMNS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]

# Other purely numerical features.
NUMERICAL_COLUMNS = ["Age", "Flight Distance"]

# Delay columns — treated as numerical.
DELAY_COLUMNS = [
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

# All feature columns after cleaning (in a stable order).
FEATURE_COLUMNS = (
    NUMERICAL_COLUMNS
    + DELAY_COLUMNS
    + SERVICE_RATING_COLUMNS
    + CATEGORICAL_COLUMNS
)

# Target mapping.
TARGET_MAPPING = {
    "neutral or dissatisfied": 0,
    "satisfied": 1,
}

TARGET_INVERSE_MAPPING = {v: k for k, v in TARGET_MAPPING.items()}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(train_path=None, test_path=None):
    """
    Load the Airline Passenger Satisfaction train and test CSV files.

    Parameters
    ----------
    train_path : str, optional
        Path to train.csv. Defaults to ``<project_root>/dataset/train.csv``.
    test_path : str, optional
        Path to test.csv. Defaults to ``<project_root>/dataset/test.csv``.

    Returns
    -------
    tuple of pd.DataFrame
        (train_df, test_df)
    """
    # Defer to the data ingestion component (logging + error handling).
    from components.data_ingestion import load_raw_data as _ingest

    return _ingest(train_path=train_path, test_path=test_path)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(df):
    """
    Clean a single DataFrame (train or test) by:
      1. Dropping ID / index columns.
      2. Removing exact duplicate rows.
      3. Imputing missing values in ``Arrival Delay in Minutes`` with the
         median (robust to outliers; aligns with the business meaning that
         many passengers have zero delay).
      4. Ensuring consistent data types.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()

    # 1. Drop identifier columns.
    cols_to_drop = [c for c in ID_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # 2. Remove exact duplicates.
    df = df.drop_duplicates().reset_index(drop=True)

    # 3. Impute missing values.
    # Arrival Delay in Minutes: only column with missing values.
    # Median is preferred over mean because delay distributions are heavily
    # right-skewed (very long delays are possible).
    if "Arrival Delay in Minutes" in df.columns:
        median_delay = df["Arrival Delay in Minutes"].median()
        df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(
            median_delay
        )

    # Ensure integer-type service and delay columns.
    int_cols = SERVICE_RATING_COLUMNS + NUMERICAL_COLUMNS + DELAY_COLUMNS + ["Age"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    return df


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------

def encode_target(df, target_column=TARGET_COLUMN):
    """
    Map the ``satisfaction`` text labels to binary integers.

    ``neutral or dissatisfied`` -> 0
    ``satisfied``               -> 1

    Returns a copy of *df* with the target column encoded as ``int8``.
    """
    df = df.copy()
    df[target_column] = df[target_column].map(TARGET_MAPPING).astype("int8")
    return df


def decode_target(encoded_value):
    """
    Convert a binary target value back to its text label.
    Useful for display / API output.
    """
    if isinstance(encoded_value, (np.integer, int)):
        return TARGET_INVERSE_MAPPING[int(encoded_value)]
    return TARGET_INVERSE_MAPPING.get(int(encoded_value), "Unknown")


# ---------------------------------------------------------------------------
# Feature utilities
# ---------------------------------------------------------------------------

def get_feature_columns():
    """
    Return the ordered list of feature column names used by the ML pipeline.
    """
    return list(FEATURE_COLUMNS)


def get_service_rating_columns():
    """Return the list of service-rating column names."""
    return list(SERVICE_RATING_COLUMNS)


def get_numerical_columns():
    """Return the list of non-rating numerical column names."""
    return list(NUMERICAL_COLUMNS + DELAY_COLUMNS)


def get_categorical_columns():
    """Return the list of categorical column names."""
    return list(CATEGORICAL_COLUMNS)


# ---------------------------------------------------------------------------
# Preprocessing pipeline (for sklearn)
# ---------------------------------------------------------------------------

class AirlineSatisfactionPreprocessor:
    """
    Build a sklearn ``ColumnTransformer`` that handles:
      - Numerical features: median imputation + standardisation.
      - Categorical features: most-frequent imputation + one-hot encoding.

    Usage::

        preprocessor = AirlineSatisfactionPreprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor.get_column_transformer()),
            ("classifier", RandomForestClassifier(...)),
        ])
    """

    def __init__(self):
        self.numerical_cols = NUMERICAL_COLUMNS + DELAY_COLUMNS + SERVICE_RATING_COLUMNS
        self.categorical_cols = CATEGORICAL_COLUMNS
        self._fitted = False

    def get_column_transformer(self):
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.numerical_cols),
                ("cat", categorical_transformer, self.categorical_cols),
            ],
            remainder="drop",
        )
        return preprocessor

    def fit_transform(self, X, y=None):
        ct = self.get_column_transformer()
        return ct.fit_transform(X, y)

    def transform(self, X):
        ct = self.get_column_transformer()
        # In practice, fit is done during training; this method is provided
        # for completeness but the fitted transformer should be stored.
        raise NotImplementedError(
            "Use the ColumnTransformer from get_column_transformer() and "
            "fit it within a full Pipeline."
        )


def preprocess_data(train_df, test_df=None):
    """
    Clean and prepare the raw data for modelling.

    Parameters
    ----------
    train_df : pd.DataFrame
        Raw training data.
    test_df : pd.DataFrame, optional
        Raw test data. If provided, it will be cleaned as well.

    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test) — if *test_df* is provided.
        Otherwise (X_train, y_train).
    """
    train_clean = clean_data(train_df)
    train_clean = encode_target(train_clean)

    y_train = train_clean[TARGET_COLUMN]
    X_train = train_clean.drop(columns=[TARGET_COLUMN])

    if test_df is not None:
        test_clean = clean_data(test_df)
        test_clean = encode_target(test_clean)
        y_test = test_clean[TARGET_COLUMN]
        X_test = test_clean.drop(columns=[TARGET_COLUMN])
        return X_train, y_train, X_test, y_test

    return X_train, y_train
