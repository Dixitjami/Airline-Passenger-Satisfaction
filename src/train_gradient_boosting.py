"""
Gradient Boosting model training for Airline Passenger Satisfaction.

Trains Gradient Boosting classifier as the fourth model using the existing
preprocessing pipeline. Evaluates on the same stratified validation split
as previous models for fair comparison. Saves the fitted pipeline
and evaluation results.

Note: GradientBoostingClassifier does not support `class_weight` parameter.
Instead, class imbalance is handled by the algorithm's inherent design
(sequential residual fitting gives more weight to misclassified samples).
"""

import os
import sys
import json
import joblib
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

try:  # imported as part of the ``src`` package
    from .preprocessing import (
        load_data,
        AirlineSatisfactionPreprocessor,
        TARGET_COLUMN,
        TARGET_INVERSE_MAPPING,
    )
    from .logger import LOG_FILE_PATH  # noqa: F401  (configures logging)
    from .exception import CustomException
except ImportError:  # script-style run (python src/train_gradient_boosting.py)
    from preprocessing import (
        load_data,
        AirlineSatisfactionPreprocessor,
        TARGET_COLUMN,
        TARGET_INVERSE_MAPPING,
    )
    from logger import LOG_FILE_PATH  # noqa: F401  (configures logging)
    from exception import CustomException

import logging

# Ensure models and reports directories exist
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_gradient_boosting_pipeline():
    """Build a sklearn Pipeline with preprocessing and Gradient Boosting Classifier."""
    preprocessor = AirlineSatisfactionPreprocessor().get_column_transformer()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", AirlineSatisfactionPreprocessor().get_column_transformer()),
            ("classifier", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features="sqrt",
                subsample=0.8,
                random_state=RANDOM_STATE,
            )),
        ]
    )
    return pipeline


def train_gradient_boosting_model(X_train, y_train, X_val, y_val):
    """
    Train the Gradient Boosting model and return metrics.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, confusion_matrix, model
    """
    model = build_gradient_boosting_pipeline()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    cm = confusion_matrix(y_val, y_pred)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]),
        "confusion_matrix": cm.tolist(),
    }

    return metrics, model


def save_metrics(metrics, report_dir=None, filename="gradient_boosting_metrics.json"):
    """Save evaluation metrics to JSON file."""
    if report_dir is None:
        report_dir = REPORTS_DIR
    report_dir = Path(report_dir)
    report_dir.mkdir(exist_ok=True)

    report_path = report_dir / filename
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Metrics saved to: %s", report_path)
    return report_path


def save_model(pipeline, model_dir=None, filename="gradient_boosting.joblib"):
    """Save fitted pipeline to disk."""
    if model_dir is None:
        model_dir = MODELS_DIR
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / filename
    joblib.dump(pipeline, model_path)
    logging.info("Model saved to: %s", model_path)
    return model_path


def main():
    """Run Gradient Boosting training pipeline."""
    logging.info("Starting Gradient Boosting training")

    try:
        # 1. Load raw data
        logging.info("Loading raw data")
        train_raw, test_raw = load_data()
        logging.info("Raw train shape: %s, test shape: %s", train_raw.shape, test_raw.shape)

        # 2. Preprocess (cleans, encodes target, returns X, y)
        # We only use train_raw here; test_raw is kept completely separate
        logging.info("Preprocessing training data")
        train_median = train_raw["Arrival Delay in Minutes"].median() if "Arrival Delay in Minutes" in train_raw.columns else None
        from preprocessing import clean_data, encode_target
        train_clean = clean_data(train_raw, arrival_median=train_median)
        train_clean = encode_target(train_clean)

        y = train_clean[TARGET_COLUMN]
        X = train_clean.drop(columns=[TARGET_COLUMN])

        logging.info("Training data preprocessed: X=%s, y=%s", X.shape, y.shape)
        logging.info("Target distribution: %s", y.value_counts().to_dict())

        # 3. Create stratified train/validation split (same split as previous models for fair comparison)
        logging.info("Creating stratified train/validation split (test_size=%.2f)", TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        logging.info("Train split: %s, Validation split: %s", X_train.shape, X_val.shape)
        logging.info("Train target distribution: %s", y_train.value_counts().to_dict())
        logging.info("Val target distribution: %s", y_val.value_counts().to_dict())

        # 4. Train Gradient Boosting model
        logging.info("Training Gradient Boosting model")
        metrics, model = train_gradient_boosting_model(X_train, y_train, X_val, y_val)

        # 5. Log results
        logging.info("Gradient Boosting evaluation (validation):")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                logging.info("  %s: %.4f", k, v)
        logging.info("  confusion_matrix: %s", metrics["confusion_matrix"])

        # 6. Save metrics
        metrics_report = {
            "model": "Gradient Boosting",
            "target_mapping": {0: "neutral or dissatisfied", 1: "satisfied"},
            "positive_class": 1,
            "positive_class_label": "satisfied",
            "random_state": RANDOM_STATE,
            "validation_split": TEST_SIZE,
            "stratified": True,
            "hyperparameters": {
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 5,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
                "subsample": 0.8,
                "random_state": RANDOM_STATE,
                "note": "class_weight not supported by GradientBoostingClassifier; imbalance handled via subsample and sequential boosting",
            },
            "metrics": {k: (v if k == "confusion_matrix" else round(v, 4)) for k, v in metrics.items()},
        }
        save_metrics(metrics_report, filename="gradient_boosting_metrics.json")

        # 7. Save model
        save_model(model, filename="gradient_boosting.joblib")

        # 8. Print summary
        print("\n" + "=" * 60)
        print("GRADIENT BOOSTING - VALIDATION RESULTS")
        print("=" * 60)
        print(f"Target mapping: 0 = {TARGET_INVERSE_MAPPING[0]}, 1 = {TARGET_INVERSE_MAPPING[1]}")
        print(f"Positive class: 1 ({TARGET_INVERSE_MAPPING[1]})")
        print(f"Validation split: {TEST_SIZE*100:.0f}% (stratified, random_state={RANDOM_STATE})")
        print("-" * 60)
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"  {k.capitalize():14s}: {v:.4f}")
        print(f"  Confusion matrix: {metrics['confusion_matrix']}")
        print("=" * 60)
        print(f"Model saved to: models/gradient_boosting.joblib")
        print(f"Metrics saved to: reports/gradient_boosting_metrics.json")
        print("=" * 60)

        logging.info("Gradient Boosting training completed successfully")
        return metrics, model

    except Exception as e:
        logging.error("Gradient Boosting training failed: %s", e)
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()