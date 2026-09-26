"""
Final evaluation script for Airline Passenger Satisfaction project.

Evaluates the pre-trained Gradient Boosting model on the held-out test set
(dataset/test.csv). This script does NOT retrain or modify the model.
"""

import os
import sys
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

try:  # imported as part of the ``src`` package
    from .preprocessing import (
        load_data,
        clean_data,
        encode_target,
        TARGET_COLUMN,
        TARGET_INVERSE_MAPPING,
    )
    from .logger import LOG_FILE_PATH  # noqa: F401  (configures logging)
    from .exception import CustomException
except ImportError:  # script-style run (python src/evaluate_final.py)
    from preprocessing import (
        load_data,
        clean_data,
        encode_target,
        TARGET_COLUMN,
        TARGET_INVERSE_MAPPING,
    )
    from logger import LOG_FILE_PATH  # noqa: F401  (configures logging)
    from exception import CustomException

import logging

# Ensure reports/figures directories exist
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model selection rationale
MODEL_NAME = "Gradient Boosting"
MODEL_PATH = MODELS_DIR / "gradient_boosting.joblib"
METRICS_OUTPUT = REPORTS_DIR / "final_test_metrics.json"
CM_FIGURE_PATH = FIGURES_DIR / "confusion_matrix_gradient_boosting_test.png"
ROC_FIGURE_PATH = FIGURES_DIR / "roc_curve_gradient_boosting_test.png"


def load_and_prepare_test_data(train_median):
    """Load and preprocess test data using the train median for imputation."""
    logging.info("Loading test data from dataset/test.csv")
    train_raw, test_raw = load_data()
    
    # Use the same train median for test imputation (no leakage)
    test_clean = clean_data(test_raw, arrival_median=train_median)
    test_clean = encode_target(test_clean)
    
    y_test = test_clean[TARGET_COLUMN]
    X_test = test_clean.drop(columns=[TARGET_COLUMN])
    
    logging.info("Test data preprocessed: X=%s, y=%s", X_test.shape, y_test.shape)
    logging.info("Test target distribution: %s", y_test.value_counts().to_dict())
    
    return X_test, y_test, test_raw


def evaluate_model(model, X_test, y_test):
    """Evaluate the fitted model on test data."""
    logging.info("Evaluating model on test set")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, 
            target_names=[TARGET_INVERSE_MAPPING[0], TARGET_INVERSE_MAPPING[1]],
            output_dict=True,
            zero_division=0,
        ),
    }
    
    return metrics, y_pred, y_proba


def save_metrics(metrics, model_name, report_path):
    """Save evaluation metrics to JSON file."""
    report = {
        "model": model_name,
        "target_mapping": {0: "neutral or dissatisfied", 1: "satisfied"},
        "positive_class": 1,
        "positive_class_label": "satisfied",
        "evaluation_set": "test (held-out)",
        "random_state": RANDOM_STATE,
        "metrics": {k: (v if k in ("confusion_matrix", "classification_report") else round(v, 4)) for k, v in metrics.items()},
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Metrics saved to: %s", report_path)
    return report_path


def save_confusion_matrix_figure(metrics, figure_path):
    """Save confusion matrix as a heatmap figure."""
    cm = np.array(metrics["confusion_matrix"])
    labels = ["neutral or dissatisfied", "satisfied"]
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": "Count"}
    )
    plt.title("Confusion Matrix - Gradient Boosting (Test Set)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Confusion matrix figure saved to: %s", figure_path)


def save_roc_curve_figure(model, X_test, y_test, figure_path):
    """Save ROC curve figure."""
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#c44e52", lw=2.5, label=f"Gradient Boosting (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - Gradient Boosting (Test Set)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("ROC curve figure saved to: %s", figure_path)


def main():
    """Run final evaluation on held-out test set."""
    logging.info("Starting final evaluation on held-out test set")
    
    try:
        # 1. Load raw data and compute train median for imputation (no leakage)
        logging.info("Loading raw training data to compute imputation median")
        train_raw, test_raw = load_data()
        train_median = train_raw["Arrival Delay in Minutes"].median() if "Arrival Delay in Minutes" in train_raw.columns else None
        logging.info("Train Arrival Delay median (for imputation): %.1f", train_median)
        
        # 2. Load and preprocess test data using train median
        X_test, y_test, test_raw = load_and_prepare_test_data(train_median)
        
        # 3. Load pre-trained model
        logging.info("Loading pre-trained model from %s", MODEL_PATH)
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logging.info("Model loaded successfully: %s", type(model).__name__)
        
        # 4. Verify feature alignment
        expected_features = list(X_test.columns)
        logging.info("Test features (%d): %s", len(expected_features), expected_features[:5])
        
        # 5. Evaluate model
        metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)
        
        # 6. Log results
        logging.info("Final test evaluation results:")
        for k, v in metrics.items():
            if k not in ("confusion_matrix", "classification_report"):
                logging.info("  %s: %.4f", k, v)
        logging.info("  confusion_matrix: %s", metrics["confusion_matrix"])
        
        # 7. Save metrics
        save_metrics(metrics, MODEL_NAME, METRICS_OUTPUT)
        
        # 8. Save confusion matrix figure
        save_confusion_matrix_figure(metrics, CM_FIGURE_PATH)
        
        # 9. Save ROC curve figure
        save_roc_curve_figure(model, X_test, y_test, ROC_FIGURE_PATH)
        
        # 10. Print summary
        print("\n" + "=" * 70)
        print(f"FINAL TEST EVALUATION - {MODEL_NAME}")
        print("=" * 70)
        print(f"Target mapping: 0 = {TARGET_INVERSE_MAPPING[0]}, 1 = {TARGET_INVERSE_MAPPING[1]}")
        print(f"Positive class: 1 ({TARGET_INVERSE_MAPPING[1]})")
        print(f"Evaluation set: dataset/test.csv (held-out, {len(y_test)} samples)")
        print("-" * 70)
        for k, v in metrics.items():
            if k not in ("confusion_matrix", "classification_report"):
                print(f"  {k.capitalize():14s}: {v:.4f}")
        print(f"  Confusion matrix: {metrics['confusion_matrix']}")
        print("-" * 70)
        print("Comparison with validation results:")
        print("  Validation accuracy:  0.9601 (Gradient Boosting)")
        print("  Test accuracy:        {:.4f}".format(metrics["accuracy"]))
        print("-" * 70)
        print(f"Artifacts saved:")
        print(f"  Metrics:  {METRICS_OUTPUT}")
        print(f"  CM fig:   {CM_FIGURE_PATH}")
        print(f"  ROC fig:  {ROC_FIGURE_PATH}")
        print("=" * 70)
        
        logging.info("Final evaluation completed successfully")
        return metrics
        
    except Exception as e:
        logging.error("Final evaluation failed: %s", e)
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()