"""
Model training module for the Airline Passenger Satisfaction project.

Trains and compares multiple classifiers, then saves the best model
to ``models/best_model.joblib``.
"""

import os
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier  # type: ignore[import-not-found]
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed — skipping XGBClassifier.")

try:  # imported as part of the ``src`` package
    from .preprocessing import (
        load_data,
        preprocess_data,
        AirlineSatisfactionPreprocessor,
    )
except ImportError:  # script-style run (python src/train.py)
    from preprocessing import (
        load_data,
        preprocess_data,
        AirlineSatisfactionPreprocessor,
    )


RANDOM_STATE = 42


def build_models():
    """
    Return a dictionary of model name -> sklearn Pipeline (each with the
    common preprocessing step attached).
    """
    models = {}

    # Model 1 — Logistic Regression (baseline)
    models["Logistic Regression"] = Pipeline(
        steps=[
            ("preprocessor", AirlineSatisfactionPreprocessor().get_column_transformer()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )),
        ]
    )

    # Model 2 — Decision Tree
    models["Decision Tree"] = Pipeline(
        steps=[
            ("preprocessor", AirlineSatisfactionPreprocessor().get_column_transformer()),
            ("classifier", DecisionTreeClassifier(
                max_depth=10,
                random_state=RANDOM_STATE,
            )),
        ]
    )

    # Model 3 — Random Forest
    models["Random Forest"] = Pipeline(
        steps=[
            ("preprocessor", AirlineSatisfactionPreprocessor().get_column_transformer()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )

    # Model 4 — Gradient Boosting
    models["Gradient Boosting"] = Pipeline(
        steps=[
            ("preprocessor", AirlineSatisfactionPreprocessor().get_column_transformer()),
            ("classifier", GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=3,
                random_state=RANDOM_STATE,
            )),
        ]
    )

    # Model 5 — XGBoost
    if HAS_XGBOOST:
        models["XGBoost"] = Pipeline(
            steps=[
                ("preprocessor", AirlineSatisfactionPreprocessor().get_column_transformer()),
                ("classifier", XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    use_label_encoder=False,
                    n_jobs=-1,
                )),
            ]
        )

    return models


def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train a single model pipeline and return evaluation metrics.

    Parameters
    ----------
    model : sklearn Pipeline
    X_train, y_train : training split
    X_val, y_val : validation split

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, model
    """
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_proba),
    }

    return metrics


def train_all_models(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE):
    """
    Train all defined models on a stratified train/validation split.

    Returns
    -------
    dict : model_name -> dict with metrics and the fitted pipeline
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train,
    )

    models = build_models()
    results = {}

    for name, pipeline in models.items():
        print(f"Training {name} ...")
        metrics = train_model(pipeline, X_tr, y_tr, X_val, y_val)
        metrics["model"] = pipeline
        results[name] = metrics
        print(
            f"  Accuracy={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}"
        )

    return results


def select_best_model(results, metric="f1"):
    """
    Select the best model from the *results* dict based on *metric*.

    Parameters
    ----------
    results : dict
        Output of ``train_all_models``.
    metric : str
        Metric to compare ('f1', 'accuracy', 'roc_auc', etc.)

    Returns
    -------
    tuple : (best_model_name, best_pipeline, best_metrics_dict)
    """
    best_name = max(
        results,
        key=lambda k: results[k][metric],
    )
    best_pipeline = results[best_name]["model"]
    best_metrics = {k: v for k, v in results[best_name].items() if k != "model"}

    return best_name, best_pipeline, best_metrics


def save_model(pipeline, model_dir=None, filename="best_model.joblib"):
    """
    Save a fitted pipeline (including preprocessing) to disk.

    Parameters
    ----------
    pipeline : sklearn Pipeline
    model_dir : str, optional
        Directory for the model file. Defaults to ``<project_root>/models``.
    filename : str
        Output filename.
    """
    if model_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, "models")

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")
    return model_path


def main():
    """
    Full training pipeline:
      1. Load data
      2. Clean & preprocess
      3. Encode target
      4. Train all models
      5. Select best
      6. Save best model
      7. Print summary
    """
    print("Loading data ...")
    train_raw, test_raw = load_data()

    print("Preprocessing ...")
    X_train, y_train, X_test, y_test = preprocess_data(train_raw, test_raw)

    print(f"Train features shape: {X_train.shape}")
    print(f"Test features shape:  {X_test.shape}")
    print(f"Train target distribution:\n{y_train.value_counts()}")

    print("\nTraining all models ...")
    results = train_all_models(X_train, y_train)

    best_name, best_pipeline, best_metrics = select_best_model(results)

    print(f"\n{'='*60}")
    print(f"Best model: {best_name}")
    print(f"Metrics (validation):")
    for k, v in best_metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print(f"{'='*60}")

    # Evaluate on the held-out test set.
    print("\nEvaluating best model on test set ...")
    y_test_pred = best_pipeline.predict(X_test)
    y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1": f1_score(y_test, y_test_pred),
        "roc_auc": roc_auc_score(y_test, y_test_proba),
    }

    print("Test set metrics:")
    for k, v in test_metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    save_model(best_pipeline)

    return best_name, best_pipeline, results, best_metrics, test_metrics


if __name__ == "__main__":
    main()
