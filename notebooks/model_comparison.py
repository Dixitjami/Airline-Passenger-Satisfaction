# model_comparison.py - Model Comparison for Airline Passenger Satisfaction
# Converted from model_comparison.ipynb - standalone script

import sys
from pathlib import Path
import os

# MUST be first: add src to path before any other imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 120

# Resolve paths relative to project root
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 120

# Load metrics from reports folder
metric_files = [
    "baseline_metrics.json",
    "decision_tree_metrics.json",
    "random_forest_metrics.json",
    "gradient_boosting_metrics.json"
]

models_data = []
for fname in metric_files:
    fpath = REPORTS_DIR / fname
    if fpath.exists():
        with open(fpath) as f:
            data = json.load(f)
        models_data.append(data)
    else:
        print(f"Warning: {fpath} not found")

print(f"Loaded {len(models_data)} model metrics")

# Create comparison DataFrame
comparison_rows = []
for m in models_data:
    metrics = m["metrics"]
    comparison_rows.append({
        "Model": m["model"],
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1-Score": metrics["f1"],
        "ROC-AUC": metrics["roc_auc"]
    })

df = pd.DataFrame(comparison_rows)
print("=== Validation Metrics Comparison ===")
print(df.to_string(index=False))

# Display accuracy as percentage
print("\nAccuracy as Percentage:")
for _, row in df.iterrows():
    print(f"{row['Model']}: {row['Accuracy'] * 100:.2f}%")

# Bar chart comparing accuracy
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]
bars = ax.bar(df["Model"], df["Accuracy"], color=colors, edgecolor="black", width=0.6)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_title("Model Accuracy Comparison (Validation Set)", fontsize=16, fontweight="bold")
ax.set_ylabel("Accuracy", fontsize=14)
ax.set_xlabel("Model", fontsize=14)
ax.set_ylim(0.8, 1.0)
ax.tick_params(axis="x", rotation=15, labelsize=12)
ax.tick_params(axis="y", labelsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'model_accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Additional comparison chart - all metrics
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x = range(len(df["Model"]))
width = 0.15

fig, ax = plt.subplots(figsize=(14, 7))
for i, metric in enumerate(metrics_to_plot):
    ax.bar([xi + i * width for xi in x], df[metric], width, label=metric)

ax.set_xlabel("Model", fontsize=14)
ax.set_ylabel("Score", fontsize=14)
ax.set_title("All Metrics Comparison Across Models", fontsize=16, fontweight="bold")
ax.set_xticks([xi + 2 * width for xi in x])
ax.set_xticklabels(df["Model"], rotation=15, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0.8, 1.0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'model_all_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Heatmap of all metrics
heatmap_data = df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].T
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="RdYlGn", center=0.9, ax=ax, cbar_kws={"label": "Score"})
ax.set_title("Model Metrics Heatmap", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'model_metrics_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ROC Curves - Individual
print("\nGenerating ROC curves...")
import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from src.preprocessing import load_data, clean_data, encode_target

# Load validation data (same split as training)
train_raw, _ = load_data()
train_median = train_raw['Arrival Delay in Minutes'].median() if 'Arrival Delay in Minutes' in train_raw.columns else None
df_clean = clean_data(train_raw, arrival_median=train_median)
df_clean = encode_target(df_clean)
X = df_clean.drop(columns=['satisfaction'])
y = df_clean['satisfaction']

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model_files = {
    'Logistic Regression': PROJECT_ROOT / 'models' / 'baseline_logistic_regression.joblib',
    'Decision Tree': PROJECT_ROOT / 'models' / 'decision_tree.joblib',
    'Random Forest': PROJECT_ROOT / 'models' / 'random_forest.joblib',
    'Gradient Boosting': PROJECT_ROOT / 'models' / 'gradient_boosting.joblib'
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()
colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b3']

for i, (name, path) in enumerate(model_files.items()):
    model = joblib.load(path)
    y_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)
    
    ax = axes[i]
    ax.plot(fpr, tpr, color=colors[i], lw=2.5, label=f'{name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5000)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{name} ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('ROC Curves - Individual Models', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'roc_curves_individual.png', dpi=150, bbox_inches='tight')
plt.close()

print("All plots saved to", FIGURES_DIR)
print("\n=== Model Comparison Complete ===")