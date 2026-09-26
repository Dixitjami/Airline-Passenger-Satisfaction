# 01_EDA.py - EDA Analysis for Airline Passenger Satisfaction
# Converted from 01_EDA.ipynb - standalone script

import sys
from pathlib import Path

# Setup paths first
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

mpl.rcParams['figure.facecolor'] = 'white'
plt.rcParams['figure.figsize'] = (9, 4.5)
plt.rcParams['figure.dpi'] = 130
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
sns.set_theme(style='whitegrid', palette='muted')
COLORS = {'satisfied': '#2ca02c', 'neutral or dissatisfied': '#d62728'}

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

print('env OK')

# Load RAW data
train_path = PROJECT_ROOT / 'dataset' / 'train.csv'
test_path = PROJECT_ROOT / 'dataset' / 'test.csv'
df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
print(f"train {df.shape}  test {df_test.shape}")
print(df.head().to_string())
print(df.tail().to_string())
assert df.shape == (103904, 25)
assert df_test.shape == (25976, 25)

# Basic dataset information
print("=== Column Information ===")
info_df = pd.DataFrame({
    'dtype': df.dtypes.astype(str),
    'unique': [df[c].nunique() for c in df.columns],
    'missing': df.isnull().sum().values,
    'missing_pct': (df.isnull().mean() * 100).round(2)
}, index=df.columns)
print(info_df.to_string())
print()
df.info()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
df.dtypes.value_counts().plot(kind='bar', ax=ax[0], color=['#4c78a8', '#f58518'])
ax[0].set_title('Columns by dtype')
ax[0].set_ylabel('count')
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax[1])
ax[1].set_title('Missing map (yellow=missing)')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_dtypes_missing.png', dpi=150, bbox_inches='tight')
plt.close()

# Missing value analysis
miss = df.isnull().sum()
miss_pct = df.isnull().mean() * 100
miss_df = pd.DataFrame({'missing': miss, 'pct': miss_pct.round(2)}).query('missing > 0')
print("Missing values:")
print(miss_df.to_string())

fig, ax = plt.subplots(figsize=(7, 3))
miss_pct[miss_pct > 0].plot(kind='bar', color='#e45756', ax=ax)
ax.set_title('Missing percentage (only Arrival Delay has missing)')
ax.set_ylabel('%')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_missing_pct.png', dpi=150, bbox_inches='tight')
plt.close()

print("Conclusion: only Arrival Delay has 310 missing (0.30%)")

# Duplicate analysis
dups = df.duplicated().sum()
print(f"\nDuplicates: {dups} ({dups/len(df)*100:.4f}%)")

# Target variable
print("\n=== Target Variable Analysis ===")
vc = df['satisfaction'].value_counts()
pct = df['satisfaction'].value_counts(normalize=True).mul(100).round(2)
target_df = pd.DataFrame({'count': vc, 'pct': pct})
print(target_df.to_string())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(x='satisfaction', data=df, palette=[COLORS[c] for c in vc.index], ax=ax[0])
ax[0].set_title('Target counts')
for p in ax[0].patches:
    ax[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height()),
                   ha='center', va='bottom', fontsize=9)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=10)
pct.plot(kind='pie', autopct='%1.1f%%', colors=[COLORS[c] for c in pct.index], ax=ax[1])
ax[1].set_title('Target share')
ax[1].set_ylabel('')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_target_dist.png', dpi=150, bbox_inches='tight')
plt.close()

print('Mild imbalance: 56.7% neutral/dissatisfied vs 43.3% satisfied')

# Numerical feature analysis
print("\n=== Numerical Feature Analysis ===")
num_cols = [c for c in df.select_dtypes(include='number').columns if c not in ['Unnamed: 0', 'id']]
print(df[num_cols].describe().T.to_string())

for c in ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    sns.histplot(df[c].dropna(), bins=40, kde=True, color='#4c78a8', ax=axes[0])
    axes[0].set_title(c + ' hist')
    sns.boxplot(x=df[c].dropna(), color='#72b7b2', ax=axes[1])
    axes[1].set_title(c + ' box')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'01_{c.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()

# Categorical feature analysis
print("\n=== Categorical Feature Analysis ===")
cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, c in zip(axes.flat, cat_cols):
    order = df[c].value_counts().index
    sns.countplot(x=c, data=df, order=order, palette='Set2', ax=ax)
    ax.set_title(c)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height()),
                    ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_categorical.png', dpi=150, bbox_inches='tight')
plt.close()

for c in cat_cols:
    print(f"\n{c}:")
    print(df[c].value_counts().to_string())
    print(df[c].value_counts(normalize=True).mul(100).round(1).to_string())

# Service rating analysis
print("\n=== Service Rating Analysis ===")
rating_cols = [
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
]
print(df[rating_cols].describe().T.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
df[rating_cols].mean().sort_values().plot(kind='barh', color='#4c78a8', ax=ax)
ax.set_title('Mean service ratings (higher = better)')
ax.set_xlabel('mean 0-5')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_service_ratings.png', dpi=150, bbox_inches='tight')
plt.close()

# Outlier analysis
print("\n=== Outlier Analysis (IQR) ===")
for c in ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    out = int(((df[c] < q1 - 1.5*iqr) | (df[c] > q3 + 1.5*iqr)).sum())
    print(f"{c}: Q1={q1:.1f} Q3={q3:.1f} IQR={iqr:.1f} outliers={out} ({out/len(df)*100:.2f}%)")

fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
for ax, c in zip(axes, ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']):
    sns.boxplot(x=df[c].dropna(), color='#ff9da7', ax=ax)
    ax.set_title(c)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_outliers.png', dpi=150, bbox_inches='tight')
plt.close()

print("Delays heavily skewed with long-tail outliers - retained")

# Correlation analysis
print("\n=== Correlation Analysis ===")
tmp = df.copy()
tmp['satisfaction_binary'] = tmp['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
corr = tmp.select_dtypes(include='number').corr(numeric_only=True)

import numpy as np
plt.figure(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, annot=False, cbar_kws={'shrink': 0.8})
plt.title('Correlation heatmap (lower triangle)')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_correlation.png', dpi=150, bbox_inches='tight')
plt.close()

target_corr = corr['satisfaction_binary'].drop('satisfaction_binary').sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#d62728' if v < 0 else '#2ca02c' for v in target_corr.values]
target_corr.plot(kind='barh', color=colors, ax=ax)
ax.set_title('Correlation with satisfaction (binary)')
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_corr_target.png', dpi=150, bbox_inches='tight')
plt.close()
print(target_corr.round(3).to_string())

# Feature vs Target
print("\n=== Feature vs Target Analysis ===")
for c in ['Type of Travel', 'Class', 'Customer Type', 'Gender']:
    ct = pd.crosstab(df[c], df['satisfaction'], normalize='index') * 100
    ct.plot(kind='bar', color=[COLORS['neutral or dissatisfied'], COLORS['satisfied']])
    plt.title(c + ' vs satisfaction (%)')
    plt.ylabel('%')
    plt.xticks(rotation=15)
    plt.legend(title='satisfaction')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'01_{c.lower().replace(" ", "_")}_vs_target.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(ct.round(1).to_string())

for c in ['Online boarding', 'Inflight wifi service', 'Seat comfort', 'Cleanliness',
          'Inflight entertainment', 'Food and drink', 'Flight Distance',
          'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Age']:
    plt.figure(figsize=(6, 3.5))
    sns.boxplot(x='satisfaction', y=c, data=df, palette=COLORS)
    plt.title(c + ' vs satisfaction')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'01_{c.lower().replace(" ", "_")}_box.png', dpi=150, bbox_inches='tight')
    plt.close()

print("\n=== EDA Complete ===")
print("All figures saved to", FIGURES_DIR)