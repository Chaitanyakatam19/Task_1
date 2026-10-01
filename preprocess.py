"""
AI & ML Internship - Task 1
Data Cleaning & Preprocessing

This script:
- explores the Titanic dataset
- handles missing values
- encodes categorical variables
- standardizes numerical variables
- detects/removes IQR outliers
- saves the processed dataset and boxplots
"""

from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "titanic.csv"
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

def load_data():
    if not DATA_PATH.exists():
        print("Local dataset not found. Downloading Titanic dataset...")
        urllib.request.urlretrieve(URL, DATA_PATH)
    return pd.read_csv(DATA_PATH)

def main():
    df = load_data()

    print("\n=== BASIC INFORMATION ===")
    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())

    # Drop columns with excessive missing values / redundant identifier-like information.
    df = df.drop(columns=["deck"], errors="ignore")

    # Handle numerical missing values with median.
    for col in ["age", "fare"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Handle categorical missing values with mode.
    for col in ["embarked", "embark_town"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print("\n=== AFTER MISSING-VALUE HANDLING ===")
    print(df.isnull().sum())

    # Separate target and features. 'alive' duplicates 'survived', so remove it.
    target = df["survived"].copy()
    features = df.drop(columns=["survived", "alive"], errors="ignore")

    categorical_cols = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_cols = features.select_dtypes(include=np.number).columns.tolist()

    # One-hot encoding for categorical features.
    encoded = pd.get_dummies(features, columns=categorical_cols, drop_first=True, dtype=int)

    # Detect outliers on original numerical features using IQR.
    outlier_mask = pd.Series(False, index=encoded.index)
    outlier_counts = {}

    for col in numerical_cols:
        q1 = encoded[col].quantile(0.25)
        q3 = encoded[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Boxplot before removing outliers.
        plt.figure(figsize=(7, 4))
        sns.boxplot(x=encoded[col])
        plt.title(f"Boxplot - {col} (before outlier removal)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{col}_before.png", dpi=150)
        plt.close()

        col_outliers = (encoded[col] < lower) | (encoded[col] > upper)
        outlier_counts[col] = int(col_outliers.sum())
        outlier_mask |= col_outliers

    print("\n=== OUTLIER COUNTS (IQR) ===")
    print(outlier_counts)
    print("Rows marked as outliers:", int(outlier_mask.sum()))

    # Remove rows with IQR outliers.
    cleaned = encoded.loc[~outlier_mask].copy()
    cleaned_target = target.loc[cleaned.index]

    # Standardize numerical features.
    scale_cols = [c for c in numerical_cols if c in cleaned.columns]
    scaler = StandardScaler()
    cleaned[scale_cols] = scaler.fit_transform(cleaned[scale_cols])

    # Add target back for the final ML-ready file.
    cleaned["survived"] = cleaned_target.astype(int)

    output_path = OUTPUT_DIR / "titanic_preprocessed.csv"
    cleaned.to_csv(output_path, index=False)

    print("\n=== FINAL RESULT ===")
    print("Original shape:", df.shape)
    print("Final shape:", cleaned.shape)
    print("Final missing values:", int(cleaned.isnull().sum().sum()))
    print("Saved:", output_path)

if __name__ == "__main__":
    main()
