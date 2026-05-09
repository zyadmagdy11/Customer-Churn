"""
src/data_prep.py — Load, clean, split, and build a preprocessor
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found at '{path}'")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows from {path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df.drop(columns=["customerID"], errors="ignore", inplace=True)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for use inside a Pipeline."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CAT_COLS),
    ])


def prepare_data(path: str, target: str = "Churn", test_size: float = 0.2):
    """
    Load → clean → split.
    Returns raw (unprocessed) splits — the Pipeline will handle preprocessing.

    Returns:
        X_train, X_test, y_train, y_test  (all raw DataFrames/Series)
    """
    df = load_data(path)
    df = clean_data(df)

    X = df.drop(columns=[target])
    y = df[target]

    print(f"  Churn rate: {y.mean():.1%}  |  Features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


