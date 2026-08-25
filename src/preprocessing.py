from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import DATA_PATH, TARGET_COLUMN


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Find a matching column name in a case-insensitive way."""
    normalized_lookup = {
        str(column).strip().lower().replace(" ", ""): column for column in df.columns
    }
    for candidate in candidates:
        key = str(candidate).strip().lower().replace(" ", "")
        if key in normalized_lookup:
            return normalized_lookup[key]
    return None


def load_dataset(data_path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load a telecom churn CSV file and validate its structure."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Please place the telecom churn CSV file there."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"The dataset at '{path}' is empty.")

    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns by trimming whitespace and leaving the original casing intact."""
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned


def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that should not be used as predictive features."""
    cleaned = df.copy()
    customer_id_col = _find_column(cleaned, ["customerID", "customerid", "CustomerID"])
    if customer_id_col is not None:
        cleaned = cleaned.drop(columns=[customer_id_col])
    return cleaned


def normalize_yes_no_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common Yes/No strings across the dataset."""
    cleaned = df.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            cleaned[column] = cleaned[column].astype(str).str.strip()
            cleaned[column] = cleaned[column].replace({
                "Yes": "Yes",
                "No": "No",
                "yes": "Yes",
                "no": "No",
                "True": "Yes",
                "False": "No",
                "Y": "Yes",
                "N": "No",
                "1": "Yes",
                "0": "No",
                "nan": pd.NA,
                "None": pd.NA,
            })
    return cleaned


def convert_total_charges_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-looking strings in TotalCharges to numeric values safely."""
    cleaned = df.copy()
    total_charges_col = _find_column(cleaned, ["TotalCharges", "totalcharges"])
    if total_charges_col is not None:
        cleaned[total_charges_col] = pd.to_numeric(
            cleaned[total_charges_col].astype(str).str.strip().replace({"": pd.NA}), errors="coerce"
        )
    return cleaned


def convert_boolean_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce values like SeniorCitizen to numeric or boolean-friendly forms."""
    cleaned = df.copy()
    senior_col = _find_column(cleaned, ["SeniorCitizen", "seniorcitizen"])
    if senior_col is not None:
        cleaned[senior_col] = pd.to_numeric(cleaned[senior_col], errors="coerce")
    return cleaned


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Return a dataset summary for EDA and validation."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def encode_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Map churn labels to 0/1 for classification."""
    target_name = _find_column(df, [target_column, target_column.lower()])
    if target_name is None:
        raise KeyError(f"Target column '{target_column}' is missing from the dataset.")

    encoded = df.copy()
    target_values = encoded[target_name].astype(str).str.strip().str.lower()
    encoded[target_name] = target_values.map({
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
    })

    if encoded[target_name].isna().any():
        invalid_values = sorted(encoded.loc[encoded[target_name].isna(), target_name].astype(str).unique())
        raise ValueError(
            f"The target column contains unexpected values: {invalid_values}. "
            "Expected Yes/No or 1/0 values."
        )

    encoded[target_name] = encoded[target_name].astype(int)
    return encoded


def prepare_features_and_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN):
    """Separate features and target while checking required columns."""
    target_name = _find_column(df, [target_column, target_column.lower()])
    if target_name is None:
        raise KeyError(f"Required target column '{target_column}' is missing.")

    X = df.drop(columns=[target_name])
    y = df[target_name].astype(int)
    return X, y


def summarize_target_distribution(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.Series:
    """Compute the distribution of the target variable."""
    target_name = _find_column(df, [target_column, target_column.lower()])
    if target_name is None:
        raise KeyError(f"Target column '{target_column}' is missing from the dataset.")

    distribution = df[target_name].value_counts(normalize=False).sort_index()
    return distribution


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise a clear error if any required columns are missing."""
    missing = []
    for column in required_columns:
        if _find_column(df, [column, column.lower()]) is None:
            missing.append(column)
    if missing:
        raise KeyError(
            "The dataset is missing required columns: " + ", ".join(missing)
        )
