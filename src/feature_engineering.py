from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SERVICE_COLUMNS


def _get_matching_column(df: pd.DataFrame, alternatives: list[str]) -> str | None:
    """Return the matching column name using a case-insensitive comparison."""
    normalized = {str(col).strip().lower().replace(" ", ""): col for col in df.columns}
    for alt in alternatives:
        key = alt.strip().lower().replace(" ", "")
        if key in normalized:
            return normalized[key]
    return None


def engineer_customer_lifetime_value(df: pd.DataFrame) -> pd.DataFrame:
    """Create the CustomerLifetimeValue feature using tenure and monthly charges."""
    engineered = df.copy()
    tenure_col = _get_matching_column(engineered, ["tenure"])
    monthly_col = _get_matching_column(engineered, ["MonthlyCharges", "monthlycharges"])
    if tenure_col and monthly_col:
        engineered["CustomerLifetimeValue"] = (
            pd.to_numeric(engineered[tenure_col], errors="coerce")
            * pd.to_numeric(engineered[monthly_col], errors="coerce")
        )
    return engineered


def engineer_charge_to_tenure_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Create a usage intensity measure comparing charges against tenure."""
    engineered = df.copy()
    tenure_col = _get_matching_column(engineered, ["tenure"])
    monthly_col = _get_matching_column(engineered, ["MonthlyCharges", "monthlycharges"])
    if tenure_col and monthly_col:
        tenure = pd.to_numeric(engineered[tenure_col], errors="coerce").fillna(0)
        monthly = pd.to_numeric(engineered[monthly_col], errors="coerce").fillna(0)
        engineered["ChargeToTenure"] = monthly / (tenure + 1)
    return engineered


def engineer_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Convert tenure into business-friendly customer lifespan bins."""
    engineered = df.copy()
    tenure_col = _get_matching_column(engineered, ["tenure"])
    if tenure_col:
        tenure = pd.to_numeric(engineered[tenure_col], errors="coerce")
        engineered["TenureGroup"] = pd.cut(
            tenure,
            bins=[-1, 12, 36, float("inf")],
            labels=["New", "Medium", "Long-Term"],
            right=False,
        )
        engineered["TenureGroup"] = engineered["TenureGroup"].fillna("New")
    return engineered


def engineer_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many premium services a customer has enabled."""
    engineered = df.copy()
    available_columns = []
    for column in SERVICE_COLUMNS:
        match = _get_matching_column(engineered, [column])
        if match:
            available_columns.append(match)

    if available_columns:
        engineered["ServiceCount"] = engineered[available_columns].replace({
            "Yes": 1,
            "No": 0,
            "yes": 1,
            "no": 0,
            "True": 1,
            "False": 0,
            "Y": 1,
            "N": 0,
            "1": 1,
            "0": 0,
            pd.NA: 0,
            None: 0,
            np.nan: 0,
        }).fillna(0).astype(int).sum(axis=1)
    else:
        engineered["ServiceCount"] = 0
    return engineered


def engineer_monthly_charge_category(df: pd.DataFrame) -> pd.DataFrame:
    """Group customers by monthly charge levels to capture price sensitivity."""
    engineered = df.copy()
    monthly_col = _get_matching_column(engineered, ["MonthlyCharges", "monthlycharges"])
    if monthly_col:
        monthly = pd.to_numeric(engineered[monthly_col], errors="coerce")
        valid = monthly.dropna()

        if valid.empty:
            engineered["MonthlyChargeCategory"] = "Low"
            return engineered

        q1 = valid.quantile(0.33)
        q2 = valid.quantile(0.66)

        if pd.isna(q1) or pd.isna(q2) or np.isclose(q1, q2) or q1 == q2:
            median = valid.median()
            engineered["MonthlyChargeCategory"] = np.select(
                [monthly <= median, monthly > median],
                ["Low", "High"],
                default="Low",
            )
            return engineered

        unique_edges = np.unique(np.array([-1.0, q1, q2, float("inf")], dtype=float))
        engineered["MonthlyChargeCategory"] = pd.cut(
            monthly,
            bins=unique_edges,
            labels=["Low", "Medium", "High"],
            right=False,
        )
        engineered["MonthlyChargeCategory"] = engineered["MonthlyChargeCategory"].fillna("Low")
    return engineered


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all telecom-specific engineered features in a single pipeline."""
    engineered = df.copy()
    engineered = engineer_customer_lifetime_value(engineered)
    engineered = engineer_charge_to_tenure_ratio(engineered)
    engineered = engineer_tenure_group(engineered)
    engineered = engineer_service_count(engineered)
    engineered = engineer_monthly_charge_category(engineered)
    return engineered
