from __future__ import annotations

from typing import Any, Dict

import joblib
import pandas as pd

from src.config import BEST_MODEL_PATH, PREPROCESSOR_PATH, RISK_THRESHOLDS
from src.feature_engineering import add_engineered_features
from src.preprocessing import convert_boolean_like_columns, convert_total_charges_to_numeric, normalize_yes_no_columns, standardize_columns


def _canonicalize_record(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Map user input keys to dataset-compatible names using a case-insensitive lookup."""
    normalized = {}
    for key, value in raw_data.items():
        canonical_key = str(key).strip()
        match_key = canonical_key.lower().replace(" ", "")
        for candidate in [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
            "customerID",
        ]:
            if match_key == candidate.lower().replace(" ", ""):
                canonical_key = candidate
                break
        normalized[canonical_key] = value
    return normalized


def get_risk_level(probability: float) -> str:
    """Map a churn probability to a business-friendly risk level."""
    if probability < RISK_THRESHOLDS["low"]:
        return "Low Risk"
    if probability < RISK_THRESHOLDS["medium"]:
        return "Medium Risk"
    return "High Risk"


def prepare_customer_input(raw_data: Dict[str, Any]) -> pd.DataFrame:
    """Convert user input into a consistent, feature-engineered DataFrame."""
    record = _canonicalize_record(raw_data)
    if "customerID" not in record:
        record["customerID"] = "PREDICT-USER"

    df = pd.DataFrame([record])
    df = standardize_columns(df)
    df = normalize_yes_no_columns(df)
    df = convert_total_charges_to_numeric(df)
    df = convert_boolean_like_columns(df)
    df = add_engineered_features(df)
    return df


def load_trained_artifacts():
    """Load the saved preprocessing transformer and best model."""
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    return preprocessor, model


def predict_customer_churn(raw_data: Dict[str, Any]):
    """Predict churn probability and label using the saved model artifacts."""
    preprocessor, model = load_trained_artifacts()
    df = prepare_customer_input(raw_data)
    expected_columns = list(preprocessor.feature_names_in_)

    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise KeyError(
            "The input record is missing required fields for prediction: " + ", ".join(missing)
        )

    X = df[expected_columns]
    transformed = preprocessor.transform(X)
    probability = model.predict_proba(transformed)[0, 1]
    label = int(model.predict(transformed)[0])
    risk_level = get_risk_level(probability)

    return {
        "Prediction": "Churn" if label == 1 else "No Churn",
        "Probability": round(float(probability) * 100, 2),
        "Risk Level": risk_level,
        "Probability Decimal": float(probability),
    }
