from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PATH, RANDOM_STATE


def generate_telecom_dataset(output_path: str | Path = DATA_PATH, n_customers: int = 5000) -> pd.DataFrame:
    """Generate a realistic synthetic telecom churn dataset for local project use."""
    rng = np.random.default_rng(RANDOM_STATE)

    data = {
        "customerID": [f"CUST-{idx:05d}" for idx in range(1, n_customers + 1)],
        "gender": rng.choice(["Female", "Male"], size=n_customers),
        "SeniorCitizen": rng.choice([0, 1], p=[0.85, 0.15], size=n_customers),
        "Partner": rng.choice(["Yes", "No"], p=[0.52, 0.48], size=n_customers),
        "Dependents": rng.choice(["Yes", "No"], p=[0.31, 0.69], size=n_customers),
        "tenure": rng.integers(1, 72, size=n_customers),
        "PhoneService": rng.choice(["Yes", "No"], p=[0.89, 0.11], size=n_customers),
        "MultipleLines": rng.choice(["No", "Yes"], p=[0.60, 0.40], size=n_customers),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], p=[0.38, 0.42, 0.20], size=n_customers),
        "OnlineSecurity": rng.choice(["Yes", "No"], p=[0.42, 0.58], size=n_customers),
        "OnlineBackup": rng.choice(["Yes", "No"], p=[0.46, 0.54], size=n_customers),
        "DeviceProtection": rng.choice(["Yes", "No"], p=[0.43, 0.57], size=n_customers),
        "TechSupport": rng.choice(["Yes", "No"], p=[0.40, 0.60], size=n_customers),
        "StreamingTV": rng.choice(["Yes", "No"], p=[0.52, 0.48], size=n_customers),
        "StreamingMovies": rng.choice(["Yes", "No"], p=[0.50, 0.50], size=n_customers),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.27, 0.18], size=n_customers),
        "PaperlessBilling": rng.choice(["Yes", "No"], p=[0.64, 0.36], size=n_customers),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            p=[0.38, 0.19, 0.21, 0.22],
            size=n_customers,
        ),
    }

    df = pd.DataFrame(data)

    base = 30 + 2.5 * df["SeniorCitizen"] + 7 * (df["Contract"] == "Month-to-month")
    base += 5 * (df["InternetService"] == "Fiber optic") + 4 * (df["PaperlessBilling"] == "Yes")
    base += 5 * (df["PaymentMethod"] == "Electronic check")
    base += 1.5 * (df["MultipleLines"] == "Yes")
    base += 2.0 * (df["OnlineSecurity"] == "No") + 2.0 * (df["TechSupport"] == "No")
    base += 3.0 * (df["tenure"] < 12)
    base -= 4.0 * (df["Contract"] == "Two year")
    base -= 3.0 * (df["tenure"] > 36)

    monthly_charges = np.clip(base + rng.normal(0, 8, size=n_customers), 18, 120)
    df["MonthlyCharges"] = monthly_charges.round(2)
    df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"] + rng.normal(0, 40, size=n_customers)).round(2)

    churn_logit = (
        -1.5
        + 1.5 * (df["Contract"] == "Month-to-month")
        + 1.2 * (df["InternetService"] == "Fiber optic")
        + 0.9 * (df["PaymentMethod"] == "Electronic check")
        + 0.7 * (df["TechSupport"] == "No")
        + 0.8 * (df["OnlineSecurity"] == "No")
        + 0.6 * (df["PaperlessBilling"] == "Yes")
        + 0.05 * (df["MonthlyCharges"]) / 10
        - 0.03 * df["tenure"]
        + 0.4 * (df["SeniorCitizen"] == 1)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn_prob = np.clip(churn_prob, 0.08, 0.82)
    df["Churn"] = rng.binomial(1, churn_prob).astype(int)

    df["Churn"] = df["Churn"].map({0: "No", 1: "Yes"})
    df["TotalCharges"] = df["TotalCharges"].where(df["TotalCharges"].notna(), 0)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


if __name__ == "__main__":
    generate_telecom_dataset()
