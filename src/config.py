from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "telecom_churn.csv"
MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
MODEL_RESULTS_PATH = MODEL_DIR / "model_results.pkl"

TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2

IDENTIFIER_COLUMNS = ["customerID"]
NON_FEATURE_COLUMNS = [TARGET_COLUMN] + IDENTIFIER_COLUMNS

SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

RISK_THRESHOLDS = {
    "low": 0.30,
    "medium": 0.60,
}

NUMERICAL_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "CustomerLifetimeValue",
    "ChargeToTenure",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureGroup",
    "MonthlyChargeCategory",
    "ServiceCount",
]
