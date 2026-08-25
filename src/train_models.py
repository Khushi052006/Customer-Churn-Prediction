from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import (
    BEST_MODEL_PATH,
    DATA_PATH,
    MODEL_DIR,
    MODEL_RESULTS_PATH,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_generator import generate_telecom_dataset
from src.evaluate_models import build_comparison_table, compute_metrics, generate_roc_data_for_model, plot_roc_curves
from src.feature_engineering import add_engineered_features
from src.preprocessing import (
    convert_boolean_like_columns,
    convert_total_charges_to_numeric,
    encode_target,
    inspect_dataset,
    load_dataset,
    normalize_yes_no_columns,
    prepare_features_and_target,
    remove_unnecessary_columns,
    standardize_columns,
    summarize_target_distribution,
    validate_required_columns,
)


def build_preprocessor(X_train: pd.DataFrame):
    """Create the dataset preprocessing pipeline for numerical and categorical features."""
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [
        column for column in X_train.columns if column not in numeric_features
    ]

    transformers = []
    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("No feature columns were found after feature engineering.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_models() -> dict:
    """Create the model dictionary for the churn prediction benchmark."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_depth=6,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def prepare_dataset_for_training(data_path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load, clean, and engineer features for model training."""
    data_path = Path(data_path)
    if not data_path.exists():
        generate_telecom_dataset(data_path)

    df = load_dataset(data_path)
    df = standardize_columns(df)
    df = normalize_yes_no_columns(df)
    df = convert_total_charges_to_numeric(df)
    df = convert_boolean_like_columns(df)
    df = remove_unnecessary_columns(df)
    df = encode_target(df, TARGET_COLUMN)
    df = add_engineered_features(df)

    required = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "TechSupport",
        "OnlineSecurity",
        TARGET_COLUMN,
    ]
    validate_required_columns(df, required)

    dataset_summary = inspect_dataset(df)
    if dataset_summary["duplicate_rows"] > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    X, y = prepare_features_and_target(df, TARGET_COLUMN)
    return X, y


def train_and_evaluate_models() -> tuple[dict, pd.DataFrame, str]:
    """Train all required models, compare their performance, and return the best model name."""
    X, y = prepare_dataset_for_training(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model_results = []
    model_roc_data = []
    trained_models = {}

    for model_name, model in build_models().items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "y_pred": y_pred,
            "y_proba": y_proba,
            "fpr": None,
            "tpr": None,
        }

        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        metrics["fpr"] = fpr
        metrics["tpr"] = tpr

        model_results.append(metrics)
        trained_models[model_name] = pipeline
        model_roc_data.append({
            "Model": model_name,
            "ROC-AUC": metrics["ROC-AUC"],
            "fpr": fpr,
            "tpr": tpr,
        })

    comparison_df = build_comparison_table([
        {
            "Model": row["Model"],
            "Accuracy": row["Accuracy"],
            "Precision": row["Precision"],
            "Recall": row["Recall"],
            "F1": row["F1"],
            "ROC-AUC": row["ROC-AUC"],
            "fpr": row["fpr"],
            "tpr": row["tpr"],
        }
        for row in model_results
    ])

    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_model_path = BEST_MODEL_PATH
    preprocessor_path = PREPROCESSOR_PATH
    joblib.dump(best_model.named_steps["model"], best_model_path)
    joblib.dump(best_model.named_steps["preprocessor"], preprocessor_path)
    comparison_df.to_pickle(MODEL_RESULTS_PATH)

    print("========================================")
    print("MODEL TRAINING COMPLETED")
    print("========================================")
    for _, row in comparison_df.iterrows():
        print(f"{row['Model']}")
        print(f"ROC-AUC: {row['ROC-AUC']:.4f}")
    print("========================================")
    print("BEST MODEL")
    print("========================================")
    print(f"Model: {best_model_name}")
    print(f"ROC-AUC: {comparison_df.iloc[0]['ROC-AUC']:.4f}")
    print("Model saved successfully.")
    return trained_models, comparison_df, best_model_name


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_and_evaluate_models()
