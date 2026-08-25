from __future__ import annotations

from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from src.config import RANDOM_STATE


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> dict:
    """Compute binary classification metrics for a trained model."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }


def build_comparison_table(results: List[dict]) -> pd.DataFrame:
    """Convert a list of model result dictionaries into a comparison table."""
    comparison = pd.DataFrame(results)
    if comparison.empty:
        raise ValueError("No model results were provided for comparison.")
    return comparison.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


def plot_roc_curves(model_results: pd.DataFrame, save_path: str | None = None):
    """Plot a combined ROC curve for all models."""
    fig = go.Figure()
    for _, row in model_results.iterrows():
        fig.add_trace(
            go.Scatter(
                x=row["fpr"],
                y=row["tpr"],
                mode="lines",
                name=f"{row['Model']} (AUC = {row['ROC-AUC']:.3f})",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Random Baseline",
        )
    )

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend_title="Model",
        template="plotly_white",
        width=980,
        height=650,
    )
    if save_path is not None:
        fig.write_html(save_path)
    return fig


def plot_model_roc_auc(model_results: pd.DataFrame, save_path: str | None = None):
    """Plot a bar chart comparing ROC-AUC across models."""
    data = model_results.sort_values("ROC-AUC", ascending=False).copy()
    fig = go.Figure(
        data=[
            go.Bar(
                x=data["Model"],
                y=data["ROC-AUC"],
                marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][: len(data)],
            )
        ]
    )
    fig.update_layout(
        title="Model Comparison by ROC-AUC",
        xaxis_title="Model",
        yaxis_title="ROC-AUC",
        template="plotly_white",
        height=500,
        width=900,
    )
    if save_path is not None:
        fig.write_html(save_path)
    return fig


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Render a confusion matrix for the best model."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Actual No", "Actual Yes"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def get_feature_importance(model, feature_names: List[str], top_n: int = 10) -> pd.DataFrame:
    """Extract feature importance values from tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)
    return feature_importance_df.head(top_n).reset_index(drop=True)


def plot_feature_importance(feature_importance_df: pd.DataFrame, save_path: str | None = None):
    """Plot the top features by importance."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=feature_importance_df["Importance"],
                y=feature_importance_df["Feature"],
                orientation="h",
                marker_color="steelblue",
            )
        ]
    )
    fig.update_layout(
        title="Top Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
        height=700,
        width=900,
    )
    if save_path is not None:
        fig.write_html(save_path)
    return fig


def summarize_model_results(model_names: List[str], model_predictions: Dict[str, dict]) -> pd.DataFrame:
    """Create a summary table of model metrics."""
    records = []
    for model_name in model_names:
        model_metrics = model_predictions[model_name]
        records.append({
            "Model": model_name,
            "Accuracy": model_metrics["accuracy"],
            "Precision": model_metrics["precision"],
            "Recall": model_metrics["recall"],
            "F1": model_metrics["f1"],
            "ROC-AUC": model_metrics["roc_auc"],
        })
    return pd.DataFrame(records).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


def generate_roc_data_for_model(model, X_test, y_test):
    """Calculate ROC curve values for a single model."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": auc,
    }
