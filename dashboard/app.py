from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import BEST_MODEL_PATH, MODEL_RESULTS_PATH, PREPROCESSOR_PATH
from src.prediction import predict_customer_churn

st.set_page_config(page_title="Telecom Churn Intelligence", layout="wide", initial_sidebar_state="collapsed")
px.defaults.template = "plotly_dark"

st.markdown(
    """
    <style>
    :root {
        --bg: #060b16;
        --bg-2: #0b1220;
        --panel: rgba(15, 23, 42, 0.92);
        --panel-2: rgba(17, 24, 39, 0.88);
        --panel-3: rgba(15, 118, 110, 0.08);
        --border: rgba(148, 163, 184, 0.18);
        --primary: #60a5fa;
        --primary-2: #8b5cf6;
        --primary-3: #38bdf8;
        --success: #34d399;
        --warning: #fbbf24;
        --danger: #f472b6;
        --text: #e5eefb;
        --muted: #9fb0c9;
        --shadow: rgba(2, 6, 23, 0.48);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #050b16 0%, #0a1220 100%);
        color: var(--text);
    }

    .stApp {
        background: linear-gradient(180deg, #050b16 0%, #0a1220 42%, #0b1322 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    [data-testid="stHeader"] {
        background: rgba(5, 11, 22, 0.7);
        backdrop-filter: blur(10px);
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.85rem 0.4rem 1.15rem 0.4rem;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid var(--border);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        font-size: 0.9rem;
        color: var(--text);
    }

    .brand-mark {
        width: 32px;
        height: 32px;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-2) 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 10px 26px rgba(96, 165, 250, 0.35);
    }

    .brand-mark span {
        font-size: 0.82rem;
        font-weight: 700;
        color: white;
    }

    .nav-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        justify-content: flex-end;
        flex-wrap: wrap;
    }

    .nav-row .stRadio {
        width: 100%;
    }

    .nav-row [data-baseweb="radio"] {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
    }

    .nav-row [data-baseweb="radio"] > div {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.55rem 0.9rem;
        box-shadow: inset 0 0 0 1px rgba(96, 165, 250, 0.05);
    }

    .nav-row [data-baseweb="radio"] label {
        color: var(--muted);
        font-weight: 600;
        font-size: 0.85rem;
    }

    .nav-row [data-baseweb="radio"] .st-br {
        display: none;
    }

    .nav-row [data-baseweb="radio"] > div[aria-checked="true"] {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.16), rgba(139, 92, 246, 0.14));
        border-color: rgba(96, 165, 250, 0.45);
        box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.18), 0 12px 24px rgba(96, 165, 250, 0.18);
    }

    .nav-row [data-baseweb="radio"] > div[aria-checked="true"] label {
        color: #dfeeff;
    }

    .main-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.94), rgba(12, 18, 30, 0.94));
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 24px 48px var(--shadow);
    }

    .section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 0.7rem 0 1rem 0;
    }

    .section-kicker {
        color: var(--primary-3);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    h1, h2, h3 {
        color: var(--text);
        margin: 0;
        letter-spacing: -0.04em;
    }

    [data-testid="stMetricValue"] {
        color: var(--text);
        font-size: 1.8rem;
        font-weight: 700;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 700;
    }

    [data-testid="stMetricContainer"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(11, 17, 28, 0.96));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 0.9rem 0.8rem;
        box-shadow: 0 18px 28px rgba(2, 6, 23, 0.25);
    }

    .stDataFrame {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid var(--border);
        background: rgba(11, 17, 28, 0.9);
    }

    .stForm {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(9, 14, 23, 0.9));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 26px 48px rgba(2, 6, 23, 0.32);
    }

    .stSelectbox, .stNumberInput {
        background: rgba(11, 17, 28, 0.9);
    }

    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, input {
        background: rgba(11, 17, 28, 0.9) !important;
        color: var(--text) !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 12px !important;
    }

    .stSlider > div[data-testid="stThumbValue"] {
        color: var(--text);
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-2) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        letter-spacing: 0.02em;
        padding: 0.8rem 1rem;
        box-shadow: 0 12px 26px rgba(96, 165, 250, 0.22);
    }

    .stButton > button:hover {
        filter: brightness(1.08);
    }

    .prediction-panel {
        background: linear-gradient(180deg, rgba(11, 17, 28, 0.96), rgba(17, 24, 39, 0.96));
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.1rem;
        box-shadow: 0 24px 52px rgba(2, 6, 23, 0.3);
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.46rem 0.8rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.03em;
    }

    .status-good {
        background: rgba(52, 211, 153, 0.12);
        border: 1px solid rgba(52, 211, 153, 0.25);
        color: #a7f3d0;
    }

    .status-risk {
        background: rgba(244, 114, 182, 0.12);
        border: 1px solid rgba(244, 114, 182, 0.25);
        color: #fbcfe8;
    }

    .gauge-wrap {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.3rem 0;
    }

    @media (max-width: 1100px) {
        .topbar {
            flex-direction: column;
            align-items: flex-start;
        }
        .nav-row {
            justify-content: flex-start;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_model_results():
    if not MODEL_RESULTS_PATH.exists():
        raise FileNotFoundError("Model results file not found. Please train the models first.")
    return pd.read_pickle(MODEL_RESULTS_PATH)


@st.cache_data
def load_dataset_summary():
    dataset_path = Path("data/telecom_churn.csv")
    if not dataset_path.exists():
        return None
    df = pd.read_csv(dataset_path)
    return df


@st.cache_data
def load_model_artifacts():
    if not PREPROCESSOR_PATH.exists() or not BEST_MODEL_PATH.exists():
        raise FileNotFoundError("Saved model artifacts not found. Please train the models first.")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    return preprocessor, model


try:
    model_results = load_model_results()
    dataset = load_dataset_summary()
    best_model_name = model_results.iloc[0]["Model"]
    best_auc = model_results.iloc[0]["ROC-AUC"]
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()


st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-mark"><span>AI</span></div>
            <span>TELECOM CHURN INTELLIGENCE</span>
        </div>
        <div class="nav-row">
            <div>
    """,
    unsafe_allow_html=True,
)
page = st.radio(
    "Navigation",
    [
        "Overview",
        "Churn Analysis",
        "Model Comparison",
        "ROC Curve",
        "Feature Importance",
        "Customer Prediction",
    ],
    index=0,
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("""</div></div></div>""", unsafe_allow_html=True)


if page == "Overview":
    st.markdown('<div class="section-header"><div><div class="section-kicker">Executive summary</div><h1>Retention performance</h1></div></div>', unsafe_allow_html=True)
    if dataset is not None:
        churned = dataset["Churn"].astype(str).str.lower().eq("yes").sum()
        total_customers = len(dataset)
        churn_rate = (churned / total_customers) * 100 if total_customers else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Customers", f"{total_customers:,}")
        c2.metric("Churned", f"{churned:,}")
        c3.metric("Churn Rate", f"{churn_rate:.1f}%")
        c4.metric("Best Model", best_model_name)
        c5.metric("ROC-AUC", f"{best_auc:.3f}")

        churn_distribution = dataset["Churn"].astype(str).value_counts()
        pie_fig = px.pie(
            values=churn_distribution.values,
            names=["Retained", "Churned"],
            title="Customer Retention Mix",
            hole=0.56,
            color_discrete_sequence=["#60a5fa", "#f472b6"],
        )
        pie_fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color="#0b1220", width=2)))
        st.plotly_chart(pie_fig, use_container_width=True)


if page == "Churn Analysis":
    st.markdown('<div class="section-header"><div><div class="section-kicker">Behavioral patterns</div><h1>Churn drivers</h1></div></div>', unsafe_allow_html=True)
    if dataset is not None:
        churn_df = dataset.copy()
        churn_df["Churn"] = churn_df["Churn"].astype(str).str.lower().eq("yes").astype(int)

        col1, col2 = st.columns(2)
        with col1:
            contract_fig = px.bar(
                churn_df.groupby("Contract")["Churn"].mean().reset_index().rename(columns={"Churn": "Churn Rate"}),
                x="Contract",
                y="Churn Rate",
                title="Churn by Contract",
                color="Contract",
                color_discrete_sequence=["#60a5fa", "#8b5cf6", "#f472b6"],
            )
            st.plotly_chart(contract_fig, use_container_width=True)

        with col2:
            internet_fig = px.bar(
                churn_df.groupby("InternetService")["Churn"].mean().reset_index().rename(columns={"Churn": "Churn Rate"}),
                x="InternetService",
                y="Churn Rate",
                title="Churn by Internet Service",
                color="InternetService",
                color_discrete_sequence=["#60a5fa", "#8b5cf6", "#f472b6"],
            )
            st.plotly_chart(internet_fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            tenure_fig = px.line(
                churn_df.groupby("tenure")["Churn"].mean().reset_index().rename(columns={"Churn": "Churn Rate"}),
                x="tenure",
                y="Churn Rate",
                title="Churn by Tenure",
                line_shape="spline",
            )
            st.plotly_chart(tenure_fig, use_container_width=True)

        with col4:
            payment_fig = px.bar(
                churn_df.groupby("PaymentMethod")["Churn"].mean().reset_index().rename(columns={"Churn": "Churn Rate"}),
                x="PaymentMethod",
                y="Churn Rate",
                title="Churn by Payment Method",
                color="PaymentMethod",
                color_discrete_sequence=["#60a5fa", "#8b5cf6", "#34d399", "#f472b6"],
            )
            st.plotly_chart(payment_fig, use_container_width=True)


if page == "Model Comparison":
    st.markdown('<div class="section-header"><div><div class="section-kicker">Model benchmark</div><h1>Algorithm comparison</h1></div></div>', unsafe_allow_html=True)
    st.dataframe(model_results[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].style.format({"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "ROC-AUC": "{:.3f}"}), use_container_width=True)

    compare_fig = px.bar(
        model_results.sort_values("ROC-AUC", ascending=False),
        x="Model",
        y="ROC-AUC",
        title="ROC-AUC by Model",
        color="Model",
        color_discrete_sequence=["#60a5fa", "#8b5cf6", "#38bdf8", "#f472b6"],
    )
    st.plotly_chart(compare_fig, use_container_width=True)


if page == "ROC Curve":
    st.markdown('<div class="section-header"><div><div class="section-kicker">Evaluation</div><h1>Receiver operating characteristic</h1></div></div>', unsafe_allow_html=True)
    if "fpr" in model_results.columns:
        roc_fig = go.Figure()
        for _, row in model_results.iterrows():
            roc_fig.add_trace(go.Scatter(x=row["fpr"], y=row["tpr"], mode="lines", line=dict(width=2.8), name=f"{row['Model']} (AUC = {row['ROC-AUC']:.3f})"))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#94a3b8", width=1.2), name="Baseline"))
        roc_fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e5eefb"},
            legend={"orientation": "h", "y": 1.15, "x": 0.0},
        )
        st.plotly_chart(roc_fig, use_container_width=True)


if page == "Feature Importance":
    st.markdown('<div class="section-header"><div><div class="section-kicker">Interpretability</div><h1>Feature importance</h1></div></div>', unsafe_allow_html=True)
    try:
        preprocessor, model = load_model_artifacts()
        if hasattr(model, "feature_importances_"):
            feature_names = preprocessor.get_feature_names_out()
            importances = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False).head(15)
            feat_fig = px.bar(
                importances,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Predictors of Churn",
                color="Importance",
                color_continuous_scale=["#1d4ed8", "#8b5cf6", "#f472b6"],
            )
            st.plotly_chart(feat_fig, use_container_width=True)
        else:
            st.info("This model does not expose feature importances.")
    except FileNotFoundError as exc:
        st.error(str(exc))


if page == "Customer Prediction":
    st.markdown('<div class="section-header"><div><div class="section-kicker">AI scoring</div><h1>Predict churn</h1></div></div>', unsafe_allow_html=True)
    with st.form("predict_churn"):
        form_cols = st.columns(3)
        with form_cols[0]:
            tenure = st.slider("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
            gender = st.selectbox("Gender", ["Female", "Male"], index=0)
            senior = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)
            partner = st.selectbox("Partner", ["Yes", "No"], index=1)
            dependents = st.selectbox("Dependents", ["Yes", "No"], index=1)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0)
        with form_cols[1]:
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"], index=0)
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=1)
            online_security = st.selectbox("Online Security", ["Yes", "No"], index=1)
            online_backup = st.selectbox("Online Backup", ["Yes", "No"], index=0)
            device_protection = st.selectbox("Device Protection", ["Yes", "No"], index=1)
            tech_support = st.selectbox("Tech Support", ["Yes", "No"], index=1)
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"], index=0)
        with form_cols[2]:
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"], index=1)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=0)
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
            total_charges = st.number_input("Total Charges", min_value=0.0, max_value=20000.0, value=600.0, step=5.0)

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            result = predict_customer_churn(payload)
            probability = float(result["Probability"])
            risk = result["Risk Level"]
            prediction = result["Prediction"]

            gauge = go.Indicator(
                mode="gauge+number",
                value=probability,
                title={"text": "Churn Probability", "font": {"size": 26, "color": "#e5eefb"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#cbd5e1"},
                    "bar": {"color": "#8b5cf6"},
                    "bgcolor": "rgba(15, 23, 42, 0.9)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(148, 163, 184, 0.35)",
                    "steps": [
                        {"range": [0, 35], "color": "rgba(52, 211, 153, 0.25)"},
                        {"range": [35, 70], "color": "rgba(96, 165, 250, 0.25)"},
                        {"range": [70, 100], "color": "rgba(244, 114, 182, 0.25)"},
                    ],
                    "threshold": {"line": {"color": "#f8fafc", "width": 3}, "thickness": 0.8, "value": 50},
                },
                domain={"x": [0, 1], "y": [0, 1]},
            )

            status_class = "status-good" if prediction == "No Churn" else "status-risk"
            status_text = "Low risk • No Churn" if prediction == "No Churn" else "High risk • Likely to churn"

            left, right = st.columns([1.2, 1.0])
            with left:
                st.markdown('<div class="prediction-panel">', unsafe_allow_html=True)
                st.markdown(f'<div class="{status_class} status-pill">{status_text}</div>', unsafe_allow_html=True)
                st.markdown('<div class="gauge-wrap">', unsafe_allow_html=True)
                st.plotly_chart(go.Figure(gauge), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with right:
                st.markdown('<div class="prediction-panel">', unsafe_allow_html=True)
                st.markdown('<div class="section-kicker">Assessment</div>', unsafe_allow_html=True)
                st.markdown(f"<h2 style='margin:0.5rem 0; color:#e5eefb;'>{prediction}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:2rem; font-weight:700; margin:0; color:#60a5fa;'>{probability:.1f}%</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:1rem; color:#9fb0c9; margin-top:0.5rem;'>Risk level: <strong>{risk}</strong></p>", unsafe_allow_html=True)
                st.caption("This estimate is generated from the trained ML model and should be used as a decision-support signal rather than a guarantee.")
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
