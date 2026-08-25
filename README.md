# Telecom Customer Churn Prediction and Model Comparison Dashboard

## Project Overview
This project builds a machine learning system to estimate whether a telecom customer is likely to churn. It includes dataset validation, preprocessing, custom feature engineering, model training, evaluation, saving of model artifacts, and a Streamlit dashboard for business analytics and customer scoring.

## Problem Statement
Telecom companies lose revenue when customers leave. By predicting churn early, a business can target the right customers with retention offers, improve customer satisfaction, and reduce revenue loss.

## Objectives
- Load and validate telecom customer data.
- Engineer meaningful telecom-specific features.
- Train and compare multiple models.
- Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
- Save the best model and preprocessing pipeline.
- Deploy a dashboard for analytics and customer prediction.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib
- Jupyter

## Dataset Description
The project expects a telecom churn CSV file at `data/telecom_churn.csv`. The dataset contains customer information such as demographics, service details, contract type, monthly charges, total charges, and churn status.

## Features
Sample features include:
- customerID
- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges
- Churn

## Feature Engineering
The project adds telecom-focused features to improve prediction quality:
- CustomerLifetimeValue = MonthlyCharges × tenure
- ChargeToTenure = MonthlyCharges / (tenure + 1)
- TenureGroup = New / Medium / Long-Term
- ServiceCount = count of additional services enabled
- MonthlyChargeCategory = Low / Medium / High

These features can reflect usage intensity, contract maturity, and price sensitivity, which are often strongly associated with churn risk.

## Machine Learning Algorithms
The project trains and compares the following models:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

## Evaluation Metrics
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## ROC-AUC Explanation
ROC-AUC measures how well the model distinguishes churners from non-churners across different probability thresholds. It is especially useful in churn prediction because it reflects ranking quality rather than a single cutoff decision.

## Project Structure
```text
Telecom-Churn-Prediction/
├── data/
│   └── telecom_churn.csv
├── models/
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── model_results.pkl
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── prediction.py
│   └── data_generator.py
├── dashboard/
│   └── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/
```

## Installation Instructions
1. Create the virtual environment:
```bash
python -m venv venv
```
2. Activate the environment on Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
3. Install dependencies:
```bash
python -m pip install -r requirements.txt
```
4. If you prefer to install directly:
```bash
python -m pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly streamlit joblib jupyter
```

## How to Train Models
Run:
```bash
python src/train_models.py
```
This will:
- load or generate the telecom dataset
- clean and engineer features
- split train/test data
- train the four models
- evaluate performance
- select the best model by ROC-AUC
- save model artifacts in `models/`

## How to Run the Dashboard
After training, launch the dashboard with:
```bash
streamlit run dashboard/app.py
```

## Expected Output
When the dashboard opens, users should see:
- KPI cards for the total number of customers, churn count, churn rate, best model, and best ROC-AUC
- churn analysis charts
- model comparison results
- ROC curve comparison plot
- feature importance plot
- an interactive customer churn prediction form

## Notes
- The project does not retrain the model every time the dashboard starts.
- The saved model and preprocessing pipeline are loaded from disk.
- The dataset should be placed in `data/telecom_churn.csv`.

## .gitignore Guidance
Generated model files and notebooks are often not committed in large repositories, but they may be committed in academic or assignment repositories if required. In most cases, keep the dataset and model artifacts out of source control unless the repository specifically requires them.
