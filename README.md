# Account Health & Risk Intelligence System

An AI-powered Business Intelligence and Early Warning System designed for monitoring enterprise account health, forecasting financial performance, and predicting customer churn.

---

## Key Features

Dual Modeling Framework
Revenue Forecasting (Regression): Predicts next-month expected revenue using Random Forest Regressor.
Churn Risk Prediction (Classification): Identifies high-risk accounts using LightGBM Classifier.
Temporal Validation Strategy: Train-test split based on time horizons to avoid data leakage and simulate real-world usage.
Interactive Streamlit Interface
Executive Dashboards: High-level KPIs and time-series trends.
Real-time Risk Calculator: Instant predictions based on dynamic account inputs.
Automated Risk Diagnostics: Actionable operational insights for at-risk accounts.

---

## Tech Stack & Architecture

Frontend / Framework: Streamlit, Plotly
Machine Learning & Data Processing: Scikit-Learn, LightGBM, XGBoost, Pandas, NumPy, PyArrow
Persistence & Models: Joblib, Parquet

---

## Project Structure

account-intelligence/
├── data/                  # Processed Parquet datasets
├── models/                # Trained model pipelines (.pkl)
├── notebooks/             # EDA, Cleaning, and ML Modeling notebooks
├── src/                   # Core Python packages & modular functions
├── pages/                 # Streamlit multi-page interface apps
├── app.py                 # Main Streamlit app entry point
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

# Account Health Monitor Platform

An AI-powered Business Intelligence, Early Warning System, and REST API designed for monitoring SaaS account health, forecasting financial performance (MRR), and predicting customer churn risk.
Key Features

## Dual Modeling Framework:
    Revenue Forecasting (Regression): Predicts next-month expected revenue (MRR) using a Random Forest Regressor pipeline.
    Churn Risk Prediction (Classification): Identifies high-risk accounts and outputs churn probability percentages using a LightGBM Classifier.

## Decoupled REST API Backend (FastAPI):   
    Real-time Single Prediction (/predict): Evaluates individual account state and returns instant risk scoring and operational diagnostics.
    High-Performance Batch Inference (/predict-portfolio): Scores the entire account portfolio in a single vectorized pass.

## Interactive Analytics UI (Streamlit & Plotly):
    Executive Portfolio Dashboard: High-level KPIs, revenue distributions, and user engagement metrics.
    Revenue Intelligence: Projected revenue bridge waterfall charts and top revenue-loss driver identification.
    Customer Risk Analytics: Risk severity breakdown (High, Medium, Low) and churn probability spectrums.
    Account 360 & Real-Time Diagnostics: Deep-dive analysis with automated operational warnings.
    What-If Scenario Simulator: Dynamic parameter tweaking to simulate retention interventions via live API calls.
    CS Action Center: Priority queue management with automated risk categorization and CSV export capabilities.


## Tech Stack & Architecture

    Backend / API Engine: FastAPI, Uvicorn, Pydantic
    Frontend / Visualization: Streamlit, Plotly, Requests
    Machine Learning: Scikit-Learn, LightGBM, Pandas, NumPy
    Model Serialization & Persistence: Joblib, Parquet

## Project Structure
├── data/                  # Partitioned Parquet datasets
│   ├── data_part_0.parquet
│   └── data_part_1.parquet
├── models/                # Pre-trained pipeline binaries (.pkl)
│   ├── preprocessor.pkl
│   ├── final_mrr_model.pkl
│   └── final_churn_model.pkl
├── src/                   # Core Python modular packages
│   ├── __init__.py
│   ├── config.py          # Central paths, feature mappings, and UI theme constants
│   ├── data_loader.py     # Data loading with vectorized NumPy cleaning
│   ├── insights.py        # Rule-based operational risk diagnostic rules
│   ├── metrics.py         # Portfolio KPI & financial aggregation functions
│   └── predictor.py       # Inference wrapper for ML asset execution
├── main.py                # FastAPI REST API Backend application
├── app.py                 # Streamlit Frontend application
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation