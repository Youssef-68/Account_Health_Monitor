# Account Health Monitor App

An end-to-end Machine Learning system designed to analyze customer behavior, forecast Monthly Recurring Revenue (MRR), and evaluate churn risks for B2B SaaS companies using a dataset of over **1.15 million monthly records**.
---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Architecture & Features](#dataset-architecture--features)
- [Data Pipeline & Feature Engineering](#data-pipeline--feature-engineering)
- [Model Performance & Evaluation](#model-performance--evaluation)
  - [1. Revenue Prediction (MRR Regression)](#1-revenue-prediction-mrr-regression)
  - [2. Churn Risk Prediction (Classification)](#2-churn-risk-prediction-classification)
- [Project Directory Structure](#project-directory-structure)
- [Installation & Setup](#installation--setup)
- [How to Run & Usage](#how-to-run--usage)

---

## Project Overview

Retaining customers and accurately forecasting revenue are critical for subscription business models. This platform provides actionable business intelligence by:
- **Predicting Future MRR:** Estimating the exact dollar value of each account for the upcoming month.
- **Early Churn Identification:** Classifying accounts at risk of subscription cancellation or revenue decline before it occurs.
- **Production-Ready Pipeline:** Featuring modular Python code, API integration, interactive dashboards, and saved preprocessors.

---

## Dataset Architecture & Features

The dataset merges two Parquet files comprising **1,150,000 rows** across **16 key features**:

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `account_id` | Int64 | Unique identifier for each customer account |
| `month` | Int64 | Sequential tracking month (1 to 23) |
| `company_size` | String | Customer tier (e.g., SMB, Enterprise) |
| `industry` | String | Industry domain of the client company |
| `contract_type` | String | Subscription plan type (e.g., Monthly, Annual) |
| `discount_pct` | Float | Discount percentage applied to the subscription |
| `regime_state` | String | Customer health behavior state (e.g., Stable, Decline) |
| `active_users` | Integer | Total active users logging into the platform monthly |
| `usage_growth` | Float | Percentage change in platform usage month-over-month |
| `feature_adoption_rate` | Float | Ratio of available features actively utilized |
| `error_rate` | Float | Customer-experienced application error rate |
| `tickets_count` | Integer | Total support tickets opened during the month |
| `ticket_growth` | Float | Month-over-month growth rate of support tickets |
| `payment_delay_flag` | Integer | Binary flag indicating late payment status (1/0) |
| `current_mrr` | Float | **Current** Monthly Recurring Revenue ($) |
| `next_month_mrr` | Float | **Regression Target:** Next month's MRR ($) |

---

## Data Pipeline & Feature Engineering

1. **Target Variable Construction:**
   - **MRR Delta:** `mrr_change = next_month_mrr - current_mrr`
   - **Churn Target:** `churn = 1` if `next_month_mrr < current_mrr` else `0`.
2. **Temporal Split Strategy (Time-Based Train/Test Split):**
   - **Training Set:** Months $1 \rightarrow 18$ (~900,000 records).
   - **Testing Set:** Months $19 \rightarrow 23$ (~250,000 records).
3. **Data Preprocessing & Transformation:**
   - Categorical variables (`company_size`, `industry`, `contract_type`, `regime_state`) encoded via `OneHotEncoder`.
   - Numerical variables scaled using `StandardScaler`.

---

## Model Performance & Evaluation

### 1. Revenue Prediction (MRR Regression)

Models trained to predict `next_month_mrr`, ranked by Mean Absolute Error (MAE):

| Model | MAE | RMSE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Random Forest** | **59.87** | **845.35** | **0.9027** |
| Gradient Boosting | 62.08 | 857.22 | 0.8999 |
| Decision Tree | 69.68 | 905.46 | 0.8884 |
| XGBoost | 118.85 | 1516.80 | 0.6869 |
| LightGBM | 145.64 | 1788.06 | 0.5648 |
| AdaBoost | 195.27 | 952.30 | 0.8766 |
| Linear Regression | 198.42 | 1006.06 | 0.8622 |

* **Best Model:** **Random Forest Regressor** achieved the highest accuracy with $R^2 = 90.27%.

---

### 2. Churn Risk Prediction (Classification)

Models evaluated for classifying accounts at risk of churn, ranked by ROC-AUC:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.8137** | **0.7817** | **0.4305** | **0.5552** | **0.8633** |
| XGBoost | 0.8129 | 0.7852 | 0.4228 | 0.5496 | 0.8629 |
| Gradient Boosting | 0.8119 | 0.7873 | 0.4158 | 0.5442 | 0.8603 |
| Random Forest | 0.8040 | 0.8078 | 0.3597 | 0.4978 | 0.8550 |
| Decision Tree | 0.8071 | 0.7675 | 0.4099 | 0.5344 | 0.8397 |
| Logistic Regression | 0.7928 | 0.6725 | 0.4541 | 0.5421 | 0.8017 |
| AdaBoost | 0.7781 | 0.7394 | 0.2758 | 0.4018 | 0.7315 |

* **Best Model:** **LightGBM Classifier** performed best overall with ROC-AUC of **0.8633** and **81.37% Accuracy**.

---

## Project Directory Structure

account_health_monitor_app/
├── data/
│   ├── data_part_0.parquet         # Parquet dataset part 1
│   ├── data_part_1.parquet         # Parquet dataset part 2
│   └── saas_50k_v1.csv             # Sample dataset (50k rows)
├── models/
│   ├── final_churn_model.pkl       # Production LightGBM Churn Classifier
│   ├── final_mrr_model.pkl         # Production Random Forest MRR Regressor
│   └── preprocessor.pkl            # Preprocessing pipeline object
├── notebooks/
│   ├── 01_Cleaning.ipynb          # Data cleaning and validation
│   ├── 02_EDA.ipynb               # Exploratory Data Analysis
│   └── 03_ML_Modeling.ipynb       # Model training, evaluation & tuning
├── src/
│   ├── data_loader.py             # Data fetching & parsing utilities
│   ├── health_engine.py           # Core business logic & health scoring
│   ├── models.py                  # ML model wrappers & inference logic
│   └── visualizations.py          # Dashboard visual components
├── .gitignore                      # Git ignored files
├── api.py                          # FastAPI / REST API service
├── app.py                          # Web interface / Streamlit Application
├── health_risk_model.joblib        # Additional serialized risk model
├── main.py                         # Application entry point
├── README.md                       # Project documentation
└── requirements.txt                # Python dependency list
