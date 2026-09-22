import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

HEALTH_MODEL_PATH = "health_risk_model.joblib"

def train_health_models(df):
    """
    Train ML pipelines to predict Health Score and future MRR.
    """
    features = [
        "company_size", "industry", "contract_type", "active_users",
        "usage_growth", "feature_adoption_rate", "error_rate",
        "tickets_count", "payment_delay_flag", "current_mrr"
    ]

    categorical_cols = ["company_size", "industry", "contract_type"]
    numerical_cols = [c for c in features if c not in categorical_cols]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ])

    X = df[features]
    y_mrr = df["next_month_mrr"]

    pipeline_mrr = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline_mrr.fit(X, y_mrr)
    joblib.dump(pipeline_mrr, HEALTH_MODEL_PATH)
    return pipeline_mrr

def load_or_train(df):
    if os.path.exists(HEALTH_MODEL_PATH):
        try:
            return joblib.load(HEALTH_MODEL_PATH)
        except Exception:
            pass
    return train_health_models(df)