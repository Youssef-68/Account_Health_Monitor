import pandas as pd
import numpy as np

def calculate_health_score(row):
    """
    Calculate Health Score (0 - 100) for a single account (dict or Series).
    Used by app.py Simulator and api.py endpoints.
    """
    score = 100.0

    # 1. Error rate penalty (Max -25 pts)
    score -= min(row.get("error_rate", 0) * 500, 25)

    # 2. Payment delay penalty (-20 pts)
    if row.get("payment_delay_flag", 0) == 1:
        score -= 20

    # 3. Support tickets penalty (Max -15 pts)
    tickets = row.get("tickets_count", 0)
    if tickets > 5:
        score -= min((tickets - 5) * 3, 15)

    # 4. Usage growth adjustment (-15 to +10 pts)
    usage_growth = row.get("usage_growth", 0)
    score += max(min(usage_growth * 20, 10), -15)

    # 5. Feature adoption reward (+10 pts)
    adoption = row.get("feature_adoption_rate", 0)
    score += (adoption * 10)

    # Clamp score strictly between 0 and 100
    return round(max(0.0, min(100.0, score)), 1)


def categorize_health(score):
    """
    Classify a numeric score into a health tier category.
    """
    if score >= 80:
        return "Healthy"
    elif score >= 55:
        return "At Risk"
    else:
        return "Critical"


def enrich_with_health_metrics_fast(df):
    """
    Highly optimized vectorized health score calculation for 1.2M+ rows.
    Runs in <0.1 seconds using NumPy arrays.
    """
    df = df.copy()

    # Base score array
    score = np.full(len(df), 100.0)

    # 1. Error rate penalty (Max -25)
    error_rate = df["error_rate"].fillna(0).values
    score -= np.minimum(error_rate * 500, 25)

    # 2. Payment delay penalty (-20)
    delay_flag = df["payment_delay_flag"].fillna(0).values
    score -= (delay_flag == 1) * 20

    # 3. Ticket penalty (Max -15)
    tickets = df["tickets_count"].fillna(0).values
    ticket_penalty = np.where(tickets > 5, np.minimum((tickets - 5) * 3, 15), 0)
    score -= ticket_penalty

    # 4. Usage growth adjustment (-15 to +10)
    growth = df["usage_growth"].fillna(0).values
    growth_adj = np.clip(growth * 20, -15, 10)
    score += growth_adj

    # 5. Adoption reward (+10)
    adoption = df["feature_adoption_rate"].fillna(0).values
    score += (adoption * 10)

    # Clip scores strictly between 0 and 100
    df["health_score"] = np.round(np.clip(score, 0.0, 100.0), 1)

    # Fast Categorization using np.select
    conditions = [
        df["health_score"] >= 80,
        df["health_score"] >= 55
    ]
    choices = ["Healthy", "At Risk"]
    df["health_status"] = np.select(conditions, choices, default="Critical")

    return df


def get_health_summary(df):
    """
    Compute full portfolio health metrics across the dataset.
    """
    if "health_score" not in df.columns:
        df = enrich_with_health_metrics_fast(df)

    total_accounts = len(df)
    avg_health = df["health_score"].mean()
    
    status_counts = df["health_status"].value_counts()
    critical_count = int(status_counts.get("Critical", 0))
    at_risk_count = int(status_counts.get("At Risk", 0))
    healthy_count = int(status_counts.get("Healthy", 0))
    
    # Calculate MRR at Risk (Critical + At Risk)
    risk_mask = df["health_status"].isin(["At Risk", "Critical"])
    mrr_at_risk = df.loc[risk_mask, "current_mrr"].sum()

    return {
        "Total Accounts": total_accounts,
        "Average Health Score": round(avg_health, 1),
        "Healthy Accounts": healthy_count,
        "At-Risk Accounts": at_risk_count,
        "Critical Accounts": critical_count,
        "MRR at Risk ($)": round(mrr_at_risk, 2)
    }