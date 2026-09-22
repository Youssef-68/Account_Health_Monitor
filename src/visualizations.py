import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_health_donut(df):
    """
    Lightweight Donut Chart showing Health Status breakdown across 100% of rows.
    """
    counts = df["health_status"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    
    fig = px.pie(
        counts,
        names="Status",
        values="Count",
        hole=0.5,
        color="Status",
        color_discrete_map={"Healthy": "#2ecc71", "At Risk": "#f1c40f", "Critical": "#e74c3c"},
        title="Overall Portfolio Health Split"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20), height=280)
    return fig


def plot_quadrant_risk_matrix(df, max_points=10000):
    """
    Interactive Risk Matrix with smart sampling for display (max 10k points).
    Medians and thresholds are computed on 100% of data.
    """
    avg_mrr = df["current_mrr"].median()
    
    # Sample points for display rendering speed
    if len(df) > max_points:
        plot_df = df.sample(n=max_points, random_state=42)
        title_suffix = f" (Sampled {max_points:,} of {len(df):,} accounts for UI speed)"
    else:
        plot_df = df
        title_suffix = ""

    hover_cols = [c for c in ["industry", "error_rate", "payment_delay_flag"] if c in plot_df.columns]

    fig = px.scatter(
        plot_df,
        x="health_score",
        y="current_mrr",
        color="health_status",
        size="tickets_count" if "tickets_count" in plot_df.columns else None,
        hover_name="account_id" if "account_id" in plot_df.columns else None,
        hover_data=hover_cols,
        color_discrete_map={"Healthy": "#2ecc71", "At Risk": "#f1c40f", "Critical": "#e74c3c"},
        labels={"health_score": "Health Score (0-100)", "current_mrr": "Current MRR ($)"},
        title="Account Risk Matrix (Health vs. Financial Value)" + title_suffix
    )
    
    # Threshold lines
    fig.add_vline(x=55, line_dash="dash", line_color="#e74c3c", annotation_text="Critical Limit (55)")
    fig.add_vline(x=80, line_dash="dash", line_color="#2ecc71", annotation_text="Healthy Limit (80)")
    fig.add_hline(y=avg_mrr, line_dash="dot", line_color="gray", annotation_text=f"Median MRR (${avg_mrr:,.0f})")

    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=380)
    return fig


def plot_industry_health_vs_risk(df):
    """
    Industry benchmark comparing average health score.
    """
    summary = df.groupby("industry").agg(
        avg_health=("health_score", "mean")
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary["industry"],
        y=summary["avg_health"],
        name="Avg Health Score",
        marker_color="#3498db"
    ))
    
    fig.update_layout(
        title="Industry Health Benchmark",
        xaxis_title="Industry",
        yaxis_title="Avg Health Score (0-100)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=280
    )
    return fig


def plot_risk_drivers(df):
    """
    Compare key metrics (Error Rate, Tickets, Payment Delay) between Healthy, At-Risk & Critical accounts.
    """
    drivers = df.groupby("health_status").agg(
        avg_error_rate=("error_rate", lambda x: x.mean() * 100),
        avg_tickets=("tickets_count", "mean"),
        delay_rate=("payment_delay_flag", lambda x: x.mean() * 100)
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Avg Error Rate (%)", 
        x=drivers["health_status"], 
        y=drivers["avg_error_rate"], 
        marker_color="#e74c3c"
    ))
    fig.add_trace(go.Bar(
        name="Avg Support Tickets", 
        x=drivers["health_status"], 
        y=drivers["avg_tickets"], 
        marker_color="#f39c12"
    ))
    fig.add_trace(go.Bar(
        name="Payment Delay (%)", 
        x=drivers["health_status"], 
        y=drivers["delay_rate"], 
        marker_color="#9b59b6"
    ))

    fig.update_layout(
        barmode="group",
        title="Root Cause Drivers: Healthy vs At-Risk Behavioral Metrics",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320
    )
    return fig


# Legacy/Fallback functions
def plot_health_distribution(df):
    fig = px.histogram(
        df,
        x="health_score",
        color="health_status",
        color_discrete_map={"Healthy": "#2ecc71", "At Risk": "#f1c40f", "Critical": "#e74c3c"},
        title="Account Health Score Distribution"
    )
    return fig

def plot_risk_vs_mrr(df):
    return plot_quadrant_risk_matrix(df)

def plot_health_by_industry(df):
    return plot_industry_health_vs_risk(df)