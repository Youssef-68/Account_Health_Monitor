import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.health_engine import get_health_summary, calculate_health_score, categorize_health
from src.visualizations import (
    plot_health_donut,
    plot_quadrant_risk_matrix,
    plot_industry_health_vs_risk,
    plot_risk_drivers
)
from src.models import load_or_train

st.set_page_config(page_title="Account Health Monitor", layout="wide")

st.title("Account Health Monitor App")
st.markdown("Real-time proactive monitoring for customer retention and revenue protection.")

@st.cache_data
def fetch_data():
    return load_data()

df = fetch_data()
model = load_or_train(df)

# Sidebar Filters
st.sidebar.header("Filter Portfolio")
selected_status = st.sidebar.multiselect(
    "Health Status",
    options=["Healthy", "At Risk", "Critical"],
    default=["Healthy", "At Risk", "Critical"]
)
selected_industry = st.sidebar.selectbox("Industry", ["All"] + list(df["industry"].unique()))

filtered_df = df[df["health_status"].isin(selected_status)]
if selected_industry != "All":
    filtered_df = filtered_df[filtered_df["industry"] == selected_industry]

# Top KPI Overview Metrics
summary = get_health_summary(filtered_df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Active Accounts", f"{summary['Total Accounts']:,}")
c2.metric("Portfolio Health Score", f"{summary['Average Health Score']} / 100")
c3.metric("Healthy Accounts", summary["Healthy Accounts"])
c4.metric("At-Risk Accounts", summary["At-Risk Accounts"])
c5.metric("MRR at Risk", f"${summary['MRR at Risk ($)']:,.0f}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "Portfolio Health Overview", 
    "At-Risk Radar & Action Center", 
    "Single Account Simulator"
])

# Tab 1: Portfolio Health Overview
with tab1:
    # Row 1: Donut split + Industry Benchmark
    r1_col1, r1_col2 = st.columns([1, 2])
    with r1_col1:
        st.plotly_chart(plot_health_donut(filtered_df), use_container_width=True)
    with r1_col2:
        st.plotly_chart(plot_industry_health_vs_risk(filtered_df), use_container_width=True)

    # Row 2: Full-width Risk Matrix
    st.plotly_chart(plot_quadrant_risk_matrix(filtered_df), use_container_width=True)


# Tab 2: At-Risk Radar & Action Center
with tab2:
    st.subheader("High Exposure Accounts (Immediate Action Required)")
    
    # Filter out critical and at-risk accounts only for this tab
    at_risk_df = filtered_df[filtered_df["health_status"].isin(["Critical", "At Risk"])].sort_values(
        by="current_mrr", ascending=False
    )

    if len(at_risk_df) == 0:
        st.success("Excellent! No accounts currently require urgent intervention.")
    else:
        # Spotlight Section: Top 3 Highest MRR at risk
        top_critical = at_risk_df.head(3)
        st.markdown("##### Top Priority Accounts to Save This Week:")
        
        card_cols = st.columns(len(top_critical))
        for idx, (_, row) in enumerate(top_critical.iterrows()):
            with card_cols[idx]:
                st.warning(
                    f"**Account ID:** {row['account_id']}\n\n"
                    f"**MRR:** ${row['current_mrr']:,.2f}\n\n"
                    f"**Health Score:** {row['health_score']}/100 ({row['health_status']})\n\n"
                    f"**Main Issue:** {'Payment Delay' if row['payment_delay_flag'] else f'{row['tickets_count']} Tickets'}"
                )

        st.markdown("---")

        # Visual Root Cause Comparison & Table
        col_chart, col_table = st.columns([1, 1.5])
        
        with col_chart:
            st.plotly_chart(plot_risk_drivers(filtered_df), use_container_width=True)

        with col_table:
            st.markdown("##### 📋 Priority Worklist")
            st.dataframe(
                at_risk_df[[
                    "account_id", "health_score", "current_mrr", 
                    "company_size", "error_rate", "payment_delay_flag", "tickets_count"
                ]],
                column_config={
                    "account_id": "Account",
                    "health_score": st.column_config.ProgressColumn(
                        "Health Score",
                        help="0 to 100 Health Score",
                        format="%d",
                        min_value=0,
                        max_value=100,
                    ),
                    "current_mrr": st.column_config.NumberColumn(
                        "MRR ($)",
                        format="$%.2f"
                    ),
                    "error_rate": st.column_config.NumberColumn(
                        "Error Rate",
                        format="%.3f"
                    ),
                    "payment_delay_flag": st.column_config.CheckboxColumn(
                        "Delayed Payment?"
                    )
                },
                use_container_width=True,
                height=320
            )


# Tab 3: Single Account Simulator
with tab3:
    st.subheader("Account Health Simulator (What-If Analysis)")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        s_size = st.selectbox("Company Size", df["company_size"].unique())
        s_ind = st.selectbox("Industry", df["industry"].unique())
        s_contract = st.selectbox("Contract Type", df["contract_type"].unique())
        s_mrr = st.number_input("Current MRR ($)", value=2500.0)

    with col_b:
        s_users = st.number_input("Active Users", value=40)
        s_growth = st.slider("Usage Growth Rate", -0.5, 0.5, -0.1)
        s_adoption = st.slider("Feature Adoption Rate", 0.0, 1.0, 0.4)

    with col_c:
        s_error = st.slider("System Error Rate", 0.0, 0.1, 0.04)
        s_tickets = st.number_input("Support Tickets Count", value=6)
        s_delay = st.selectbox("Payment Delayed?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    if st.button("Assess Health Score", type="primary"):
        input_dict = {
            "company_size": s_size, "industry": s_ind, "contract_type": s_contract,
            "active_users": s_users, "usage_growth": s_growth, "feature_adoption_rate": s_adoption,
            "error_rate": s_error, "tickets_count": s_tickets, "payment_delay_flag": s_delay,
            "current_mrr": s_mrr
        }
        
        sim_score = calculate_health_score(input_dict)
        sim_status = categorize_health(sim_score)
        
        df_sim = pd.DataFrame([input_dict])
        pred_mrr = float(model.predict(df_sim)[0])

        st.markdown("### Simulation Result")
        res1, res2, res3 = st.columns(3)
        res1.metric("Calculated Health Score", f"{sim_score} / 100")
        res2.metric("Health Category", sim_status)
        res3.metric("Projected Next MRR", f"${pred_mrr:,.2f}", delta=f"${pred_mrr - s_mrr:,.2f}")