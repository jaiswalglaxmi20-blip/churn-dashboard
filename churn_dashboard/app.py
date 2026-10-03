import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(page_title="European Bank Churn Analytics", layout="wide")

# ---------------------------------------------------------
# Load & prepare data
# ---------------------------------------------------------
@st.cache_data
def load_data():
       import os
   df = pd.read_csv(os.path.join(os.path.dirname(__file__), "European_Bank.csv"))
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[0, 29, 45, 60, 200],
        labels=["<30", "30-45", "46-60", "60+"]
    )
    df["CreditBand"] = pd.cut(
        df["CreditScore"], bins=[0, 579, 739, 900],
        labels=["Low", "Medium", "High"]
    )
    df["TenureGroup"] = pd.cut(
        df["Tenure"], bins=[-1, 2, 6, 10],
        labels=["New (0-2y)", "Mid (3-6y)", "Long (7-10y)"]
    )
    nonzero_75 = df.loc[df["Balance"] > 0, "Balance"].quantile(0.75)
    df["BalanceSegment"] = np.where(
        df["Balance"] == 0, "Zero-balance",
        np.where(df["Balance"] >= nonzero_75, "High-balance", "Low-balance")
    )
    df["ActiveLabel"] = df["IsActiveMember"].map({1: "Active", 0: "Inactive"})
    return df

df = load_data()

# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.header("Filters")

geo = st.sidebar.multiselect("Geography", sorted(df["Geography"].unique()), default=list(df["Geography"].unique()))
gender = st.sidebar.multiselect("Gender", sorted(df["Gender"].unique()), default=list(df["Gender"].unique()))
age_group = st.sidebar.multiselect("Age Group", list(df["AgeGroup"].cat.categories), default=list(df["AgeGroup"].cat.categories))
active = st.sidebar.multiselect("Membership Status", ["Active", "Inactive"], default=["Active", "Inactive"])
balance_seg = st.sidebar.multiselect("Balance Segment", ["Zero-balance", "Low-balance", "High-balance"], default=["Zero-balance", "Low-balance", "High-balance"])

filtered = df[
    df["Geography"].isin(geo) &
    df["Gender"].isin(gender) &
    df["AgeGroup"].isin(age_group) &
    df["ActiveLabel"].isin(active) &
    df["BalanceSegment"].isin(balance_seg)
]

st.sidebar.markdown(f"**{len(filtered):,}** customers match current filters")

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("Customer Segmentation & Churn Pattern Analytics")
st.caption("European Retail Banking — France, Germany, Spain")

if filtered.empty:
    st.warning("No customers match the selected filters. Adjust filters in the sidebar.")
    st.stop()

# ---------------------------------------------------------
# KPI row
# ---------------------------------------------------------
overall_churn = filtered["Exited"].mean()
hv_churn = filtered.loc[filtered["BalanceSegment"] == "High-balance", "Exited"].mean() if (filtered["BalanceSegment"] == "High-balance").any() else np.nan
geo_risk = filtered.groupby("Geography")["Exited"].mean()
engagement_drop = filtered.groupby("ActiveLabel")["Exited"].mean()
balance_at_risk = filtered.loc[filtered["Exited"] == 1, "Balance"].sum()
balance_at_risk_pct = balance_at_risk / filtered["Balance"].sum() if filtered["Balance"].sum() > 0 else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Overall Churn Rate", f"{overall_churn:.1%}")
k2.metric("High-Value Churn Ratio", f"{hv_churn:.1%}" if not np.isnan(hv_churn) else "N/A")
k3.metric("Highest-Risk Country", geo_risk.idxmax() if not geo_risk.empty else "N/A", f"{geo_risk.max():.1%}" if not geo_risk.empty else "")
k4.metric("Inactive vs Active Churn", f"{engagement_drop.get('Inactive', 0):.1%} / {engagement_drop.get('Active', 0):.1%}")
k5.metric("Balance at Risk", f"€{balance_at_risk/1e6:.1f}M", f"{balance_at_risk_pct:.1%} of total")

st.divider()

# ---------------------------------------------------------
# Tabs = Core Modules from the brief
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Overall Churn Summary",
    "Geography-wise Churn",
    "Age & Tenure Comparison",
    "High-Value Customer Explorer"
])

# --- Tab 1: Overall churn summary ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        churn_counts = filtered["Exited"].map({0: "Retained", 1: "Churned"}).value_counts().reset_index()
        churn_counts.columns = ["Status", "Count"]
        fig = px.pie(churn_counts, names="Status", values="Count", title="Retained vs Churned",
                     color="Status", color_discrete_map={"Retained": "#2E86AB", "Churned": "#E74C3C"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        prod = filtered.groupby("NumOfProducts")["Exited"].mean().reset_index()
        fig = px.bar(prod, x="NumOfProducts", y="Exited", title="Churn Rate by Number of Products",
                     labels={"Exited": "Churn Rate"}, text_auto=".1%")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        gender_churn = filtered.groupby("Gender")["Exited"].mean().reset_index()
        fig = px.bar(gender_churn, x="Gender", y="Exited", title="Churn Rate by Gender",
                     labels={"Exited": "Churn Rate"}, text_auto=".1%")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        active_churn = filtered.groupby("ActiveLabel")["Exited"].mean().reset_index()
        fig = px.bar(active_churn, x="ActiveLabel", y="Exited", title="Churn Rate: Active vs Inactive",
                     labels={"Exited": "Churn Rate", "ActiveLabel": "Membership"}, text_auto=".1%")
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Geography-wise churn ---
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        geo_churn = filtered.groupby("Geography")["Exited"].agg(["mean", "count"]).reset_index()
        geo_churn.columns = ["Geography", "ChurnRate", "Customers"]
        fig = px.bar(geo_churn, x="Geography", y="ChurnRate", title="Churn Rate by Country",
                     labels={"ChurnRate": "Churn Rate"}, text_auto=".1%",
                     color="Geography")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(geo_churn, x="Geography", y="Customers", title="Customer Base by Country",
                     color="Geography")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Geography × Age Interaction")
    cross = pd.crosstab(filtered["Geography"], filtered["AgeGroup"], values=filtered["Exited"], aggfunc="mean")
    fig = px.imshow(cross, text_auto=".1%", aspect="auto", color_continuous_scale="Reds",
                     labels=dict(color="Churn Rate"), title="Churn Rate Heatmap: Country vs Age Group")
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Age & tenure comparison ---
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        age_churn = filtered.groupby("AgeGroup", observed=True)["Exited"].agg(["mean", "count"]).reset_index()
        age_churn.columns = ["AgeGroup", "ChurnRate", "Customers"]
        fig = px.bar(age_churn, x="AgeGroup", y="ChurnRate", title="Churn Rate by Age Group",
                     labels={"ChurnRate": "Churn Rate"}, text_auto=".1%")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        ten_churn = filtered.groupby("TenureGroup", observed=True)["Exited"].agg(["mean", "count"]).reset_index()
        ten_churn.columns = ["TenureGroup", "ChurnRate", "Customers"]
        fig = px.bar(ten_churn, x="TenureGroup", y="ChurnRate", title="Churn Rate by Tenure Group",
                     labels={"ChurnRate": "Churn Rate"}, text_auto=".1%")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Credit Score Bands")
    cs_churn = filtered.groupby("CreditBand", observed=True)["Exited"].agg(["mean", "count"]).reset_index()
    cs_churn.columns = ["CreditBand", "ChurnRate", "Customers"]
    fig = px.bar(cs_churn, x="CreditBand", y="ChurnRate", title="Churn Rate by Credit Score Band",
                 labels={"ChurnRate": "Churn Rate"}, text_auto=".1%")
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: High-value customer explorer ---
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        bal_churn = filtered.groupby("BalanceSegment")["Exited"].agg(["mean", "count"]).reset_index()
        bal_churn.columns = ["BalanceSegment", "ChurnRate", "Customers"]
        fig = px.bar(bal_churn, x="BalanceSegment", y="ChurnRate", title="Churn Rate by Balance Segment",
                     labels={"ChurnRate": "Churn Rate"}, text_auto=".1%")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(filtered, x="Balance", y="EstimatedSalary", color=filtered["Exited"].map({0: "Retained", 1: "Churned"}),
                          title="Balance vs Salary (colored by churn)", opacity=0.5,
                          color_discrete_map={"Retained": "#2E86AB", "Churned": "#E74C3C"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("High-Value Churners (top balance quartile, exited)")
    hv_table = filtered[(filtered["BalanceSegment"] == "High-balance") & (filtered["Exited"] == 1)][
        ["CustomerId", "Geography", "Age", "Balance", "EstimatedSalary", "NumOfProducts", "ActiveLabel"]
    ].sort_values("Balance", ascending=False)
    st.dataframe(hv_table, use_container_width=True, height=300)
    st.caption(f"{len(hv_table)} high-balance customers churned, holding €{hv_table['Balance'].sum():,.0f} combined.")

st.divider()
st.caption("Data: European_Bank.csv — Unified Mentor / European Central Bank churn analytics project")
