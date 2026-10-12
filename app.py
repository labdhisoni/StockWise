
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="StockWise | Inventory Intelligence",
    page_icon="📦",
    layout="wide"
)

st.title("📦 StockWise")
st.caption("Smart Inventory Management & Stockout Predictor")

DATA_DIR = Path(__file__).parent / "data"

@st.cache_data
def load_data():
    products = pd.read_csv(DATA_DIR / "products.csv")
    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["date"])
    return products, sales

products, sales = load_data()

# Use the most recent 14 calendar days in the dataset
end_date = sales["date"].max()
start_date = end_date - pd.Timedelta(days=13)
recent = sales[sales["date"].between(start_date, end_date)]

demand = (
    recent.groupby("product_id")["units_sold"]
    .sum()
    .rename("units_sold_14d")
)

df = products.merge(demand, on="product_id", how="left")
df["units_sold_14d"] = df["units_sold_14d"].fillna(0)
df["avg_daily_demand"] = df["units_sold_14d"] / 14
df["lead_time_demand"] = (
    df["avg_daily_demand"] * df["lead_time_days"]
)
df["safety_stock"] = np.ceil(
    df["lead_time_demand"] * 0.20
)
df["reorder_point"] = np.ceil(
    df["lead_time_demand"] + df["safety_stock"]
)
df["days_of_stock"] = np.where(
    df["avg_daily_demand"] > 0,
    df["current_stock"] / df["avg_daily_demand"],
    np.inf
)
df["recommended_order_qty"] = np.where(
    df["avg_daily_demand"] > 0,
    np.maximum(0, np.ceil(
        df["reorder_point"] - df["current_stock"]
    )),
    0
).astype(int)

df["stock_status"] = np.select(
    [
        df["avg_daily_demand"].eq(0),
        df["days_of_stock"] <= df["lead_time_days"],
        df["current_stock"] <= df["reorder_point"]
    ],
    ["Insufficient demand data", "High Risk", "Reorder Soon"],
    default="Healthy"
)

# Sidebar filters
st.sidebar.header("Filters")
categories = ["All"] + sorted(df["category"].unique().tolist())
selected_category = st.sidebar.selectbox("Category", categories)
search = st.sidebar.text_input("Search product")

filtered = df.copy()
if selected_category != "All":
    filtered = filtered[filtered["category"] == selected_category]
if search:
    filtered = filtered[
        filtered["product_name"].str.contains(
            search, case=False, na=False
        )
    ]

# KPI cards
total_products = len(filtered)
inventory_value = (
    filtered["current_stock"] * filtered["unit_cost"]
).sum()
high_risk = (filtered["stock_status"] == "High Risk").sum()
reorder_count = filtered["stock_status"].isin(
    ["High Risk", "Reorder Soon"]
).sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Products", f"{total_products}")
c2.metric("Inventory Value", f"₹{inventory_value:,.0f}")
c3.metric("High-Risk Products", f"{high_risk}")
c4.metric("Need Reordering", f"{reorder_count}")

st.divider()

# Visualizations
left, right = st.columns(2)

with left:
    st.subheader("Current Stock vs Reorder Point")
    chart_df = filtered.melt(
        id_vars=["product_name"],
        value_vars=["current_stock", "reorder_point"],
        var_name="Metric",
        value_name="Units"
    )
    fig = px.bar(
        chart_df,
        x="product_name",
        y="Units",
        color="Metric",
        barmode="group",
        title="Inventory Thresholds"
    )
    fig.update_layout(xaxis_title="", yaxis_title="Units")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Inventory Status")
    status_counts = (
        filtered["stock_status"].value_counts().rename_axis("Status")
        .reset_index(name="Products")
    )
    fig2 = px.pie(
        status_counts, names="Status", values="Products",
        hole=0.45
    )
    st.plotly_chart(fig2, use_container_width=True)

# Reorder recommendations
st.subheader("📋 Inventory & Reorder Recommendations")
priority = {"High Risk": 0, "Reorder Soon": 1,
            "Healthy": 2, "Insufficient demand data": 3}
filtered = filtered.copy()
filtered["priority"] = filtered["stock_status"].map(priority)
filtered = filtered.sort_values(
    ["priority", "days_of_stock"], ascending=[True, True]
)

display_cols = [
    "product_name", "category", "current_stock",
    "avg_daily_demand", "lead_time_days", "reorder_point",
    "days_of_stock", "recommended_order_qty", "stock_status"
]
table = filtered[display_cols].copy()
table["days_of_stock"] = table["days_of_stock"].replace(
    [np.inf, -np.inf], np.nan
).round(1)
table["avg_daily_demand"] = table["avg_daily_demand"].round(2)

st.dataframe(table, use_container_width=True, hide_index=True)

csv = table.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Inventory Report",
    data=csv,
    file_name="stockwise_inventory_report.csv",
    mime="text/csv"
)

st.caption(
    "StockWise uses a preliminary 20% safety-stock assumption. "
    "Recommendations are estimates and should be validated against "
    "real demand, supplier reliability, and outstanding orders."
)
