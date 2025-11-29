import streamlit as st
import pandas as pd
import altair as alt
from app import run_smartcorp_ai
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime

st.set_page_config(page_title="SmartCorp AI GenAI Dashboard", layout="wide")
st.title("AI-Powered Corporate Insights Platform")

# --- Load processed data ---
df = run_smartcorp_ai()

# --- Sidebar Filters ---
st.sidebar.header("Filters & Options")

# Department filter
departments = df["department"].unique().tolist()
selected_departments = st.sidebar.multiselect("Select Departments", options=departments, default=departments)

# KPI filter
kpi_options = ["attrition_rate", "engagement_score", "project_delay_index"]
selected_kpis = st.sidebar.multiselect("Select KPIs to view trends", options=kpi_options, default=kpi_options)

# Date range filter
min_date, max_date = df["date"].min(), df["date"].max()
selected_dates = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Summarization style
summary_style = st.sidebar.selectbox("Summarization Style", ["Brief", "Detailed", "Action-Oriented"])

# Apply filters
filtered_df = df[
    (df["department"].isin(selected_departments)) &
    (df["date"] >= pd.to_datetime(selected_dates[0])) &
    (df["date"] <= pd.to_datetime(selected_dates[1]))
]

# --- AI Risk Prediction using Decision Tree ---
st.header("Predicted Risk Levels (AI)")
risk_features = ["attrition_rate", "engagement_score", "project_delay_index"]
# Encode risk labels for training
le = LabelEncoder()
df["risk_label_encoded"] = le.fit_transform(df["risk_level"])
# Train simple model on full dataset
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(df[risk_features], df["risk_label_encoded"])
# Predict on filtered
filtered_df["predicted_risk_encoded"] = clf.predict(filtered_df[risk_features])
filtered_df["predicted_risk"] = le.inverse_transform(filtered_df["predicted_risk_encoded"])
st.dataframe(filtered_df[["date","department","meeting_summary","predicted_risk","sentiment_label","employee_feedback"]])

# --- Sentiment Distribution ---
st.header("Sentiment Distribution by Department")
sentiment_count = filtered_df.groupby(["department", "sentiment_label"]).size().reset_index(name="count")
sentiment_chart = alt.Chart(sentiment_count).mark_bar().encode(
    x=alt.X("department:N", title="Department"),
    y=alt.Y("count:Q", title="Number of Feedback"),
    color=alt.Color("sentiment_label:N", title="Sentiment"),
    tooltip=["department", "sentiment_label", "count"]
).properties(width=700, height=400)
st.altair_chart(sentiment_chart)

# --- KPI Trends with Forecast ---
st.header("KPI Trends & Forecasts")
for dept in filtered_df["department"].unique():
    st.subheader(f"Department: {dept}")
    dept_df = filtered_df[filtered_df["department"] == dept]

    for kpi in selected_kpis:
        # Forecast using simple linear regression
        lr = LinearRegression()
        X = np.arange(len(dept_df)).reshape(-1, 1)
        y = dept_df[kpi].values
        lr.fit(X, y)
        forecast = lr.predict(X)
        dept_df[f"{kpi}_forecast"] = forecast

        # Combine actual + forecast in chart
        trend_chart = alt.Chart(dept_df).transform_fold(
            fold=[kpi, f"{kpi}_forecast"],
            as_=["KPI", "Value"]
        ).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("Value:Q", title=kpi),
            color=alt.Color("KPI:N", title="Actual / Forecast"),
            tooltip=["date:T", "KPI:N", "Value:Q"]
        ).properties(width=700, height=300)
        st.altair_chart(trend_chart)

# --- Download Filtered Data ---
st.sidebar.header("Download Data")
csv_data = filtered_df.to_csv(index=False)
st.sidebar.download_button("Download Filtered CSV", data=csv_data, file_name="filtered_smartcorp_data.csv", mime="text/csv")
