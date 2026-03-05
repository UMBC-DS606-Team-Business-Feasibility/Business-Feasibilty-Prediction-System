import streamlit as st
import pandas as pd
import joblib
import os

st.title("Business Feasibility Prediction System")

st.write(
    """
This AI system predicts how feasible it is to open a business in a city
based on:

• Yelp business data  
• Census income data  
• BLS employment data  
"""
)

# Load dataset
data_path = "outputs/master_dataset.csv"

if not os.path.exists(data_path):
    st.error("Run main.py first to generate dataset.")
    st.stop()

df = pd.read_csv(data_path)

# City selector
cities = sorted(df["city"].unique())

city = st.selectbox("Select a city", cities)

row = df[df["city"] == city].iloc[0]

st.subheader("City Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Feasibility Score", round(row["feasibility_score"], 3))
col2.metric("Median Income", int(row["median_income"]))
col3.metric("Employment Rate", round(row["employment_rate"], 2))

st.subheader("Business Activity")

col1, col2 = st.columns(2)

col1.metric("Avg Rating", round(row["avg_rating"], 2))
col2.metric("Review Count", int(row["avg_review_count"]))


# Top cities
st.subheader("Top Cities for Business")

top = df.sort_values("feasibility_score", ascending=False).head(10)

st.dataframe(top[["city", "feasibility_score"]])


# Visualizations
st.subheader("EDA Visualizations")

if os.path.exists("outputs/top_cities.png"):
    st.image("outputs/top_cities.png")

if os.path.exists("outputs/correlation_heatmap.png"):
    st.image("outputs/correlation_heatmap.png")

if os.path.exists("outputs/income_vs_feasibility.png"):
    st.image("outputs/income_vs_feasibility.png")
