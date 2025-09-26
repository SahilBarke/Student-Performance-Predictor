import streamlit as st
import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sm

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as s:
    scaler = pickle.load(s)

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("Student Performance Index Predictor")
st.markdown("Enter the student data below:")

# Input fields (daily-based constraints)
hours = st.number_input("Hours Studied (per day)", min_value=0.0, max_value=16.0, value=4.0)
previous_scores = st.number_input(
    "Previous Scores (%)", min_value=0.0, max_value=100.0, value=75.0
)
extracurricular = st.number_input(
    "Extracurricular Activities (scale 0–10)", min_value=0.0, max_value=10.0, value=5.0
)
sleep = st.number_input("Sleep Hours (per day)", min_value=0.0, max_value=12.0, value=7.0)
sample_papers = st.number_input("Sample Papers Practiced", min_value=0.0, value=5.0)

# Normalize study + sleep if > 24 hours
total_time = hours + sleep
if total_time > 24:
    hours = round((hours / total_time) * 24, 1)
    sleep = round((sleep / total_time) * 24, 1)
    st.warning(f"⏳ Adjusted to fit 24h/day: Study = {hours}h, Sleep = {sleep}h")

if st.button("Predict Performance Index"):
    # Prepare input data
    input_dict = {
        "Hours Studied": hours,
        "Previous Scores": previous_scores,
        "Extracurricular Activities": extracurricular,
        "Sleep Hours": sleep,
        "Sample Question Papers Practiced": sample_papers,
    }

    input_df = pd.DataFrame([input_dict])

    # Scale input data
    input_scaled = scaler.transform(input_df)

    # Add constant/intercept column 
    input_scaled_const = sm.add_constant(input_scaled, has_constant="add")

    # Predict using statsmodels OLS
    prediction = model.predict(input_scaled_const)[0]

    # Display result
    st.success(f"📈 Predicted Performance Index: **{prediction:.2f}**")
    st.markdown(
        "This index is a composite score based on various factors affecting student performance."
    )
    st.markdown(
        "Predictions are generated using a general dataset and might not fully represent individual performance. Please use this as a reference only."
    )
