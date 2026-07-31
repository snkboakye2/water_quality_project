import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

st.title("Water Quality Prediction App")

# Display local image (relative path from app.py)
st.image("assets/odaso.jpeg", caption="Water treatment at Odaso, GH", use_column_width=True)  # [1]

@st.cache_resource
def load_model():
    return joblib.load("ml_pipeline.joblib")

model = load_model()

st.subheader("Model performance")

@st.cache_data
def compute_metrics():
    # Load the same data used for training/evaluation
    df = pd.read_csv("/assets/water_quality_cleaned.csv")

    feature_cols = ['year', 'month', 'day', 'dayofweek', 'quarter', "ph", "turbidity", "conductivity", "temperature"]
    target_col = "coagulant_dosage_log"

    X = df[feature_cols]
    y = df[target_col]

    # Use the pipeline to get predictions on this dataset
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    return r2, rmse

r2, rmse = compute_metrics()

# Show metrics in the main area (or in the sidebar if you prefer)
col1, col2 = st.columns(2)
with col1:
    st.metric("R² score", f"{r2:.3f}")
with col2:
    st.metric("RMSE (kg/day)", f"{rmse:.3f}")

st.subheader("Enter water quality values")

# Date picker
selected_date = st.date_input("Select Date")

# Other inputs
ph = st.number_input("pH", value=7.0, format="%.2f")
turbidity = st.number_input("Turbidity (NTU)", value=0.0, format="%.2f")
conductivity = st.number_input("Conductivity (µS/cm)", value=0.0, format="%.2f")
temperature = st.number_input("Temperature (°C)", value=25.0, format="%.2f")

# Create DataFrame
input_df = pd.DataFrame([{
    "date": pd.to_datetime(selected_date),
    "ph": ph,
    "turbidity": turbidity,
    "conductivity": conductivity,
    "temperature": temperature
}])

# Set date as index
input_df.set_index("date", inplace=True)

# Extract date features
input_df["year"] = input_df.index.year
input_df["month"] = input_df.index.month
input_df["day"] = input_df.index.day
input_df["dayofweek"] = input_df.index.dayofweek    # Monday=0, Sunday=6
input_df["quarter"] = input_df.index.quarter

input_df = input_df[['year', 'month', 'day', 'dayofweek', 'quarter', 'ph', 'turbidity',
                       'conductivity', 'temperature']]


if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction = np.exp(prediction) # convert coagulant dosage from log scale to actual values
    st.success(f"Predicted coagulant dosage (kg/day): {prediction:.3f}")