import streamlit as st 
import pandas as pd
import numpy as np
import joblib
from custom_transformers import DataFrameSelector , CombinedAttributesAdder

# Load model and pipeline
model = joblib.load("housing_model.pkl")
pipeline = joblib.load("full_pipeline.pkl")  # You need to dump this from your notebook

st.set_page_config(page_title="California House Price Predictor", layout="centered")
st.title("🏡 California Housing Price Prediction")

st.markdown("Enter the details below to predict the housing price:")

# Sidebar inputs for original raw features
longitude = st.sidebar.slider("Longitude", -124.35, -114.31, -120.0)
latitude = st.sidebar.slider("Latitude", 32.54, 41.95, 37.0)
housing_median_age = st.sidebar.slider("Housing Median Age", 1, 52, 20)
total_rooms = st.sidebar.number_input("Total Rooms", 1, 10000, 2000)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", 1, 3000, 400)
population = st.sidebar.number_input("Population", 1, 10000, 1000)
households = st.sidebar.number_input("Households", 1, 5000, 500)
median_income = st.sidebar.slider("Median Income", 0.5, 15.0, 4.0)
ocean_proximity = st.sidebar.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

# Assemble into DataFrame
input_data = pd.DataFrame([{
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity
}])

# Apply same transformation pipeline before prediction
prepared_data = pipeline.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(prepared_data)[0]
    st.success(f"🏠 Predicted House Value: ${prediction:,.2f}")

