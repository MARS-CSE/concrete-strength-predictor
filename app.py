
import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🧱")

st.title("🧱 Concrete Compressive Strength Predictor")
st.write("Enter the concrete mix details to predict compressive strength (MPa).")

# Load trained model
model = joblib.load("linear_regression_concrete.pkl")

# Input fields
cement = st.number_input("Cement (kg/m³)", min_value=0.0, value=281.17)
slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, value=73.90)
flyash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, value=54.19)
water = st.number_input("Water (kg/m³)", min_value=0.0, value=181.57)
superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, value=6.20)
coarse_aggregate = st.number_input("Coarse Aggregate (kg/m³)", min_value=0.0, value=972.92)
fine_aggregate = st.number_input("Fine Aggregate (kg/m³)", min_value=0.0, value=773.58)
age = st.number_input("Age (days)", min_value=1.0, value=28.0)

if st.button("Predict Strength"):
    features = np.array([[cement, slag, flyash, water, superplasticizer,
                          coarse_aggregate, fine_aggregate, age]])

    pred = model.predict(features)[0]

    # Strength classification
    if pred < 15:
        category = "Very Weak"
    elif pred < 30:
        category = "Weak"
    elif pred < 45:
        category = "Moderate"
    else:
        category = "Strong"

    st.success(f"Predicted Compressive Strength: {pred:.2f} MPa")
    st.info(f"Strength Category: **{category}**")
