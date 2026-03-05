import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🧱")

# Title
st.title("🧱 Concrete Compressive Strength Predictor")

st.write(
    "Enter the concrete mix components below to predict the compressive strength "
    "of concrete using a Linear Regression model."
)

# Strength category table
st.subheader("Concrete Strength Categories")

st.table({
    "Strength Range (MPa)": ["< 15", "15 – 30", "30 – 45", "> 45"],
    "Category": ["Very Weak", "Weak", "Moderate", "Strong"]
})

# Load trained model
model = joblib.load("linear_regression_concrete.pkl")

st.subheader("Enter Concrete Mix Values")

# Input fields
cement = st.number_input("Cement (kg/m³)", min_value=0.0, value=281.17)
slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, value=73.90)
flyash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, value=54.19)
water = st.number_input("Water (kg/m³)", min_value=0.0, value=181.57)
superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, value=6.20)
coarse_aggregate = st.number_input("Coarse Aggregate (kg/m³)", min_value=0.0, value=972.92)
fine_aggregate = st.number_input("Fine Aggregate (kg/m³)", min_value=0.0, value=773.58)
age = st.number_input("Age (days)", min_value=1.0, value=28.0)

# Prediction button
if st.button("Predict Strength"):

    features = np.array([[cement, slag, flyash, water,
                          superplasticizer, coarse_aggregate,
                          fine_aggregate, age]])

    prediction = model.predict(features)[0]

    # Strength classification
    if prediction < 15:
        category = "Very Weak"
        st.error(f"Predicted Strength: {prediction:.2f} MPa | Category: {category}")

    elif prediction < 30:
        category = "Weak"
        st.warning(f"Predicted Strength: {prediction:.2f} MPa | Category: {category}")

    elif prediction < 45:
        category = "Moderate"
        st.info(f"Predicted Strength: {prediction:.2f} MPa | Category: {category}")

    else:
        category = "Strong"
        st.success(f"Predicted Strength: {prediction:.2f} MPa | Category: {category}")

# Footer
st.markdown("---")
st.caption("Model: Multiple Linear Regression trained on the Concrete Compressive Strength Dataset")
