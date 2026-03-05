import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🧱",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #eef2f3, #dfe9f3);
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #1f4e79;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
}

.card {
    padding: 20px;
    border-radius: 10px;
    background-color: white;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">🧱 Concrete Strength Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict concrete compressive strength using Linear Regression</p>', unsafe_allow_html=True)

# Load model
model = joblib.load("linear_regression_concrete.pkl")

# Strength table
st.subheader("Concrete Strength Categories")

st.table({
    "Strength Range (MPa)": ["< 15", "15 – 30", "30 – 45", "> 45"],
    "Category": ["Very Weak", "Weak", "Moderate", "Strong"]
})

st.divider()

st.subheader("Enter Concrete Mix Components")

# Two column layout
col1, col2 = st.columns(2)

with col1:
    cement = st.number_input("Cement (kg/m³)", 0.0, value=281.17)
    slag = st.number_input("Blast Furnace Slag (kg/m³)", 0.0, value=73.90)
    flyash = st.number_input("Fly Ash (kg/m³)", 0.0, value=54.19)
    water = st.number_input("Water (kg/m³)", 0.0, value=181.57)

with col2:
    superplasticizer = st.number_input("Superplasticizer (kg/m³)", 0.0, value=6.20)
    coarse = st.number_input("Coarse Aggregate (kg/m³)", 0.0, value=972.92)
    fine = st.number_input("Fine Aggregate (kg/m³)", 0.0, value=773.58)
    age = st.number_input("Age (days)", 1.0, value=28.0)

st.divider()

# Prediction button
if st.button("Predict Strength", use_container_width=True):

    features = np.array([[cement, slag, flyash, water,
                          superplasticizer, coarse,
                          fine, age]])

    pred = model.predict(features)[0]

    # Classification
    if pred < 15:
        category = "Very Weak"
        st.error(f"🔴 Predicted Strength: {pred:.2f} MPa | Category: {category}")

    elif pred < 30:
        category = "Weak"
        st.warning(f"🟠 Predicted Strength: {pred:.2f} MPa | Category: {category}")

    elif pred < 45:
        category = "Moderate"
        st.info(f"🟡 Predicted Strength: {pred:.2f} MPa | Category: {category}")

    else:
        category = "Strong"
        st.success(f"🟢 Predicted Strength: {pred:.2f} MPa | Category: {category}")

st.divider()

st.caption("Machine Learning Model: Multiple Linear Regression | Dataset: Concrete Compressive Strength Dataset")
