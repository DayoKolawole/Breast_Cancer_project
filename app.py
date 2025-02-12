import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
try:
    scaler, model = joblib.load("model.pkl")
except:
    st.error("⚠ Model file not found! Train and save the model first.")
    st.stop()

# Define feature input fields
st.title("🔬 Breast Cancer Prediction App")
st.write("Enter the feature values to predict breast cancer diagnosis.")

# Feature names (Ensure they match those in training)
feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 
                 'mean smoothness', 'worst radius', 'worst texture', 'worst perimeter', 
                 'worst area', 'worst smoothness']

# Create input fields for user
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, value=1.0)
    user_input.append(value)

# Prediction button
if st.button("Predict"):
    user_data = np.array(user_input).reshape(1, -1)
    user_data_scaled = scaler.transform(user_data)  # Scale input data
    prediction = model.predict(user_data_scaled)  # Make prediction

    # Display result
    result = "Malignant (Cancerous)" if prediction[0] == 1 else "Benign (Non-Cancerous)"
    st.success(f"🎯 Prediction: {result}")