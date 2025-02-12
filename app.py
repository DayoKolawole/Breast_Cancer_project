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

# ✅ Define `feature_names` before using it
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# ✅ Now use `feature_names`
st.sidebar.header("Enter Features:")
user_input = []
for feature in feature_names:
    value = st.sidebar.number_input(f"{feature}", min_value=0.0, value=1.0)
    user_input.append(value)
