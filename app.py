import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
try:
    scaler, model = joblib.load("model.pkl")
except:
    st.error("⚠ Model file not found! Train and save the model first.")
    st.stop()

# Set page configuration
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

# Apply custom background
page_bg_img = """
<style>
body {
    background-image: url("https://source.unsplash.com/1600x900/?health,hospital,medical");
    background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Custom CSS for UI
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
        padding: 20px;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    .big-font {
        font-size: 30px !important;
        text-align: center;
        color: #ff4b4b;
        font-weight: bold;
    }
    .sub-font {
        font-size: 18px !important;
        text-align: center;
        color: #444;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown('<p class="big-font">🔬 Breast Cancer Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">Enter patient details below to predict if the tumor is Malignant or Benign.</p>', unsafe_allow_html=True)

# Define Feature Names
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Layout: Sidebar for Inputs, Main Area for Results
st.sidebar.header("📝 Enter Features")
user_input = []
for feature in feature_names:
    value = st.sidebar.number_input(f"{feature}", min_value=0.0, value=1.0)
    user_input.append(value)

# Convert to NumPy array
user_data = np.array(user_input).reshape(1, -1)

# Create Columns for Layout
col1, col2 = st.columns([2, 3])

with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6f/Cancer_Cell_Image.jpg", use_column_width=True)

with col2:
    st.subheader("Model Prediction")

    # Predict Button
    if st.button("🔍 Predict", help="Click to predict if the tumor is Malignant or Benign"):
        # Validate input dimensions before transformation
        if user_data.shape[1] != scaler.n_features_in_:
            st.error(f"⚠ Expected {scaler.n_features_in_} features, but received {user_data.shape[1]}. Please check inputs.")
            st.stop()

        # Scale input data
        user_data_scaled = scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)

        # Display Result
        result = "⚠ Malignant (Cancerous)" if prediction[0] == 1 else "✅ Benign (Non-Cancerous)"
        st.success(f"🎯 Prediction: **{result}**")

# Visualizing Feature Importance (Mock Example)
st.subheader("📊 Feature Importance")
feature_importance = np.random.rand(len(feature_names))  # Mock importance values
fig, ax = plt.subplots()
ax.barh(feature_names[:10], feature_importance[:10], color="#ff4b4b")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
ax.set_title("Top 10 Most Important Features")
st.pyplot(fig)

# Footer
st.markdown('<p class="footer">Developed with ❤️ using Streamlit</p>', unsafe_allow_html=True)

