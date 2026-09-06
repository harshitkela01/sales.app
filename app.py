import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Sales Prediction App")
st.title("Sales Prediction using Linear Regression")

# Load the trained model
# Make sure 'linear_regression_model.sav' is in the same directory as this script
try:
    model = joblib.load('linear_regression_model.sav')
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'linear_regression_model.sav' not found. Please ensure the model file is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.write("### Enter advertising budgets to predict sales:")

# Input fields for features
tv = st.number_input("TV Advertising Budget ($)", min_value=0.0, value=300.0, step=150.0)
radio = st.number_input("Radio Advertising Budget ($)", min_value=0.0, value=50.0, step=25.0)
newspaper = st.number_input("Newspaper Advertising Budget ($)", min_value=0.0, value=100.0, step=50.0)

if st.button("Predict Sales"):
    # Create a NumPy array from the inputs
    features = np.array([[tv, radio, newspaper]])

    # Make prediction
    prediction = model.predict(features)[0]

    st.write(f"#### Predicted Sales: {prediction:.2f}")
    st.markdown("--- ")
    st.write("**Note:** This prediction is based on the trained linear regression model.")
