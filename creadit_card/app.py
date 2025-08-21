import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model, scaler, and selected features
model = joblib.load("rf_smote_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

st.title("Credit Card Fraud Detection")

# 🎯 Sample data for autofill
sample_inputs = {
    "Non-Fraudulent": {
        'V14': -0.021053, 'V17': 0.066928, 'V12': -0.110474, 'V10': -0.255425, 'V11': 0.098698,
        'V16': 0.128539, 'V4': 1.378155, 'V3': 2.536347, 'V9': 0.363787, 'V18': 0.277838,
        'V7': 0.239599, 'V2': -0.072781, 'Amount': 149.62
    },
    "Fraudulent": {
        'V14': 0.429781, 'V17': -0.355519, 'V12': -0.186117, 'V10': -4.622730, 'V11': 9.435084,
        'V16': 0.697103, 'V4': 5.664820, 'V3': -16.298091, 'V9': -6.795398, 'V18': -0.832074,
        'V7': -14.716668, 'V2': 8.075240, 'Amount': 34.12
    }
}

# 🧪 Choose autofill or manual
sample_choice = st.radio("Choose Sample or Manual Input:", ["Manual", "Non-Fraudulent", "Fraudulent"])

user_input = {}

if sample_choice == "Manual":
    st.write("Enter the values manually:")
    for feature in selected_features:
        user_input[feature] = st.number_input(f"Enter {feature}:", step=0.01)
else:
    user_input = sample_inputs[sample_choice]
    st.write(f"Using sample data for **{sample_choice}**")
    for feature in selected_features:
        st.text(f"{feature}: {user_input[feature]}")

# ➕ Make prediction
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])[selected_features]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction.")
