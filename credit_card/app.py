import streamlit as st
import joblib
import numpy as np

# Load trained model
rf_model = joblib.load("rf_smote_model.pkl")

# ------------------- UI -------------------
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection Demo")
st.markdown("Enter transaction details below and click **Predict** to see if the transaction is fraudulent.")

# Input fields
with st.form("fraud_form"):
    step = st.number_input("Step (time step)", min_value=1, value=1)
    type_input = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
    amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
    
    # Submit button
    submitted = st.form_submit_button("Predict")

# ------------------- Prediction -------------------
if submitted:
    # Encode 'type' using same mapping as training
    type_mapping = {'CASH_OUT':0, 'CASH_IN':1, 'DEBIT':2, 'PAYMENT':3, 'TRANSFER':4}
    type_encoded = type_mapping[type_input]

    # Prepare features
    X_new = np.array([[step, type_encoded, amount]])

    # Predict
    prediction = rf_model.predict(X_new)
    prob = rf_model.predict_proba(X_new)[0][1] * 100  # convert to percent

    # Show results nicely
    st.markdown("### Prediction Result:")
    if prediction[0] == 1:
        st.error(f"⚠️ Fraudulent Transaction! Probability: {prob:.2f}%")
    else:
        st.success(f"✅ Not Fraudulent. Probability of Fraud: {prob:.2f}%")
