import streamlit as st
import pandas as pd
import requests

st.title("Fraud Detection API")

# User input form for single transaction prediction
st.subheader("Single Transaction Prediction")
with st.form("fraud_form"):
    amount = st.number_input("Amount", min_value=0.0, format="%.2f")
    oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, format="%.2f")
    newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, format="%.2f")
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, format="%.2f")
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, format="%.2f")
    step = st.number_input("Step", min_value=0, format="%d")
    type_ = st.selectbox("Transaction Type", ["CASH-IN", "CASH-OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    nameOrig = st.text_input("Origin Name", value="C12345678")
    nameDest = st.text_input("Destination Name", value="M98765432")
    isFlaggedFraud = st.radio("Is Flagged Fraud?", [0, 1])

    submitted = st.form_submit_button("Predict")

    if submitted:
        data = {
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "step": step,
            "type": type_,
            "nameOrig": nameOrig,
            "nameDest": nameDest,
            "isFlaggedFraud": isFlaggedFraud,
        }
        response = requests.post("http://localhost:8080/api/predict", json=data)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['prediction']}")
            st.write(f"Probability of Non-Fraud: {result['probability_non_fraud']:.4f}")
            st.write(f"Probability of Fraud: {result['probability_fraud']:.4f}")
        else:
            st.error(f"Error: {response.json()['error']}")

# File upload for batch predictions
st.subheader("Batch Prediction via File Upload")
uploaded_file = st.file_uploader("Upload CSV or Parquet file", type=["csv", "parquet"])

if uploaded_file:
    files = {"file": uploaded_file}
    response = requests.post("http://localhost:8080/api/predict_file", files=files)

    if response.status_code == 200:
        df = pd.read_json(response.text)
        st.dataframe(df)
    else:
        st.error(f"Error: {response.json()['error']}")
