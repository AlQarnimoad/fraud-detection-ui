import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("fraud_detection_model.pkl")

st.set_page_config(page_title="Healthcare Fraud Detection", layout="centered")
st.title("🩺 Healthcare Provider Fraud Risk Checker")

st.markdown("Use this tool to estimate whether a provider may be flagged as fraudulent based on billing behavior.")

with st.form("fraud_form"):
    st.subheader("📄 Claim Summary")
    
    avg_claim_amt = st.slider(
        "💸 Average Claim Amount (SAR)", 
        0, 20000, 3500,
        help="The average amount billed per claim. Found in provider billing summaries."
