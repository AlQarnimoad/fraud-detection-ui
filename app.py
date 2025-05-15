import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("fraud_detection_model.pkl")

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("🔍 Healthcare Provider Fraud Detection")
st.write("Enter provider-level billing data below to check if it's potentially fraudulent.")

with st.form("fraud_form"):
    st.subheader("📄 Claim Summary")
    avg_claim_amt = st.slider("Average Claim Amount (SAR)", 0, 20000, 3500)
    total_claim_amt = st.slider("Total Claimed Amount (SAR)", 0, 200000, 90000)
    num_claims = st.slider("Number of Claims", 1, 200, 5)
    unique_patients = st.slider("Unique Patients", 1, 200, 20)

    st.subheader("🧑‍⚕️ Patient Profile")
    avg_patient_age = st.slider("Average Patient Age", 18, 100, 65)
    avg_chronic_conditions = st.slider("Avg Chronic Conditions", 0, 10, 5)
    claims_after_death = st.slider("Claims After Death", 0, 10, 0)

    st.subheader("⚙️ Clinical Details")
    avg_diagnosis_count = st.slider("Avg Diagnosis Count", 0, 10, 4)
    avg_procedure_count = st.slider("Avg Procedure Count", 0.0, 5.0, 0.2, step=0.1)
    num_inpatient_claims = st.slider("Inpatient Claims", 0, 50, 5)

    submitted = st.form_submit_button("🔍 Predict Fraud Risk")

if submitted:
    input_data = pd.DataFrame([{
        'avg_claim_amt': avg_claim_amt,
        'total_claim_amt': total_claim_amt,
        'num_claims': num_claims,
        'unique_patients': unique_patients,
        'claims_after_death': claims_after_death,
        'avg_patient_age': avg_patient_age,
        'avg_diagnosis_count': avg_diagnosis_count,
        'avg_procedure_count': avg_procedure_count,
        'num_inpatient_claims': num_inpatient_claims,
        'avg_chronic_conditions': avg_chronic_conditions
    }])

    prediction = model.predict(input_data)[0]
    fraud_prob = model.predict_proba(input_data)[0][1]

    st.success("✅ Not Fraudulent" if prediction == 0 else "🚨 Potentially Fraudulent")
    st.metric(label="Fraud Probability", value=f"{fraud_prob:.2%}")
