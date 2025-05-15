import streamlit as st
import joblib
import pandas as pd

# Load your saved model
model = joblib.load("fraud_detection_model.pkl")

st.title("🔍 Healthcare Provider Fraud Detection")

st.markdown("Enter provider-level features to predict fraud.")

# Input fields
avg_claim_amt = st.number_input("Average Claim Amount", value=3500.0)
total_claim_amt = st.number_input("Total Claim Amount", value=90000.0)
num_claims = st.number_input("Number of Claims", value=25)
unique_patients = st.number_input("Unique Patients", value=20)
claims_after_death = st.number_input("Claims After Death", value=0)
avg_patient_age = st.number_input("Average Patient Age", value=70.0)
avg_diagnosis_count = st.number_input("Avg Diagnosis Count", value=4.0)
avg_procedure_count = st.number_input("Avg Procedure Count", value=0.2)
num_inpatient_claims = st.number_input("Inpatient Claims Count", value=5)
avg_chronic_conditions = st.number_input("Avg Chronic Conditions", value=5.0)

# Create input DataFrame
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

# Predict button
if st.button("Predict Fraud Risk"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    st.write(f"🧠 **Prediction:** {'Fraudulent' if pred == 1 else 'Not Fraudulent'}")
    st.write(f"📊 **Probability of Fraud:** {prob:.2%}")
