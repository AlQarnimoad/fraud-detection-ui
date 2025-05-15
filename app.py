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
        min_value=0, max_value=20000, value=3500,
        help="The average amount billed per claim. Found in provider billing summaries."
    )

    total_claim_amt = st.slider(
        "📊 Total Claim Amount (SAR)", 
        min_value=0, max_value=300000, value=90000,
        help="Total reimbursement received for all claims submitted by the provider."
    )

    num_claims = st.slider(
        "🧾 Number of Claims Submitted", 
        min_value=1, max_value=1000, value=25,
        help="Total number of claims the provider submitted during the period."
    )

    unique_patients = st.slider(
        "👥 Unique Patients Seen", 
        min_value=1, max_value=500, value=20,
        help="Total number of different patients the provider treated."
    )

    st.subheader("👤 Patient Profile")

    avg_patient_age = st.slider(
        "🎂 Average Patient Age", 
        min_value=18, max_value=100, value=65,
        help="Estimated average age of the provider's patients."
    )

    avg_chronic_conditions = st.slider(
        "🫀 Average Chronic Conditions", 
        min_value=0, max_value=10, value=5,
        help="Average number of chronic illnesses (e.g., diabetes, hypertension) per patient."
    )

    claims_after_death = st.slider(
        "☠️ Claims After Patient Death", 
        min_value=0, max_value=10, value=0,
        help="Number of claims submitted after the listed date of patient death (should be 0)."
    )

    st.subheader("⚙️ Clinical Complexity")

    avg_diagnosis_count = st.slider(
        "🧪 Avg Diagnosis Codes per Claim", 
        min_value=0, max_value=10, value=4,
        help="How many diagnoses are recorded per claim on average."
    )

    avg_procedure_count = st.slider(
        "🛠️ Avg Procedure Codes per Claim", 
        min_value=0.0, max_value=5.0, value=0.2, step=0.1,
        help="Average number of medical procedures billed per claim."
    )

    num_inpatient_claims = st.slider(
        "🏥 Inpatient Claims Count", 
        min_value=0, max_value=200, value=5,
        help="How many inpatient (hospitalized) claims the provider submitted."
    )

    submitted = st.form_submit_button("🔍 Check Fraud Risk")

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

    st.markdown("## 🧠 Prediction Result")
    st.success("✅ Provider is **Not Fraudulent**" if prediction == 0 else "🚨 Provider is **Potentially Fraudulent**")
    st.metric(label="📈 Fraud Probability", value=f"{fraud_prob:.2%}")
