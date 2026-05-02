#%%
# ================================
# IMPORT LIBRARIES
# ================================
import streamlit as st
import pandas as pd
import pickle

#%%
# ================================
# LOAD MODEL + THRESHOLD
# ================================
## model = pickle.load(open("loan_model.pkl", "rb"))
## threshold = pickle.load(open("threshold.pkl", "rb"))


model = pickle.load(open("model/loan_model.pkl", "rb"))
threshold = pickle.load(open("model/threshold.pkl", "rb"))

#%%
# ================================
# TITLE
# ================================
st.title("Loan Default Prediction System")

st.write("Enter applicant details below:")

#%%
# ================================
# USER INPUTS
# ================================
annual_income = st.number_input("Annual Income", min_value=1000.0, value=50000.0)
loan_amount = st.number_input("Loan Amount", min_value=100.0, value=10000.0)
installment = st.number_input("Installment", min_value=10.0, value=300.0)
int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0)
dti = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, value=15.0)
total_acc = st.number_input("Total Accounts", min_value=1, value=10)
issue_year = st.number_input("Issue Year", min_value=2000, max_value=2030, value=2020)
issue_month = st.number_input("Issue Month", min_value=1, max_value=12, value=6)

#%%
# ================================
# FEATURE ENGINEERING
# ================================
income_loan_ratio = annual_income / loan_amount if loan_amount != 0 else 0
interest_burden = installment / annual_income if annual_income != 0 else 0

#%%
# ================================
# CREATE INPUT DATAFRAME
# ================================
input_data = pd.DataFrame([[
    annual_income,
    loan_amount,
    installment,
    int_rate,
    dti,
    total_acc,
    issue_year,
    issue_month,
    income_loan_ratio,
    interest_burden
]], columns=[
    'annual_income',
    'loan_amount',
    'installment',
    'int_rate',
    'dti',
    'total_acc',
    'issue_year',
    'issue_month',
    'income_loan_ratio',
    'interest_burden'
])

#%%
# ================================
# PREDICTION BUTTON
# ================================
if st.button("Predict Risk"):

    # Get probability
    probability = model.predict_proba(input_data)[0][1]

    # Display probability
    st.subheader(f"Default Probability: {round(probability, 3)}")
    st.write(f"Model Threshold: {round(threshold, 3)}")

    # Final Decision
    if probability > threshold:
        st.error("High Risk: Loan is likely to default")
    else:
        st.success("Low Risk: Loan is safe")

    #%%
    # ================================
    # RISK INTERPRETATION (VERY IMPORTANT)
    # ================================
    if probability < 0.3:
        st.info("Risk Level: Very Low")
    elif probability < 0.6:
        st.warning("Risk Level: Moderate")
    else:
        st.error("Risk Level: Very High")

    #%%
    # ================================
    # DEBUG INFO (FOR YOU)
    # ================================
    st.write("Probability:", probability)
    st.write("Threshold:", threshold)
    
    st.write("Input Data Used for Prediction:")
    st.write(input_data)
