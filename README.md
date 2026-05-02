Loan Default Prediction System
Project Overview

This project predicts whether a loan applicant is likely to default using machine learning. It helps financial institutions assess risk and make better lending decisions.

Features
Real-time loan risk prediction using Streamlit
Feature engineering (income-to-loan ratio, interest burden)
Custom threshold tuning using F1-score
Risk classification (Low, Moderate, High)

Integration with Power BI dashboard

Tech Stack
Python
Pandas, NumPy
Scikit-learn
Streamlit
Power BI

Project Structure
app/        → Streamlit application  
data/       → Dataset files  
model/      → Trained model + threshold  
src/        → Training code  


How to Run
pip install -r requirements.txt
streamlit run app/app.py
