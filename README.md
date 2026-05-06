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


## Power BI Dashboard

### Summary Dashboard
<img width="1265" height="704" alt="Summary" src="https://github.com/user-attachments/assets/3023f6f9-dd4d-4501-937b-7aa6ef507bf9" />


### Loan Overview Dashboard
<img width="1261" height="701" alt="Overview" src="https://github.com/user-attachments/assets/edc58da5-33ff-4750-b90d-81b0083e6e4b" />


### Predictive Analysis Dashboard
<img width="1254" height="699" alt="Prediction" src="https://github.com/user-attachments/assets/7aed1846-5537-4637-9984-4737974a3962" />

