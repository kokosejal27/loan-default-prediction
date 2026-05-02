Loan Default Prediction System for Risk Analysis

This project focuses on building a machine learning-based system to predict the likelihood of loan default, enabling financial institutions to make data-driven lending decisions and minimize risk.

The system is developed using Python, leveraging libraries such as Pandas and NumPy for data preprocessing and analysis, and Scikit-learn for model building. Extensive data cleaning and feature engineering techniques are applied to extract meaningful insights from financial data, including the creation of derived features like income-to-loan ratio and interest burden, which improve model performance.

A Random Forest Classifier is used for prediction, along with a custom threshold tuning approach based on precision-recall trade-offs. This allows better control over classification performance, especially in handling imbalanced data and improving the identification of high-risk borrowers.

To make the solution interactive and user-friendly, the model is deployed using Streamlit. The web application allows users to input applicant details and receive real-time predictions, including default probability, risk classification, and interpretability insights.

Additionally, the predictive outputs are integrated into a Power BI dashboard to enhance business intelligence capabilities. This enables better visualization of trends, supports decision-making, and provides a comprehensive view of loan risk analysis.
