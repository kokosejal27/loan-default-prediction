#%%
# ================================
# IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve
)

#%%
# ================================
# LOAD DATA
# ================================
df = pd.read_csv(r"df = pd.read_csv("data/financial_loan.csv")")

#%%
# ================================
# TARGET CLEANING
# ================================
df = df[df['loan_status'] != 'Current']

df['loan_status'] = df['loan_status'].map({
    'Fully Paid': 0,
    'Charged Off': 1
})

#%%
# ================================
# DATE FEATURES
# ================================
df['issue_date'] = pd.to_datetime(df['issue_date'], errors='coerce')
df['issue_year'] = df['issue_date'].dt.year
df['issue_month'] = df['issue_date'].dt.month

#%%
# ================================
# SELECT FEATURES
# ================================
df = df[[
    'annual_income',
    'loan_amount',
    'installment',
    'int_rate',
    'dti',
    'total_acc',
    'issue_year',
    'issue_month',
    'loan_status'
]]

#%%
# ================================
# FEATURE ENGINEERING
# ================================
df['income_loan_ratio'] = (df['annual_income'] / df['loan_amount']) * 2
df['interest_burden'] = df['installment'] / df['annual_income']
df['loan_to_income'] = df['loan_amount'] / df['annual_income']

#%%
# ================================
# CLEAN DATA
# ================================
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.fillna(df.median(numeric_only=True))

#%%
# ================================
# HEATMAP (EDA)
# ================================
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#%%
# ================================
# SPLIT DATA
# ================================
X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#%%
# ================================
# MODEL TRAINING
# ================================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

#%%
# ================================
# PROBABILITY
# ================================
y_prob = model.predict_proba(X_test)[:, 1]

#%%
# ================================
# BEST THRESHOLD
# ================================
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]

print("Best Threshold:", best_threshold)

#%%
# ================================
# FINAL PREDICTIONS
# ================================
y_pred = (y_prob > best_threshold).astype(int)

#%%
# ================================
# EVALUATION
# ================================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#%%
# ================================
# FEATURE IMPORTANCE
# ================================
importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importance:\n")
print(importance.sort_values(ascending=False))

#%%
# ================================
# SAVE MODEL
# ================================
pickle.dump(model, open("loan_model.pkl", "wb"))
pickle.dump(best_threshold, open("threshold.pkl", "wb"))

print("\nModel saved successfully")