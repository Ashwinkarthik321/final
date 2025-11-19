Credit Risk Model (LightGBM + SHAP)
1. Overview

This project builds a model that predicts whether a loan will default.
It uses LightGBM for prediction and SHAP to explain how the model makes decisions.
The goal is to have a model that is both accurate and easy to understand.

2. Steps

Load the dataset (or create one if not found).

Split data into Train, Validation, and Test sets.

Try different model settings and choose the best one.

Train the final model using the best settings.

Evaluate the model using AUC, F1, Precision, and Recall.

Use SHAP to:

Find important features

Create summary and dependence plots

Explain high-risk and low-risk loan predictions

Save all results, plots, explanations, and reports.

3. How to Run

Install required packages:

pip install numpy pandas scikit-learn lightgbm shap matplotlib joblib


Run the script:

python credit_risk_model_shap.py


After running, check the folder:

shap_credit_out/


This folder contains:

Model file

Evaluation metrics

SHAP plots

Feature importance

Local explanations

Reports

4. Interpretation

SHAP summary plot shows which features most affect loan default.

Dependence plots show how risk changes as a feature increases or decreases.

Local SHAP explanations show why the model rated a specific loan as high-risk or low-risk.

Positive SHAP values = higher risk

Negative SHAP values = lower risk

This helps understand both overall model behavior and individual loan decisions.
