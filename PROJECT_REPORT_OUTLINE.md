# Project Report: AI-Based Customer Churn Prediction

## Abstract
Customer churn is a significant business problem because losing existing customers can reduce recurring revenue and increase acquisition costs. This project develops a binary classification system to estimate whether a customer is likely to discontinue a service. Customer demographic, account, contract, billing and service attributes are used as model inputs. The workflow includes data cleaning, preprocessing, model training and evaluation. A Streamlit dashboard provides customer-level risk predictions and exploratory churn analytics. The model is intended to support customer-retention decisions, not replace business judgement.

## Problem Statement
Businesses often identify customer dissatisfaction only after a customer has already discontinued a service. A predictive system can help identify customers with elevated churn risk earlier, allowing teams to investigate service problems and plan appropriate retention actions.

## Objectives
1. Build a binary classification model for customer churn.
2. Prepare customer data through cleaning and preprocessing.
3. Evaluate the model using suitable classification metrics.
4. Identify variables associated with churn.
5. Present predictions and churn patterns through an interactive dashboard.

## Methodology
1. Dataset acquisition and inspection.
2. Data cleaning and missing-value handling.
3. Encoding categorical variables and scaling numerical variables.
4. Stratified train-test split.
5. Random Forest model training.
6. Evaluation using accuracy, precision, recall, F1-score, ROC-AUC and confusion matrix.
7. Streamlit dashboard development.

## Expected Outcome
The application returns a churn probability and a binary prediction for an entered customer profile. It also displays churn distribution and churn rate by contract type when the source dataset is available.

## Future Enhancements
- Compare Logistic Regression, Decision Tree, Random Forest and Gradient Boosting.
- Tune the classification threshold using business retention costs.
- Add SHAP-based local explanations.
- Add batch scoring for uploaded customer files.
- Monitor model drift and retrain using recent customer data.
