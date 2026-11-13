# AI-Based Customer Churn Prediction

A machine-learning classification project that predicts whether a telecom customer is likely to churn and provides a Streamlit interface for customer-level predictions and basic churn analytics.

## 1. Dataset

Use the IBM Telco Customer Churn dataset (commonly distributed as `WA_Fn-UseC_-Telco-Customer-Churn.csv`). Save or rename the CSV to:

`data/telco_churn.csv`

The dataset should contain a `Churn` target column with values `Yes` and `No`. The project is designed around the standard IBM Telco dataset fields.

## 2. Setup

Create and activate a virtual environment:

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows
```bash
python -m venv .venv
.venv\\Scripts\\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3. Train the model

```bash
python train_model.py
```

This creates:
- `models/churn_pipeline.joblib`
- `models/metrics.json`

The pipeline performs imputation, scaling, one-hot encoding, and Random Forest classification. A stratified 80/20 train-test split is used.

## 4. Run the dashboard

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## 5. Project workflow

1. Load customer data.
2. Clean `TotalCharges` and remove the customer ID.
3. Separate input features and churn target.
4. Split data into training and test sets.
5. Preprocess numeric and categorical features.
6. Train a Random Forest classifier.
7. Evaluate with accuracy, precision, recall, F1-score and ROC-AUC.
8. Use the saved model in the Streamlit dashboard.

## 6. Important evaluation note

Churn datasets are often imbalanced. Do not judge the model using accuracy alone. Review recall, precision, F1-score, ROC-AUC and the confusion matrix. False negatives represent customers who churn but were not flagged; false positives represent customers flagged as risky who would have stayed.

## 7. Limitations and responsible use

Predictions are probabilistic estimates, not certainty. Validate on recent data before deployment, monitor performance over time, and avoid using sensitive attributes or proxies in ways that create unfair customer treatment. Feature importance describes association in this model and does not establish causation.
