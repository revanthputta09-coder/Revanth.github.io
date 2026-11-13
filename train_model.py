
import json
import os
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "models/churn_pipeline.joblib"


def main():

    # Load dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Dataset not found. Save the IBM Telco Customer Churn CSV "
            "as data/telco_churn.csv."
        )

    df = pd.read_csv(DATA_PATH)

    # Dataset cleanup
    if "Churn Value" not in df.columns:
        raise ValueError("Expected target column 'Churn Value'.")

    # Remove customer ID
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])
    # Remove unnecessary geographic and administrative columns
    unused_columns = [
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "CLTV"
]

    df = df.drop(
    columns=[c for c in unused_columns if c in df.columns]
)

    # Convert Total Charges into numeric format
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(
            df["Total Charges"], errors="coerce"
        )

    # Select target column
    y = pd.to_numeric(df["Churn Value"], errors="coerce")

    # Remove invalid target rows
    valid = y.notna()

    leakage_columns = [
        "Churn Value",
        "Churn Label",
        "Churn Score",
        "Churn Reason"
    ]

    X = df.loc[valid].drop(
        columns=[c for c in leakage_columns if c in df.columns]
    )

    y = y.loc[valid].astype(int)

    # Identify numeric and categorical columns
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object","str", "category", "bool"]).columns.tolist()

    # Numeric preprocessing
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Preprocessing pipeline
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])

    # Random Forest model pipeline
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    # Evaluate model
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(
            y_test,
            pred,
            output_dict=True,
            zero_division=0
        )
    }

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Save trained model
    joblib.dump(pipeline, MODEL_PATH)

    # Save evaluation metrics
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(
        "models/feature_importance.csv",
        index=False
    )

    # Display results
    print("Model saved to", MODEL_PATH)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
