
import os
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

MODEL_PATH = "models/churn_pipeline.joblib"
DATA_PATH = "data/telco_churn.csv"

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Based Customer Churn Prediction")
st.caption(
    "Predict customer churn risk and understand the factors "
    "associated with customer attrition."
)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


# ---------------- LOAD DATASET ----------------

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH)

    if "Churn Value" in df.columns:
        churn_values = pd.to_numeric(
            df["Churn Value"], errors="coerce"
        )
        df["Churn"] = churn_values.map({
            0: "No",
            1: "Yes"
        })

    elif "Churn Label" in df.columns:
        df["Churn"] = df["Churn Label"]

    return df


model = load_model()
data = load_data()

if model is None:
    st.warning(
        "Model not found. Run `python train_model.py` first."
    )
    st.stop()


# ---------------- APPLICATION TABS ----------------

tab1, tab2, tab3 = st.tabs([
    "Predict a customer",
    "Churn analytics",
    "About the model"
])


# ==================================================
# TAB 1: CUSTOMER CHURN PREDICTION
# ==================================================

with tab1:

    st.subheader("Customer risk assessment")

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        senior = st.selectbox(
            "Senior citizen", ["No", "Yes"]
        )

        partner = st.selectbox(
            "Has partner?", ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Has dependents?", ["Yes", "No"]
        )

    with c2:
        tenure = st.slider(
            "Tenure (months)", 0, 72, 12
        )

        phone = st.selectbox(
            "Phone service", ["Yes", "No"]
        )

        multiple = st.selectbox(
            "Multiple lines",
            ["No", "Yes", "No phone service"]
        )

        internet = st.selectbox(
            "Internet service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online security",
            ["Yes", "No", "No internet service"]
        )

    with c3:
        online_backup = st.selectbox(
            "Online backup",
            ["Yes", "No", "No internet service"]
        )

        device = st.selectbox(
            "Device protection",
            ["Yes", "No", "No internet service"]
        )

        tech = st.selectbox(
            "Tech support",
            ["Yes", "No", "No internet service"]
        )

        streaming = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        movies = st.selectbox(
            "Streaming movies",
            ["Yes", "No", "No internet service"]
        )

    c4, c5, c6 = st.columns(3)

    with c4:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless = st.selectbox(
            "Paperless billing", ["Yes", "No"]
        )

    with c5:
        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly charges ($)",
            min_value=0.0,
            max_value=300.0,
            value=70.0
        )

    with c6:
        total = st.number_input(
            "Total charges ($)",
            min_value=0.0,
            max_value=20000.0,
            value=float(monthly * tenure)
        )

    # PREDICTION BUTTON

    if st.button("Predict churn risk", type="primary"):

        # IMPORTANT: Column names now match train_model.py

        customer = pd.DataFrame([{
            "Gender": gender,
            "Senior Citizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure Months": tenure,
            "Phone Service": phone,
            "Multiple Lines": multiple,
            "Internet Service": internet,
            "Online Security": online_security,
            "Online Backup": online_backup,
            "Device Protection": device,
            "Tech Support": tech,
            "Streaming TV": streaming,
            "Streaming Movies": movies,
            "Contract": contract,
            "Paperless Billing": paperless,
            "Payment Method": payment,
            "Monthly Charges": monthly,
            "Total Charges": total
        }])

        try:
            # Match the exact training feature order

            expected_columns = list(model.feature_names_in_)

            missing_columns = [
                col for col in expected_columns
                if col not in customer.columns
            ]

            if missing_columns:
                st.error(
                    f"Missing input columns: {missing_columns}"
                )
                st.stop()

            customer = customer[expected_columns]

            # Predict probability

            probabilities = model.predict_proba(customer)[0]

            classes = list(model.classes_)
            churn_index = classes.index(1)

            probability = float(
                probabilities[churn_index]
            )

            prediction = int(probability >= 0.5)

            st.metric(
                "Estimated churn probability",
                f"{probability:.1%}"
            )

            if prediction == 1:

                st.error(
                    "High churn classification: "
                    "This customer is predicted to churn."
                )

                st.write(
                    "Suggested action: Review service issues, "
                    "assess plan fit, and consider a relevant "
                    "retention offer."
                )

            else:

                st.success(
                    "Low churn classification: "
                    "This customer is predicted to stay."
                )

                st.write(
                    "Suggested action: Maintain service quality "
                    "and monitor customer engagement."
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ==================================================
# TAB 2: CHURN ANALYTICS
# ==================================================

with tab2:

    st.subheader("Dataset overview")

    if data is None:

        st.info(
            "Add the Telco Customer Churn CSV at "
            "`data/telco_churn.csv` to enable analytics."
        )

    elif "Churn" not in data.columns:

        st.error(
            "The dataset must contain a Churn column."
        )

    else:

        d = data.copy()

        churn_counts = (
            d["Churn"]
            .dropna()
            .astype(str)
            .value_counts()
            .rename_axis("Customer status")
            .reset_index(name="Customers")
        )

        a, b = st.columns(2)

        with a:

            st.plotly_chart(
                px.pie(
                    churn_counts,
                    names="Customer status",
                    values="Customers",
                    title="Customer churn distribution",
                    hole=0.45
                ),
                width="stretch"
            )

        with b:

            if "Contract" in d.columns:

                contract_churn = (
                    d.groupby("Contract", dropna=False)["Churn"]
                    .apply(
                        lambda s: (
                            s.astype(str)
                            .str.lower()
                            .isin(["yes", "1", "true"])
                        ).mean()
                    )
                    .reset_index(name="Churn rate")
                )

                st.plotly_chart(
                    px.bar(
                        contract_churn,
                        x="Contract",
                        y="Churn rate",
                        title="Churn rate by contract type",
                        text_auto=".1%"
                    ),
                    width="stretch"
                )

        st.subheader("Customer dataset")

        st.dataframe(
            d.head(20),
            width="stretch"
        )

        st.caption(
            f"Rows: {len(d):,} | Columns: {len(d.columns)}"
        )


# ==================================================
# TAB 3: MODEL INFORMATION
# ==================================================

with tab3:

    st.subheader("Model information")

    st.write("""
    - **Task:** Binary classification (Churn = Yes / No)
    - **Default algorithm:** Random Forest classifier
    - **Preprocessing:** Median imputation and scaling for
      numeric variables; most-frequent imputation and one-hot
      encoding for categorical variables.
    - **Evaluation:** Hold-out test set with accuracy,
      precision, recall, F1-score and ROC-AUC.
    - **Decision threshold:** 0.50 by default.

    Model predictions are estimates, not guarantees.
    Validate performance on recent, representative customer
    data before using predictions for business decisions.
    """)

    # MODEL EVALUATION METRICS

    metrics_path = "models/metrics.json"

    if os.path.exists(metrics_path):

        with open(metrics_path) as f:
            metrics = json.load(f)

        st.subheader("Model evaluation metrics")

        metric_cols = st.columns(5)

        metric_names = [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1 Score", "f1_score"),
            ("ROC-AUC", "roc_auc")
        ]

        for col, (label, key) in zip(
            metric_cols, metric_names
        ):

            with col:
                st.metric(
                    label,
                    f"{metrics[key]:.2%}"
                )

        st.subheader("Confusion matrix")

        if "confusion_matrix" in metrics:
            st.write(
                pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=["Actual No", "Actual Yes"],
                    columns=["Predicted No", "Predicted Yes"]
                )
            )

    # FEATURE IMPORTANCE

    importance_path = "models/feature_importance.csv"

    if os.path.exists(importance_path):

        importance = pd.read_csv(
            importance_path
        ).head(15).sort_values("importance")

        st.subheader(
            "Top factors associated with churn"
        )

        st.plotly_chart(
            px.bar(
                importance,
                x="importance",
                y="feature",
                orientation="h",
                title="Random Forest feature importance"
            ),
            width="stretch"
        )

        st.caption(
            "Feature importance shows which inputs the fitted "
            "model relied on most; it does not prove causation."
        )
