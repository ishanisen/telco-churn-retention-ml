import streamlit as st
import pandas as pd
import joblib

from features import create_features

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📉",
    layout="wide"
)


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("models/preprocessor.pkl")
    model = joblib.load("models/xgboost_churn_model.pkl")
    config = joblib.load("models/model_config.pkl")
    shap_importance = pd.read_csv("reports/top_shap_features.csv")
    return preprocessor, model, config, shap_importance


def clean_driver_name(name):
    name = name.replace("num__", "")
    name = name.replace("cat__", "")
    name = name.replace("_", " ")
    return name.title()


def get_risk_label(probability):
    if probability < 0.30:
        return "Low", "🟢"
    elif probability < 0.60:
        return "Medium", "🟠"
    return "High", "🔴"


def get_retention_action(probability, contract, monthly_charges):
    if probability >= 0.60:
        if contract == "Month-to-month":
            return "Offer a contract incentive and prioritize a retention call."
        elif monthly_charges >= 80:
            return "Offer a pricing or bundle discount and assign follow-up outreach."
        return "Prioritize proactive outreach from the retention team."
    elif probability >= 0.30:
        return "Send a targeted retention message and monitor recent activity."
    return "No immediate action needed; continue regular monitoring."


preprocessor, model, config, shap_importance = load_artifacts()
threshold = config["threshold"]

st.title("Telco Customer Churn Prediction")
st.write(
    "Enter one customer's information to estimate churn risk, view the risk level, "
    "see the top churn drivers, and get a suggested retention action."
)

st.markdown("---")

with st.form("churn_form"):
    st.subheader("Customer Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("SeniorCitizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

    with col2:
        st.markdown("### Services")
        phone_service = st.selectbox("PhoneService", ["Yes", "No"])
        multiple_lines = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])

    with col3:
        st.markdown("### Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("PaperlessBilling", ["Yes", "No"])
        payment_method = st.selectbox(
            "PaymentMethod",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "MonthlyCharges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )
        total_charges = st.number_input(
            "TotalCharges",
            min_value=0.0,
            max_value=10000.0,
            value=1000.0
        )

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    input_df = create_features(input_df)
    input_processed = preprocessor.transform(input_df)

    churn_probability = model.predict_proba(input_processed)[:, 1][0]
    predicted_class = int(churn_probability >= threshold)

    risk_label, risk_icon = get_risk_label(churn_probability)
    suggested_action = get_retention_action(
        churn_probability,
        contract,
        monthly_charges
    )

    top_3_drivers = shap_importance.head(3)["feature"].apply(clean_driver_name).tolist()

    st.markdown("---")
    st.subheader("Prediction Result")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Churn Probability", f"{churn_probability:.2%}")
    with m2:
        st.metric("Predicted Class", "Churn" if predicted_class == 1 else "No Churn")
    with m3:
        st.metric("Risk Level", f"{risk_icon} {risk_label}")

    st.subheader("Top Drivers")
    for driver in top_3_drivers:
        st.write(f"- {driver}")

    st.subheader("Suggested Action")
    st.info(suggested_action)

    st.subheader("Model Threshold")
    st.write(f"The current operating threshold for churn classification is `{threshold}`.")