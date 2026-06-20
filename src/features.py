import pandas as pd
import numpy as np

def create_features(df):
    """Create engineered features for Telco churn prediction."""
    df = df.copy()

    # 1. Tenure buckets
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["new", "mid_term", "long_term", "loyal"]
    )

    # 2. Count subscribed services
    service_cols = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["service_count"] = df[service_cols].apply(
        lambda row: (row == "Yes").sum(),
        axis=1
    )

    # 3. Contract risk flags
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_long_contract"] = df["Contract"].isin(["One year", "Two year"]).astype(int)

    # 4. Payment risk flag
    df["uses_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    # 5. Support / security flags
    df["has_security_support"] = (
        ((df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes"))
    ).astype(int)

    # 6. Charges per month of tenure
    df["charges_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

    # 7. High monthly charge flag
    monthly_charge_threshold = df["MonthlyCharges"].median()
    df["high_monthly_charges"] = (df["MonthlyCharges"] > monthly_charge_threshold).astype(int)

    return df