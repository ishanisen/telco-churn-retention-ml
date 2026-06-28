from preprocess import load_data, clean_data, split_data, build_preprocessor
from features import create_features
import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)


def apply_threshold(y_proba, threshold=0.4):
    return (y_proba >= threshold).astype(int)


print("=== LOADING DATA ===")
df = load_data("data/raw/telco_churn.csv")

print("=== CLEANING DATA ===")
df = clean_data(df)

print("=== CREATING FEATURES ===")
df = create_features(df)

print("=== SPLITTING DATA ===")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

print("=== LOADING PREPROCESSOR ===")
preprocessor = joblib.load("models/preprocessor.pkl")

print("=== TRANSFORMING TEST DATA ===")
X_test_processed = preprocessor.transform(X_test)

print("=== LOADING MODEL ===")
model = joblib.load("models/xgboost_churn_model.pkl")

print("=== LOADING THRESHOLD ===")
config = joblib.load("models/model_config.pkl")
threshold = config["threshold"]

print("=== TEST EVALUATION ===")
y_test_proba = model.predict_proba(X_test_processed)[:, 1]
y_test_pred = apply_threshold(y_test_proba, threshold)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_test_proba))
print("PR-AUC:", average_precision_score(y_test, y_test_proba))

def simulate_retention_strategy(
    model_name,
    y_true,
    y_proba,
    thresholds=[0.3, 0.4, 0.5, 0.6],
    contact_cost=10,
    save_rate=0.25,
    recovered_value=200
):
    rows = []

    print(f"\n=== RETENTION SIMULATION FOR {model_name.upper()} ===")

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        customers_contacted = tp + fp
        campaign_cost = customers_contacted * contact_cost
        expected_saved_customers = tp * save_rate
        expected_recovered_value = expected_saved_customers * recovered_value
        net_value = expected_recovered_value - campaign_cost

        rows.append({
            "model": model_name,
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "customers_contacted": customers_contacted,
            "campaign_cost": campaign_cost,
            "expected_saved_customers": expected_saved_customers,
            "expected_recovered_value": expected_recovered_value,
            "net_value": net_value
        })

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    return results


retention_results = simulate_retention_strategy(
    model_name="XGBoost",
    y_true=y_test,
    y_proba=y_test_proba,
    thresholds=[0.3, 0.4, 0.5, 0.6],
    contact_cost=10,
    save_rate=0.25,
    recovered_value=200
)

retention_results.to_csv("retention_simulation_results.csv", index=False)
print("\nSaved retention simulation results to retention_simulation_results.csv")