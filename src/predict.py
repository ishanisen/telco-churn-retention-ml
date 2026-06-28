from preprocess import load_data, clean_data, split_data
from features import create_features

import pandas as pd
import joblib


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

print("=== TRANSFORMING DATA ===")
X_test_processed = preprocessor.transform(X_test)

print("=== LOADING MODEL AND THRESHOLD ===")
model = joblib.load("models/xgboost_churn_model.pkl")
config = joblib.load("models/model_config.pkl")
threshold = config["threshold"]

print("=== PREDICTING ===")
y_proba = model.predict_proba(X_test_processed)[:, 1]
y_pred = apply_threshold(y_proba, threshold)

results = X_test.copy()
results["actual_churn"] = y_test.values
results["predicted_probability"] = y_proba
results["predicted_churn"] = y_pred

print(results[["actual_churn", "predicted_probability", "predicted_churn"]].head(10))

results.to_csv("test_predictions.csv", index=False)
print("Saved predictions to test_predictions.csv")