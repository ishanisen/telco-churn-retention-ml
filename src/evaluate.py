from preprocess import load_data, clean_data, split_data, build_preprocessor
from features import create_features

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