from preprocess import load_data, clean_data, split_data, build_preprocessor
from features import create_features

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)
from pathlib import Path
import joblib

def evaluate_model(model_name, y_true, y_pred, y_proba):
    print(f"\n=== {model_name} VALIDATION RESULTS ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("ROC-AUC:", roc_auc_score(y_true, y_proba))
    print("PR-AUC:", average_precision_score(y_true, y_proba))


def apply_threshold(y_proba, threshold):
    return (y_proba >= threshold).astype(int)


def evaluate_thresholds(model_name, y_true, y_proba, thresholds=[0.3, 0.4, 0.5, 0.6]):
    rows = []

    print(f"\n=== THRESHOLD TESTS FOR {model_name.upper()} ===")

    for threshold in thresholds:
        y_pred_thresh = apply_threshold(y_proba, threshold)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        rows.append({
            "model": model_name,
            "threshold": threshold,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    return results


print("=== LOADING DATA ===")
df = load_data("data/raw/telco_churn.csv")
print("Raw data shape:", df.shape)

print("\n=== CLEANING DATA ===")
df = clean_data(df)
print("Cleaned data shape:", df.shape)

print("\n=== CREATING FEATURES ===")
df = create_features(df)
print("Feature-engineered data shape:", df.shape)
print("Sample engineered columns:")
print(df[[
    "tenure_bucket",
    "service_count",
    "is_month_to_month",
    "uses_electronic_check",
    "charges_per_tenure"
]].head())

print("\n=== SPLITTING DATA ===")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)

print("\n=== BUILDING PREPROCESSOR ===")
preprocessor = build_preprocessor()

print("\n=== TRANSFORMING DATA ===")
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)
print("Processed X_train shape:", X_train_processed.shape)
print("Processed X_val shape:", X_val_processed.shape)
print("Processed X_test shape:", X_test_processed.shape)

# ---------------- LOGISTIC REGRESSION ----------------
print("\n=== TRAINING LOGISTIC REGRESSION ===")
log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train_processed, y_train)
print("Logistic Regression trained successfully.")

log_y_val_pred = log_model.predict(X_val_processed)
log_y_val_proba = log_model.predict_proba(X_val_processed)[:, 1]

evaluate_model(
    model_name="LOGISTIC REGRESSION",
    y_true=y_val,
    y_pred=log_y_val_pred,
    y_proba=log_y_val_proba
)

log_threshold_results = evaluate_thresholds(
    model_name="Logistic Regression",
    y_true=y_val,
    y_proba=log_y_val_proba
)

# ---------------- RANDOM FOREST ----------------
print("\n=== TRAINING RANDOM FOREST ===")
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
rf_model.fit(X_train_processed, y_train)
print("Random Forest trained successfully.")

rf_y_val_pred = rf_model.predict(X_val_processed)
rf_y_val_proba = rf_model.predict_proba(X_val_processed)[:, 1]

evaluate_model(
    model_name="RANDOM FOREST",
    y_true=y_val,
    y_pred=rf_y_val_pred,
    y_proba=rf_y_val_proba
)

rf_threshold_results = evaluate_thresholds(
    model_name="Random Forest",
    y_true=y_val,
    y_proba=rf_y_val_proba
)

# ---------------- XGBOOST ----------------
print("\n=== TRAINING XGBOOST ===")
xgb_model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)
xgb_model.fit(X_train_processed, y_train)
print("XGBoost trained successfully.")

xgb_y_val_pred = xgb_model.predict(X_val_processed)
xgb_y_val_proba = xgb_model.predict_proba(X_val_processed)[:, 1]

evaluate_model(
    model_name="XGBOOST",
    y_true=y_val,
    y_pred=xgb_y_val_pred,
    y_proba=xgb_y_val_proba
)

xgb_threshold_results = evaluate_thresholds(
    model_name="XGBoost",
    y_true=y_val,
    y_proba=xgb_y_val_proba
)

# ---------------- SAVE THRESHOLD RESULTS ----------------
all_threshold_results = pd.concat(
    [log_threshold_results, rf_threshold_results, xgb_threshold_results],
    ignore_index=True
)

all_threshold_results.to_csv("threshold_results.csv", index=False)
print("\nSaved threshold results to threshold_results.csv")

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

joblib.dump(preprocessor, models_dir / "preprocessor.pkl")
joblib.dump(xgb_model, models_dir / "xgboost_churn_model.pkl")
joblib.dump({"threshold": 0.4}, models_dir / "model_config.pkl")

print("Saved preprocessor to models/preprocessor.pkl")
print("Saved final model to models/xgboost_churn_model.pkl")
print("Saved model config to models/model_config.pkl")