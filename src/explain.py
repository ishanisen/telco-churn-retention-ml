from preprocess import load_data, clean_data, split_data
from features import create_features

from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap


print("=== LOADING DATA ===")
df = load_data("data/raw/telco_churn.csv")

print("=== CLEANING DATA ===")
df = clean_data(df)

print("=== CREATING FEATURES ===")
df = create_features(df)

print("=== SPLITTING DATA ===")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

print("=== LOADING ARTIFACTS ===")
preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/xgboost_churn_model.pkl")

print("=== TRANSFORMING TEST DATA ===")
X_test_processed = preprocessor.transform(X_test)

print("=== GETTING FEATURE NAMES ===")
feature_names = preprocessor.get_feature_names_out()
X_test_processed_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)

reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)

print("=== SAVING XGBOOST FEATURE IMPORTANCE PLOT ===")
importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

top_n = 15
plt.figure(figsize=(10, 6))
plt.barh(
    importance["feature"].head(top_n)[::-1],
    importance["importance"].head(top_n)[::-1]
)
plt.xlabel("XGBoost Feature Importance")
plt.title("Top 15 XGBoost Feature Importances")
plt.tight_layout()
plt.savefig(reports_dir / "xgboost_feature_importance.png", dpi=300)
plt.close()

importance.to_csv(reports_dir / "xgboost_feature_importance.csv", index=False)

print("=== COMPUTING SHAP VALUES ===")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_processed_df)

print("=== SAVING SHAP BAR PLOT ===")
plt.figure()
shap.summary_plot(
    shap_values,
    X_test_processed_df,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(reports_dir / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
plt.close()

print("=== SAVING SHAP SUMMARY PLOT ===")
plt.figure()
shap.summary_plot(
    shap_values,
    X_test_processed_df,
    show=False
)
plt.tight_layout()
plt.savefig(reports_dir / "shap_summary_plot.png", dpi=300, bbox_inches="tight")
plt.close()

print("=== SAVING TOP SHAP FEATURES ===")
mean_abs_shap = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

mean_abs_shap.to_csv(reports_dir / "top_shap_features.csv", index=False)

print(mean_abs_shap.head(15))
print("Saved outputs to reports/")