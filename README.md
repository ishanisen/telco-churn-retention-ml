## Project Overview

This project builds an end-to-end customer churn prediction pipeline using the IBM Telco Customer Churn dataset. The main objective is to identify customers who are most likely to leave and translate those predictions into retention-focused business insights. This project is designed to demonstrate practical machine learning skills including data cleaning, preprocessing, feature engineering, model evaluation, and business interpretation.

## Dataset

The dataset used in this project is the IBM Telco Customer Churn dataset. Each row represents a telecom customer, and the columns include demographic information, subscribed services, billing details, contract type, payment method, and the target variable `Churn`. This dataset is well suited for a supervised classification problem because it contains a realistic mix of numerical and categorical features along with a business-relevant prediction target.

## Preprocessing

The preprocessing pipeline is designed to make the dataset clean, reproducible, and ready for machine learning. The raw data is first cleaned by fixing data types, converting numeric-like string columns into proper numeric format, and standardizing inconsistent service labels such as replacing `"No internet service"` with `"No"`. After cleaning, the data is split into train, validation, and test sets using stratified sampling so the churn class balance is preserved across all splits.
For model-ready preprocessing, a `ColumnTransformer` is used to handle mixed feature types. Numerical columns are imputed using the mean, categorical columns are imputed using the most frequent value, categorical features are one-hot encoded, and scaling can be applied for linear models when needed. This setup makes the preprocessing pipeline reusable and reduces leakage risk by learning transformations only from the training data.

## Engineered Features

The following engineered features were created to better capture customer lifecycle, service adoption, contract risk, and billing behavior.

| Feature | Type | Description | Why it may help |
|---|---|---|---|
| `tenure_bucket` | Categorical | Groups customers by tenure into lifecycle stages such as new, mid-term, long-term, and loyal. | Newer customers often behave differently from long-term customers and may have higher churn risk. |
| `service_count` | Numerical | Counts the number of subscribed add-on services, excluding basic `PhoneService`. | Customers with more bundled services may show different retention behavior than customers with fewer services. |
| `is_month_to_month` | Binary | Indicates whether the customer is on a month-to-month contract. | Flexible contracts are often associated with higher churn risk. |
| `is_long_contract` | Binary | Indicates whether the customer has a one-year or two-year contract. | Longer contracts can signal stronger retention and lower churn risk. |
| `uses_electronic_check` | Binary | Indicates whether the customer uses electronic check as payment method. | Certain payment methods may correlate with weaker retention behavior. |
| `has_security_support` | Binary | Indicates whether the customer has either `OnlineSecurity` or `TechSupport`. | Security and support services may reflect stronger engagement and lower churn risk. |
| `charges_per_tenure` | Numerical | Approximates billing intensity by dividing `TotalCharges` by `tenure + 1`. | Captures the relationship between billing history and customer age. |
| `high_monthly_charges` | Binary | Flags customers whose monthly charges are above the dataset median. | Higher monthly charges can be associated with churn in telecom datasets. |