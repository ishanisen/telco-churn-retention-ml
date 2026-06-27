import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(filepath):
    """Load the Telco Customer Churn dataset."""
    return pd.read_csv(filepath)


def clean_data(df):
    df = df.copy()
    #converting all string values to numeric values
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce')

    service_cols = [
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies']

    for col in service_cols:
        df[col] = df[col].replace({'No internet service': 'No'})
        
    return df

def build_preprocessor():
    """Build ColumnTransformer for numerical imputation, categorical imputation, one-hot encoding, scaling"""

    numeric_features = [
        'tenure',
        'MonthlyCharges',
        'TotalCharges',
        'service_count',
        'charges_per_tenure',
        'is_month_to_month',
        'is_long_contract',
        'uses_electronic_check',
        'has_security_support',
        'high_monthly_charges',
        'SeniorCitizen'
    ]

    categorical_features = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod', 'InternetService', 'tenure_bucket'
    ]

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    return preprocessor


def split_data(df, target_col='Churn'):
    """Split data into train (80%), validation (10%), test (10%) with stratification"""
    X = df.drop(columns=['Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # First split: 80% train, 20% val+test
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  
    )
    
    # Second split: 50/50 → 10% val, 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.5, random_state=42, stratify=y_val_test 
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


    
    
    