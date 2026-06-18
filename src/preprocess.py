import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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

def encode_columns(df):
    df = df.copy()

    yes_no_cols = [
        'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

    df = pd.get_dummies(
        df,
        columns=['Contract', 'PaymentMethod', 'InternetService'],
        drop_first=True
    )

    return df


def split_data(df, target_col='Churn'):
    """Split the data into train, validation, and test sets"""
    X = df.drop(columns=['Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    
    
    