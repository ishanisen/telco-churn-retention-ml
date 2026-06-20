import os
from preprocess import load_data, clean_data
from features import create_features

def save_processed_dataset(
    input_path="data/raw/telco_churn.csv",
    output_path="data/processed/telco_features.csv"
):
    # Load raw data
    df = load_data(input_path)

    # Clean raw data
    df = clean_data(df)

    # Create engineered features
    df = create_features(df)

    # Make sure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save processed dataset
    df.to_csv(output_path, index=False)

    print(f"Processed dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print("\nNew engineered columns:")
    print([
        "tenure_bucket",
        "service_count",
        "is_month_to_month",
        "uses_electronic_check",
        "charges_per_tenure"
    ])

if __name__ == "__main__":
    save_processed_dataset()