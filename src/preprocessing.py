import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_PATH = os.path.join(BASE_DIR, "data/train/churn_train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data/test/churn_test.csv")

# Load the dataset
def load_data(file_path):
    """Loads CSV data into a Pandas DataFrame."""
    return pd.read_csv(file_path)

# Handle missing values
def clean_data(df):
    """Handles missing values by filling or dropping."""
    df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numeric missing values
    df.fillna("Unknown", inplace=True)  # Fill categorical missing values
    return df

# Encode categorical variables
def encode_data(df):
    """Encodes categorical features using Label Encoding."""
    encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = encoder.fit_transform(df[col])
    return df

if __name__ == "__main__":
    print("Starting data preprocessing...")  # Debugging statement

    # Load datasets
    print(f"Loading training data from: {TRAIN_PATH}")
    train_df = load_data(TRAIN_PATH)
    print(f"Training data loaded. Shape: {train_df.shape}")

    print(f"Loading testing data from: {TEST_PATH}")
    test_df = load_data(TEST_PATH)
    print(f"Testing data loaded. Shape: {test_df.shape}")

    # Clean data
    print("Cleaning training data...")
    train_df = clean_data(train_df)
    print("Training data cleaned.")

    print("Cleaning testing data...")
    test_df = clean_data(test_df)
    print("Testing data cleaned.")

    # Encode categorical variables
    print("Encoding training data...")
    train_df = encode_data(train_df)
    print("Training data encoded.")

    print("Encoding testing data...")
    test_df = encode_data(test_df)
    print("Testing data encoded.")

    # Ensure output directories exist
    os.makedirs(os.path.join(BASE_DIR, "data/train"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data/test"), exist_ok=True)

    # Save cleaned data
    print("Saving cleaned training data...")
    train_df.to_csv(os.path.join(BASE_DIR, "data/train/cleaned_train.csv"), index=False)
    print("Cleaned training data saved.")

    print("Saving cleaned testing data...")
    test_df.to_csv(os.path.join(BASE_DIR, "data/test/cleaned_test.csv"), index=False)
    print("Cleaned testing data saved.")

    print("Data preprocessing completed. Cleaned files saved.")
