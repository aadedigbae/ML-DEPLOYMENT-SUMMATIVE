import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_PATH = os.path.join(BASE_DIR, "data/train/churn_train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data/test/churn_test.csv")
ENCODER_PATH = os.path.join(BASE_DIR, "models/encoder.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")
CLEANED_TRAIN_PATH = os.path.join(BASE_DIR, "data/train/cleaned_train.csv")
CLEANED_TEST_PATH = os.path.join(BASE_DIR, "data/test/cleaned_test.csv")

def load_data(file_path):
    """Loads CSV data into a Pandas DataFrame."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Handles missing values by filling or dropping."""
    df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numeric missing values
    df.fillna("Unknown", inplace=True)  # Fill categorical missing values
    return df

def encode_data(df):
    """Encodes categorical features using Label Encoding."""
    encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = encoder.fit_transform(df[col])
    # Save the encoder
    joblib.dump(encoder, ENCODER_PATH)
    print(f"Encoder saved at {ENCODER_PATH}")
    return df

def scale_data(df, numerical_columns):
    """Scales numeric features using StandardScaler."""
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    # Save the scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved at {SCALER_PATH}")
    return df

def remove_leaking_features(df, leaking_features):
    """Removes leaking features from the dataset."""
    return df.drop(columns=leaking_features, errors='ignore')

def preprocess_data(df, target_column, leaking_features):
    """Preprocess the dataset by encoding, scaling, and removing leaking features."""
    df = clean_data(df)
    df = encode_data(df)
    if leaking_features:
        df = remove_leaking_features(df, leaking_features)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def scale_and_balance_data(X, y):
    """Scale numerical features and apply SMOTE for class balancing."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

    return X_balanced, y_balanced, scaler

if __name__ == "__main__":
    print("Starting data preprocessing...")  # Debugging statement

    # Define numerical columns and leaking features
    numerical_columns = [
        'Age', 'Avg Monthly GB Download', 'Avg Monthly Long Distance Charges',
        'Monthly Charge', 'Tenure in Months', 'Total Charges', 
        'Total Extra Data Charges', 'Total Long Distance Charges', 
        'Total Refunds', 'Total Revenue'
    ]
    leaking_features = [
        'Churn Score', 'Satisfaction Score', 'Contract', 'Churn Reason',
        'Churn Category', 'Customer Status', 'Country', 'Quarter', 'State'
    ]

    # Preprocess training data
    print(f"Preprocessing training data from: {TRAIN_PATH}")
    df_train = load_data(TRAIN_PATH)
    X_train, y_train = preprocess_data(df_train, target_column='Churn', leaking_features=leaking_features)
    X_train, y_train, scaler = scale_and_balance_data(X_train, y_train)
    print(f"Training data preprocessed. Features shape: {X_train.shape}, Target shape: {y_train.shape}")

    # Preprocess testing data
    print(f"Preprocessing testing data from: {TEST_PATH}")
    df_test = load_data(TEST_PATH)
    X_test, y_test = preprocess_data(df_test, target_column='Churn', leaking_features=leaking_features)
    X_test = scaler.transform(X_test)
    print(f"Testing data preprocessed. Features shape: {X_test.shape}, Target shape: {y_test.shape}")

    # Ensure output directories exist
    os.makedirs(os.path.join(BASE_DIR, "data/train"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data/test"), exist_ok=True)

    # Save cleaned data
    print("Saving cleaned training data...")
    pd.concat([X_train, y_train], axis=1).to_csv(CLEANED_TRAIN_PATH, index=False)
    print(f"Cleaned training data saved at {CLEANED_TRAIN_PATH}")

    print("Saving cleaned testing data...")
    pd.concat([X_test, y_test], axis=1).to_csv(CLEANED_TEST_PATH, index=False)
    print(f"Cleaned testing data saved at {CLEANED_TEST_PATH}")

    print("Data preprocessing completed. Cleaned files saved.")
