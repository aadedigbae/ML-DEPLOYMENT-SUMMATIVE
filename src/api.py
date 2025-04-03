from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import pickle

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load the trained model
MODEL_PATH = os.path.join(BASE_DIR, "models/model.pkl")
model = joblib.load(MODEL_PATH)

# Load preprocessing objects (encoder and scaler)
ENCODER_PATH = os.path.join(BASE_DIR, "models/encoder.pkl")  # One-hot or label encoder
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")  # StandardScaler or MinMaxScaler

encoder = joblib.load(ENCODER_PATH)

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
scaler = joblib.load(SCALER_PATH)

# Load top feature columns
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "models/feature_columns.pkl")
FEATURE_COLUMNS = joblib.load(FEATURE_COLUMNS_PATH)

# All features are numerical
numerical_features = FEATURE_COLUMNS

print("Top features used in the API:", FEATURE_COLUMNS)

@app.route('/predict', methods=['POST'])
def predict():
    """Accepts JSON input and returns churn prediction"""
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Debug: Print input data and expected features
        print("Input data:", df)
        print("Expected features:", FEATURE_COLUMNS)

        # Ensure all features exist in the dataframe
        df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

        # Scale numerical features
        df[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])

        # Make prediction
        prediction = model.predict(df)[0]

        return jsonify({'Churn Prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
