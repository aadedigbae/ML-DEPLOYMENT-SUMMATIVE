from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load the trained model
MODEL_PATH = os.path.join(BASE_DIR, "models/churn_model.pkl")
model = joblib.load(MODEL_PATH)

# Load preprocessing objects (scaler and feature columns)
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "models/feature_columns.pkl")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
scaler = joblib.load(SCALER_PATH)

if not os.path.exists(FEATURE_COLUMNS_PATH):
    raise FileNotFoundError(f"Feature columns file not found at {FEATURE_COLUMNS_PATH}")
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

print("API initialized with the following features:", feature_columns)

# Route to serve frontend
@app.route('/')
def home():
    """Serve the frontend page."""
    return render_template('index.html', features=feature_columns)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON input and returns churn prediction.
    """
    try:
        # Parse input JSON
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Ensure all features exist
        df = df.reindex(columns=feature_columns, fill_value=0)

        # Scale numerical features
        df_scaled = scaler.transform(df)

        # Make prediction
        prediction = model.predict(df_scaled)[0]

        return jsonify({'Churn Prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

"""
from flask import Flask, request, jsonify, render_template
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
ENCODER_PATH = os.path.join(BASE_DIR, "models/encoder.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")

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

# Route to serve frontend
@app.route('/')
def home():
    return render_template('index.html', features=FEATURE_COLUMNS)

@app.route('/predict', methods=['POST'])
def predict():
    #Accepts JSON input and returns churn prediction
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Debug
        print("Input data:", df)
        print("Expected features:", FEATURE_COLUMNS)

        # Ensure all features exist
        df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

        # Debug: Check if all features are present
        print("Reindexed data:", df)

        # Scale numerical features
        df_scaled = pd.DataFrame(scaler.transform(df[FEATURE_COLUMNS]), columns=FEATURE_COLUMNS)

        # Convert to NumPy array to remove feature names
        df_array = df.to_numpy()

        # Predict
        prediction = model.predict(df_scaled)[0]

        return jsonify({'Churn Prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
"""