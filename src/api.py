from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import sys
import os
from werkzeug.utils import secure_filename


# Add the project directory to the Python module search path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)


from src.model import train_model, evaluate_model, save_artifacts
from src.preprocessing import preprocess_data, scale_and_balance_data

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths for model and preprocessing artifacts
MODEL_PATH = os.path.join(BASE_DIR, "models/churn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "models/feature_columns.pkl")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data/uploads")

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model and preprocessing artifacts
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

print("API initialized with the following features:", feature_columns)

# Route to serve frontend
@app.route('/')
def home():
    return render_template('index.html', features=feature_columns)

@app.route('/predict', methods=['POST'])
def predict():
    """Accepts JSON input and returns churn prediction."""
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

@app.route('/upload', methods=['POST'])
def upload_data():
    """Allows users to upload bulk data (CSV file)."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        return jsonify({'message': f'File uploaded successfully: {filename}', 'file_path': file_path})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Triggers the retraining process using the uploaded data."""
    try:
        # Get the file path from the request
        file_path = request.json.get('file_path')
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Invalid or missing file path'}), 400

        # Load the uploaded data
        df = pd.read_csv(file_path)

        # Define target column and leaking features
        target_column = "Churn"
        leaking_features = [
            'Churn Score', 'Satisfaction Score', 'Contract', 'Churn Reason',
            'Churn Category', 'Customer Status', 'Country', 'Quarter', 'State'
        ]

        # Preprocess the data
        X, y = preprocess_data(df, target_column, leaking_features)
        X_balanced, y_balanced, scaler = scale_and_balance_data(X, y)

        # Convert X_balanced back to a DataFrame
        X_balanced = pd.DataFrame(X_balanced, columns=X.columns)

        # Retrain the model
        new_model = train_model(X_balanced, y_balanced)

        # Save the updated artifacts
        save_artifacts(new_model, scaler, X_balanced.columns.tolist())

        return jsonify({'message': 'Model retrained and artifacts updated successfully'})

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