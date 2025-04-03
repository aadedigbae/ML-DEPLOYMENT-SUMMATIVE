import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Absolute paths
MODEL_PATH = os.path.join(BASE_DIR, "models/model.pkl")
CLEANED_TRAIN_PATH = os.path.join(BASE_DIR, "data/train/cleaned_train.csv")
CLEANED_TEST_PATH = os.path.join(BASE_DIR, "data/test/cleaned_test.csv")

# Load model
model = joblib.load(MODEL_PATH)

# Load data
train_df = pd.read_csv(CLEANED_TRAIN_PATH)
test_df = pd.read_csv(CLEANED_TEST_PATH)

# Extract features and target
X_train, y_train = train_df.drop(columns=['Churn']), train_df['Churn']
X_test, y_test = test_df.drop(columns=['Churn']), test_df['Churn']

# Predict
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Compute evaluation metrics
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
