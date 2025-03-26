import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEANED_TRAIN_PATH = os.path.join(BASE_DIR, "data/train/cleaned_train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models/model.pkl")

# Load cleaned dataset
df = pd.read_csv(CLEANED_TRAIN_PATH)

# Split features and target
X = df.drop(columns=['Churn'])  # Replace 'Churn' with actual target column name
y = df['Churn']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"Model trained. Validation Accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")
