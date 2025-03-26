import pandas as pd
import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models/model.pkl")
CLEANED_TEST_PATH = os.path.join(BASE_DIR, "data/test/cleaned_test.csv")

# Load the trained model
model = joblib.load(MODEL_PATH)

# Load cleaned test data
test_df = pd.read_csv(CLEANED_TEST_PATH)

# Drop ID columns if necessary
X_test = test_df.drop(columns=['Churn'])  # Adjust based on dataset

# Make predictions
predictions = model.predict(X_test)

# Save predictions
output_path = os.path.join(BASE_DIR, "data/test/predictions.csv")
output_df = test_df.copy()
output_df['Churn_Predicted'] = predictions
output_df.to_csv(output_path, index=False)

print(f"Predictions saved to '{output_path}'")
