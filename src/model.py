import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEANED_TRAIN_PATH = os.path.join(BASE_DIR, "data/train/cleaned_train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models/model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "models/feature_columns.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")

# Load cleaned dataset
df = pd.read_csv(CLEANED_TRAIN_PATH)

# Initialize the encoder
encoder = LabelEncoder()

# Apply the encoder to categorical columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(df[col])

# Split features and target
X = df.drop(columns=['Churn'])  # Replace 'Churn' with actual target column name
y = df['Churn']

# Check for data leakage
print("Correlation with target:\n", df.corr()['Churn'].sort_values(ascending=False))

# Remove leaking features
leaking_features = ['Churn Score', 'Satisfaction Score', 'Contract', 'Churn Reason', 'Churn Category', 'Customer Status', 'Country', 'Quarter', 'State']
X = X.drop(columns=leaking_features)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract feature names before scaling
FEATURE_COLUMNS = X_train.columns.tolist()

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

X_train = pd.DataFrame(X_train, columns=FEATURE_COLUMNS)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=10, min_samples_leaf=5)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Model trained. Validation Accuracy: {accuracy:.2f}")

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", scores)
print("Mean Cross-Validation Accuracy:", scores.mean())

# Plot feature importance
feature_importances = model.feature_importances_
sorted_indices = feature_importances.argsort()

# Select top N features based on importance
N = 10
top_features = [FEATURE_COLUMNS[i] for i in sorted_indices[-N:]]
print(f"Top {N} features selected:", top_features)

# Update FEATURE_COLUMNS to match the top features
FEATURE_COLUMNS = top_features

# Filter the dataset to include only the top features
X = X[FEATURE_COLUMNS]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

X_train = pd.DataFrame(X_train, columns=FEATURE_COLUMNS)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=10, min_samples_leaf=5)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")

# Save the top features
joblib.dump(top_features, FEATURE_COLUMNS_PATH)
print(f"Top feature columns saved at {FEATURE_COLUMNS_PATH}")

# Save the scaler
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved at {SCALER_PATH}")

# Load and inspect the scaler
scaler = joblib.load(SCALER_PATH)
print("Scaler feature names (if available):", getattr(scaler, "feature_names_in_", "Not available"))

# Plot feature importance
plt.figure(figsize=(10, 8))

# Use only the top N features for plotting
top_sorted_indices = sorted_indices[-N:]  # Indices of the top N features
top_feature_importances = feature_importances[top_sorted_indices]  # Importance values for the top N features
top_feature_names = [FEATURE_COLUMNS[i] for i in range(len(FEATURE_COLUMNS))]  # Names of the top N features

plt.barh(range(len(top_sorted_indices)), top_feature_importances, align='center')
plt.yticks(range(len(top_sorted_indices)), top_feature_names)
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Random Forest")
plt.show()

# Load and inspect the model
model = joblib.load(MODEL_PATH)
print("Features the model was trained on:", getattr(model, "feature_names_in_", "Not available"))

