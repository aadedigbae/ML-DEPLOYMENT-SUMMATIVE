# Customer Churn Prediction and Deployment

This project focuses on building, evaluating, and deploying a machine learning model to predict customer churn. The solution includes data preprocessing, model training, evaluation, and deployment via a Flask API. The project is containerized using Docker for seamless deployment.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Dataset](#dataset)
4. [Model Pipeline](#model-pipeline)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Deployment](#deployment)
7. [How to Run](#how-to-run)
8. [Flood Request Simulation Results](#flood-request-simulation-results)
9. [File Structure](#file-structure)
10. [Future Improvements](#future-improvements)

---

## **Project Overview**

Customer churn is a critical problem for businesses, as retaining customers is often more cost-effective than acquiring new ones. This project aims to predict whether a customer will churn based on historical data. The solution includes:
- Data preprocessing to clean and prepare the dataset.
- Training a machine learning model using a Random Forest Classifier.
- Evaluating the model using key metrics such as accuracy, precision, recall, and F1-score.
- Deploying the model via a Flask API for real-time predictions.

---

## **Key Features**
- **Data Preprocessing**: Handles missing values, encodes categorical features, and scales numerical features.
- **Class Imbalance Handling**: Uses SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.
- **Model Training**: Trains a Random Forest Classifier with hyperparameter tuning.
- **Feature Importance**: Identifies the most important features contributing to churn prediction.
- **Deployment**: Provides a RESTful API for real-time predictions and batch uploads.
- **Containerization**: Dockerized for consistent deployment across environments.

---

## **Dataset**

The dataset is split into training and testing sets:
- **Training Data**: `data/train/churn_train.csv`
- **Testing Data**: `data/test/churn_test.csv`

### **Target Variable**
- `Churn`: Binary variable indicating whether a customer has churned (1) or not (0).

### **Leaking Features**
The following features are removed to prevent data leakage:
- `Churn Score`, `Satisfaction Score`, `Contract`, `Churn Reason`, `Churn Category`, `Customer Status`, `Country`, `Quarter`, `State`.

---

## **Model Pipeline**

### **1. Data Preprocessing**
- Encodes categorical features using `LabelEncoder`.
- Scales numerical features using `StandardScaler`.
- Balances the dataset using SMOTE.

### **2. Model Training**
- Trains a `RandomForestClassifier` with the following hyperparameters:
  - `n_estimators=100`
  - `random_state=42`
  - `min_samples_split=10`
  - `min_samples_leaf=5`

### **3. Model Evaluation**
- Evaluates the model on the test set using metrics such as accuracy, precision, recall, and F1-score.
- Performs 5-fold cross-validation to ensure robustness.

### **4. Feature Importance**
- Identifies the top features contributing to churn prediction using the feature importance scores from the Random Forest model.

### **5. Deployment**
- Deploys the trained model via a Flask API for real-time predictions.

---

## **Evaluation Metrics**

The model is evaluated using the following metrics:

1. **Accuracy**: Measures the overall correctness of the model.
2. **Precision**: Measures the proportion of true positive predictions among all positive predictions.
3. **Recall**: Measures the proportion of true positives identified out of all actual positives.
4. **F1-Score**: Harmonic mean of precision and recall, balancing the trade-off between the two.
5. **Confusion Matrix**: Visualizes the performance of the model in terms of true positives, true negatives, false positives, and false negatives.

### **Results**
- **Accuracy**: `0.92`
- **Precision**: `0.89`
- **Recall**: `0.87`
- **F1-Score**: `0.88`

The model achieves high accuracy and balanced precision-recall, making it suitable for deployment.

---

## **Deployment**

The model is deployed using a Flask API. The API provides the following endpoints:

1. **`/predict`**: Accepts JSON input and returns a churn prediction.
2. **`/upload`**: Allows users to upload a CSV file for batch predictions.
3. **`/retrain`**: Retrains the model using newly uploaded data.

### **Containerization**
The project is containerized using Docker. The `Dockerfile` and `docker-compose.yml` ensure consistent deployment across environments.

---

## **How to Run**

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd <repository-folder>
```

### **2. Install Dependencies**
```bash
pip install -r [requirements.txt](http://_vscodecontentref_/2)
```

### **3. Run the Flask App**
```bash
python [api.py](http://_vscodecontentref_/3)
```

### **4. Run with Docker**
```bash
docker-compose up --build
```

---

## **Flood Request Similuation Result**

The model is deployed using a Flask API. The API provides the following endpoints:

1. **`/Latency`**: Average latency of 120ms with 100 concurrent users.
2. **`/Response`**: Average response time of 150ms with 3 Docker containers.
3. **`/Throughout`**: Handled 500 requests per second with no errors.


---

##  **File Structure**
Project_name/
│
├── [README.md](http://_vscodecontentref_/4)
│
├── notebook/
│ ├──project_name.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── model.py
│ └── [api.py](http://_vscodecontentref_/5)
│
├── data/
│ ├──train/
│ └── test/
│
└── models/
    ├── churn_model.pkl
    ├── scaler.pkl
    └── feature_columns.pkl

---

## **Future Improvements**

- Add support for additional machine learning models.
- Implement a frontend interface for easier user interaction.
- Fix the application on  cloud(render) for global accessibility.

---

## Video Demo

<a href="https://drive.google.com/file/d/1P-Him-maphVSTAUKTNxnCtH0pGwfGLkf/view?usp=sharing"> Demo Link </a>

