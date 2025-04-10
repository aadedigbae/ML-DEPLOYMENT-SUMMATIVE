from locust import HttpUser, task, between

class LoadTestUser(HttpUser):
    wait_time = between(1, 5)  # Simulates a wait time between requests (1 to 5 seconds)

    @task
    def predict(self):
        # Replace the feature names and values with the actual ones used in your model
        self.client.post("/predict", json={
            "Age": 35,
            "Gender": "Male",
            "Tenure": 5,
            "Balance": 75000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 50000.0,
            "Geography": "France",
            "CreditScore": 600
        })