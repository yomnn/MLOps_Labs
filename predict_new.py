import pandas as pd
import joblib

# Load the pipeline
model = joblib.load("models/model.joblib")

# Example input data (you can replace this with your own or load from CSV)
data = pd.DataFrame([
    {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    },
    {
        "Pclass": 1,
        "Sex": "female",
        "Age": 38.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 71.2833,
        "Embarked": "C"
    }
])

# Predict survival
predictions = model.predict(data)

# Display the result
for i, pred in enumerate(predictions):
    status = "Survived" if pred == 1 else "Did not survive"
    print(f"Passenger {i+1}: {status}")
