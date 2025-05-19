from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import mlflow
import mlflow.pyfunc
import pandas as pd
import os

# OPTIONAL: Set these if you need to authenticate
# os.environ["MLFLOW_TRACKING_USERNAME"] = "your-username"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "your-dagshub-token"

# Set the MLflow tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/yomna.wael.mu/MLOps_Labs.mlflow")

# Load the model from the DagsHub MLflow registry
model = mlflow.pyfunc.load_model(model_uri="models:/best_model/Production")

app = FastAPI(title="Titanic Survival Prediction API")

# Define input schema using Pydantic
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

# Define the prediction endpoint
@app.post("/predict")
def predict(passengers: List[Passenger]):
    # Convert list of passenger objects to DataFrame
    df = pd.DataFrame([p.dict() for p in passengers])
    # Make prediction
    predictions = model.predict(df)
    # Return predictions
    return {"predictions": predictions.tolist()}
