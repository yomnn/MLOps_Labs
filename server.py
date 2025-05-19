from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Load the pipeline
model = joblib.load("models/model.joblib")

# FastAPI app instance
app = FastAPI(title="Titanic Survival Predictor")

# Input schema using Pydantic
class Passenger(BaseModel):
    Pclass: int = Field(..., example=3)
    Sex: str = Field(..., example="male")
    Age: float = Field(..., example=22.0)
    SibSp: int = Field(..., example=1)
    Parch: int = Field(..., example=0)
    Fare: float = Field(..., example=7.25)
    Embarked: str = Field(..., example="S")

@app.post("/predict")
def predict_survival(passenger: Passenger):
    # Convert input to DataFrame
    input_data = pd.DataFrame([passenger.dict()])

    # Make prediction
    prediction = model.predict(input_data)[0]

    return {
        "prediction": int(prediction),
        "survived": bool(prediction),
        "message": "Survived" if prediction == 1 else "Did not survive"
    }
