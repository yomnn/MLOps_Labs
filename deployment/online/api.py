import numpy as np
import joblib
import litserve as ls
from deployment.online.request import InferenceRequest

class InferenceAPI(ls.LitAPI):
    def setup(self, device="cpu"):
        # Load the trained pipeline (preprocessor + model)
        with open("models/model.pkl", "rb") as f:
            self._model = joblib.load(f)
        print("✅ Model pipeline loaded successfully.")

    def decode_request(self, request):
        try:
            input_data = InferenceRequest(**request["input"])

            features = [
                input_data.Pclass,
                input_data.Sex,
                input_data.Age,
                input_data.SibSp,
                input_data.Parch,
                input_data.Fare,
                input_data.Embarked
            ]

            x = np.array([features], dtype=object)  # keep dtype object for mixed types
            return x
        except Exception as e:
            print(f"❌ Decode error: {e}")
            return None

    def predict(self, x):
        if x is not None:
            try:
                return self._model.predict(x)
            except Exception as e:
                print(f"❌ Prediction error: {e}")
                return None
        return None

    def encode_response(self, output):
        if output is None:
            return {
                "message": "Prediction failed",
                "prediction": None
            }
        return {
            "message": "Prediction successful",
            "prediction": output.tolist()
        }
