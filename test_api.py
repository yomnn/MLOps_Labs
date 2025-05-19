import requests

# API endpoint
url = "http://127.0.0.1:8000/predict"

# Example passenger data
passenger_data = {
    "Pclass": 2,
    "Sex": "female",
    "Age": 29,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 21.0,
    "Embarked": "S"
}

# Send POST request
response = requests.post(url, json=passenger_data)

# Display the response
if response.status_code == 200:
    result = response.json()
    print("Prediction result:")
    print(result)
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)
