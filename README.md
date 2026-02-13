# ML_MODEL_INNOVATHON
In your backend folder:

pip install fastapi uvicorn joblib pandas numpy xgboost

Step 2 — Create app.py

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Load model package
package = joblib.load("smart_crowd_forecast_system.pkl")

model = package["model"]
le = package["label_encoder"]
features = package["features"]

app = FastAPI()

# Request schema
class CrowdInput(BaseModel):
    location: str
    hour: int
    weekday: int
    rssi: float
    value: float
    prev_density: float
    prev2_density: float
    rolling_mean_3: float

@app.post("/predict")
def predict(data: CrowdInput):

    # Encode location
    encoded_location = le.transform([data.location])[0]

    # Prepare dataframe
    input_data = pd.DataFrame([{
        "location": encoded_location,
        "hour": data.hour,
        "weekday": data.weekday,
        "rssi": data.rssi,
        "value": data.value,
        "prev_density": data.prev_density,
        "prev2_density": data.prev2_density,
        "rolling_mean_3": data.rolling_mean_3
    }])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_density": float(prediction)
    }
Step 3 — Run Backend

uvicorn app:app --reload
