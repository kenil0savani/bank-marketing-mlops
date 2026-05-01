from pathlib import Path
import joblib
from fastapi import FastAPI

from pydantic import BaseModel, Field
import pandas as pd


# App
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="Predict whether a customer will subscribe to a term deposit.",
    version="1.0.0",
)

# Load trained model once
MODEL_PATH = Path("models/final_model.joblib")
model = joblib.load(MODEL_PATH)


# Request schema
class CustomerData(BaseModel):
    age: int = Field(..., example=35)
    job: str = Field(..., example="management")
    marital: str = Field(..., example="married")
    education: str = Field(..., example="tertiary")
    default: str = Field(..., example="no")
    balance: float = Field(..., example=1200.5)
    housing: str = Field(..., example="yes")
    loan: str = Field(..., example="no")
    contact: str = Field(..., example="cellular")
    day: int = Field(..., example=15)
    month: str = Field(..., example="may")
    duration: float = Field(..., example=180.0)
    campaign: int = Field(..., example=2)
    pdays: int = Field(..., example=-1)
    previous: int = Field(..., example=0)
    poutcome: str = Field(..., example="unknown")


# Routs
@app.get("/")
def root():
    return {"message": "Bank Marketing Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model_path": str(MODEL_PATH)}


@app.post("/predict")
def predict(data: CustomerData):
    # Convert incoming data to DataFrame with one row
    input_df = pd.DataFrame([data.model_dump()])

    # Predict class
    prediction = model.predict(input_df)[0]

    try:
        # Predict probability for "YES"
        classes = list(model.classes_)
        yes_index = classes.index("yes")
        probability_yes = float(model.predict_proba(input_df)[0][yes_index])

        return {"prediction": prediction, "probability_yes": round(probability_yes, 4)}

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
