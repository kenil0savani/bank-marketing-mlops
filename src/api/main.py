import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Bank Marketing Prediction API",
    description="Predict whether a customer will subscribe to a term deposit.",
    version="1.0.0",
)

# The trained model is produced by the DVC pipeline and is not committed to git,
# so the service starts without it and reports that through /health.
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/final_model.joblib"))
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


class CustomerData(BaseModel):
    age: int = Field(..., json_schema_extra={"example": 35})
    job: str = Field(..., json_schema_extra={"example": "management"})
    marital: str = Field(..., json_schema_extra={"example": "married"})
    education: str = Field(..., json_schema_extra={"example": "tertiary"})
    default: str = Field(..., json_schema_extra={"example": "no"})
    balance: float = Field(..., json_schema_extra={"example": 1200.5})
    housing: str = Field(..., json_schema_extra={"example": "yes"})
    loan: str = Field(..., json_schema_extra={"example": "no"})
    contact: str = Field(..., json_schema_extra={"example": "cellular"})
    day: int = Field(..., json_schema_extra={"example": 15})
    month: str = Field(..., json_schema_extra={"example": "may"})
    duration: float = Field(..., json_schema_extra={"example": 180.0})
    campaign: int = Field(..., json_schema_extra={"example": 2})
    pdays: int = Field(..., json_schema_extra={"example": -1})
    previous: int = Field(..., json_schema_extra={"example": 0})
    poutcome: str = Field(..., json_schema_extra={"example": "unknown"})


@app.get("/")
def root():
    return {"message": "Bank Marketing Prediction API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available at {MODEL_PATH}. Run the DVC pipeline first.",
        )

    input_df = pd.DataFrame([data.model_dump()])

    try:
        prediction = model.predict(input_df)[0]
        classes = list(model.classes_)
        yes_index = classes.index("yes")
        probability_yes = float(model.predict_proba(input_df)[0][yes_index])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"prediction": prediction, "probability_yes": round(probability_yes, 4)}
