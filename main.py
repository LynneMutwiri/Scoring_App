from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# load model
model_package = joblib.load("credit_default_best_model.joblib")

pipeline = model_package["pipeline"]
features = model_package["selected_features"]

@app.get("/")
def home():
    return {"message": "Credit Scoring API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        df = df[features]

        proba = pipeline.predict_proba(df)[0][1]

        return {
            "default_probability": float(proba),
            "risk": "High Risk" if proba > 0.5 else "Low Risk"
        }

    except Exception as e:
        return {"error": str(e)}   