from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the model
model_path = "/tmp/model.pkl"
model = joblib.load(model_path)

@app.post("/predict")
async def predict(data: dict):
    # Convert input data into a DataFrame
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}
