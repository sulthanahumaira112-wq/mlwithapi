from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model=joblib.load("house_model.pkl")
app=FastAPI()

class HouseData(BaseModel):
    area:float

@app.get("/")
def home():
    return {"message":"House Price Prediction model is running"}

@app.post("/predict")
def predict_price(data:HouseData):

    prediction=model.predict([
        [data.area]
    ])

    return{
        "area":data.area,
        "predicted_price":round(float(prediction[0]),2)
    }