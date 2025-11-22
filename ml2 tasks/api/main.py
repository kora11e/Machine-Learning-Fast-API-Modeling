from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pickle
import numpy as np
from pydantic import BaseModel
import os

#   print(os.getcwd())

with open('../models/rf_regressor.pkl', 'rb') as f, \
     open('../models/dt_regressor.pkl', 'rb') as f2, \
     open('../models/xgb_regressor.pkl', 'rb') as f3:
    
    rf_model = pickle.load(f)
    dt_model = pickle.load(f2)
    xgb_model = pickle.load(f3)

app = FastAPI(title='Insurance Charges Prediction')

class InsuranceData(BaseModel):
    age:        int
    bmi:        float
    children:   int
    sex_0 :     float
    sex_1 :     float
    region_0:   float 
    region_1:   float
    region_2:   float
    region_3:   float
    smoker_0:   float
    smoker_1:   float

@app.middleware("http")
async def add_prediction_header(request: Request, call_next):
    response = await call_next(request)

    try:
        body = response.body.decode()
        import json
        data = json.loads(body)
        if "predicted_price" in data:
            response.headers["RF-Predicted-Price"] = str(data["predicted_price"])
            
    except Exception as e:
        print("Header middleware error:", e)

    return response

@app.get('/')
def main():
    return{'message': 'Welcome to Insurance API'}

@app.post('/predict_charge/rf')
def predict_charge_rf(data : InsuranceData):
    features = np.array(
        [[
            #list(data.__annotations__.keys())
            data.age, 
            data.bmi, 
            data.children,
            data.sex_0, 
            data.sex_1, 
            data.region_0,
            data.region_1,
            data.region_2,
            data.region_3, 
            data.smoker_0,
            data.smoker_1
        ]]
    )

    prediction = rf_model.predict(features)

    return {'predicted_price': float(prediction[0])}

@app.post('/predict_charge/dt')
def predict_charge_dt(data : InsuranceData):
    features = np.array(
        [[
            data.age, 
            data.bmi, 
            data.children,
            data.sex_0, 
            data.sex_1, 
            data.region_0,
            data.region_1,
            data.region_2,
            data.region_3, 
            data.smoker_0,
            data.smoker_1
        ]]
    )
    
    prediction = dt_model.predict(features)

    return {'predicted_price': float(prediction[0])}


@app.post('/predict_charge/xgb')
def predict_charge_xgb(data : InsuranceData):
    features = np.array(
        [[
            data.age, 
            data.bmi, 
            data.children,
            data.sex_0, 
            data.sex_1, 
            data.region_0,
            data.region_1,
            data.region_2,
            data.region_3, 
            data.smoker_0,
            data.smoker_1
        ]]
    )
    
    prediction = xgb_model.predict(features)

    return {'predicted_price': float(prediction[0])}
