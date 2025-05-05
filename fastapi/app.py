from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from minio import Minio

import os
from dotenv import load_dotenv
load_dotenv()

ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
SECRET_KEY = os.getenv('MINIO_SECRET_KEY')

app = FastAPI()

# with open('src/model.pkl', 'rb') as f:
#     model = pickle.load(f)

# upload from s3 (minio)

client = Minio("s3:9099",
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=False
)
model = pickle.load(client.get_object('credit-model', 'model.pkl'))

class PredictionInput(BaseModel):
    age: int
    sex: str
    job: str
    housing: str
    credit_amount: float
    duration: int


@app.get('/health')
def health_check():
    return {"status": 'healthy'}

@app.post('/predict')
def predict(input_data: PredictionInput):
    data_to_predict = pd.DataFrame({
        'age': [input_data.age],
        'sex': [input_data.sex],
        'job': [input_data.job],
        'housing': [input_data.housing],
        'credit_amount': [input_data.credit_amount],
        'duration': [input_data.duration]
    })

    prediction = model.predict(data_to_predict)[0]
    result = 'Good client' if prediction == 1 else 'Bad client'

    return {'prediction': result}