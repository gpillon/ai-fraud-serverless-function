from typing import Union
import os
import pickle
import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Initialize FastAPI with metadata for Swagger/OpenAPI
app = FastAPI(
    title="Fraud Detection API",
    description="""
    This API provides fraud detection capabilities for credit card transactions.
    It uses a machine learning model to predict whether a transaction is fraudulent based on various parameters.
    
    Parameters explained:
    - distance: Distance from last transaction location (in km)
    - ratio_to_median: Ratio of transaction amount compared to median transaction amount
    - pin: Whether PIN was used (1) or not (0)
    - chip: Whether chip was used (1) or not (0)
    - online: Whether it's an online transaction (1) or not (0)
    """,
    version="1.0.0",
    docs_url="/docs",    # Swagger UI endpoint
    redoc_url="/redoc"   # ReDoc endpoint
)

# Load the scaler
with open('scaler.pkl', 'rb') as handle:
    scaler = pickle.load(handle)

# Configuration
FRAUD_MODEL_URL = os.getenv('FRAUD_MODEL_URL', 'https://fraud-predictor-bionda.apps.okd-01.ocp.pillon.org/v2/models/fraud/infer')
THRESHOLD = float(os.getenv('FRAUD_THRESHOLD', '0.95'))

class TransactionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float

def call_fraud_model(normalized_data):
    json_data = {
        "inputs": [
            {
                "name": "dense_input",
                "shape": [1, 5],
                "datatype": "FP32",
                "data": normalized_data
            }
        ]
    }
    
    try:
        response = requests.post(FRAUD_MODEL_URL, json=json_data)
        response.raise_for_status()
        return response.json()['outputs'][0]['data'][0]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling fraud model: {str(e)}")

@app.get("/predict", 
         response_model=TransactionResponse,
         summary="Predict fraud probability for a transaction",
         description="""
         Predicts the probability of fraud for a credit card transaction based on the given parameters.
         
         Example values:
         - Normal transaction: distance=0, ratio_to_median=1, pin=1, chip=1, online=0
         - Suspicious transaction: distance=100, ratio_to_median=1.2, pin=0, chip=0, online=1
         """)
async def predict_fraud(
    distance: float = Query(..., description="Distance from last transaction location in km", example=0.0),
    ratio_to_median: float = Query(..., description="Ratio of transaction amount to median amount", example=1.0),
    pin: int = Query(..., description="PIN used (1) or not (0)", ge=0, le=1, example=1),
    chip: int = Query(..., description="Chip used (1) or not (0)", ge=0, le=1, example=1),
    online: int = Query(..., description="Online transaction (1) or not (0)", ge=0, le=1, example=0)
):
    # Convert input to list format expected by scaler
    raw_data = [distance, ratio_to_median, float(pin), float(chip), float(online)]
    
    # Normalize the data
    normalized_data = scaler.transform([raw_data]).tolist()[0]
    
    # Call the model
    fraud_probability = call_fraud_model(normalized_data)
    
    # Return prediction
    return TransactionResponse(
        is_fraud=fraud_probability > THRESHOLD,
        fraud_probability=fraud_probability
    )

@app.get("/health",
         summary="Health check endpoint",
         description="Returns the health status of the API")
async def health_check():
    return {"status": "healthy"}

@app.get("/",
         summary="API Root",
         description="Redirects to the API documentation")
async def root():
    return {"message": "Welcome to Fraud Detection API. Visit /docs for Swagger documentation or /redoc for ReDoc documentation"}
