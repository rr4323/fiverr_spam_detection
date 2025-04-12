from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import json
import os
from typing import List, Dict, Any

app = FastAPI(title="Spammer Detection API",
             description="API for detecting spammers using machine learning",
             version="1.0.0")

# Load model artifacts
model_path = os.path.join('..', 'models', 'spammer_detector.pkl')
scaler_path = os.path.join('..', 'models', 'scaler.pkl')
feature_mapping_path = os.path.join('..', 'models', 'feature_mapping.pkl')
metrics_path = os.path.join('..', 'models', 'model_metrics.json')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_mapping = joblib.load(feature_mapping_path)
    with open(metrics_path, 'r') as f:
        model_metrics = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {str(e)}")

class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    is_spammer: bool
    risk_factors: List[str]

class ModelInfo(BaseModel):
    metrics: Dict[str, float]
    feature_mapping: Dict[str, Dict[str, str]]

@app.get("/")
async def root():
    return {"message": "Spammer Detection API"}

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the trained model"""
    return {
        "metrics": model_metrics,
        "feature_mapping": feature_mapping
    }

def validate_features(features: Dict[str, float]) -> np.ndarray:
    """Validate and convert features to numpy array"""
    # Check if all required features are present
    missing_features = [f for f in feature_mapping.keys() if f not in features]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing_features}"
        )
    
    # Convert features to array in correct order
    return np.array([features[f] for f in feature_mapping.keys()])

def identify_risk_factors(features: Dict[str, float]) -> List[str]:
    """Identify potential risk factors based on feature values"""
    risk_factors = []
    for feature_name, value in features.items():
        feature_info = feature_mapping[feature_name]
        if feature_info['type'] == 'boolean' and value == 1:
            risk_factors.append(feature_name)
        elif feature_info['type'] == 'numeric':
            # You might want to add specific thresholds for numeric features
            if value > 0:  # Example threshold
                risk_factors.append(feature_name)
    return risk_factors

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction for a single user"""
    try:
        # Validate and convert features
        features_array = validate_features(request.features)
        
        # Scale features
        features_scaled = scaler.transform(features_array.reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0, 1]
        
        # Identify risk factors
        risk_factors = identify_risk_factors(request.features)
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "is_spammer": bool(prediction),
            "risk_factors": risk_factors
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Make predictions for multiple users"""
    try:
        results = []
        for request in requests:
            # Validate and convert features
            features_array = validate_features(request.features)
            
            # Scale features
            features_scaled = scaler.transform(features_array.reshape(1, -1))
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0, 1]
            
            # Identify risk factors
            risk_factors = identify_risk_factors(request.features)
            
            results.append({
                "prediction": int(prediction),
                "probability": float(probability),
                "is_spammer": bool(prediction),
                "risk_factors": risk_factors
            })
        
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 