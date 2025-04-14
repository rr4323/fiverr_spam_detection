from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import json
import os
from typing import List, Dict, Any
from huggingface_hub import hf_hub_download
import tempfile
from pathlib import Path

app = FastAPI(title="Spammer Detection API",
             description="API for detecting spammers using machine learning",
             version="1.0.0")

# Cache directory for model files
CACHE_DIR = Path.home() / ".cache" / "fiverr_spam_detection"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Model files configuration
MODEL_FILES = {
    "spammer_detector.pkl": "spammer_detector.pkl",
    "scaler.pkl": "scaler.pkl",
    "feature_mapping.pkl": "feature_mapping.pkl",
    "model_metrics.json": "model_metrics.json"
}

def download_model_file(filename: str) -> str:
    """Download model file from Hugging Face if not in cache"""
    cache_path = CACHE_DIR / filename
    
    # Return cached file if it exists
    if cache_path.exists():
        return str(cache_path)
    
    # Download file if not in cache
    try:
        downloaded_path = hf_hub_download(
            repo_id="blackrajeev/fiverr_spam_detection",
            filename=filename,
            cache_dir=str(CACHE_DIR)
        )
        return downloaded_path
    except Exception as e:
        raise RuntimeError(f"Failed to download {filename} from Hugging Face: {str(e)}")

def load_model_artifacts():
    """Load model artifacts from cache or download from Hugging Face"""
    try:
        # Download or get cached model files
        model_path = download_model_file("spammer_detector.pkl")
        scaler_path = download_model_file("scaler.pkl")
        feature_mapping_path = download_model_file("feature_mapping.pkl")
        metrics_path = download_model_file("model_metrics.json")
        print(2)
        # Load the files
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(feature_mapping_path)
        feature_mapping = joblib.load(feature_mapping_path)
        print(feature_mapping)
        with open(metrics_path, 'r') as f:
            model_metrics = json.load(f)
            
        return model, scaler, feature_mapping, model_metrics
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifacts: {str(e)}")

# Load model and artifacts
model, scaler, feature_mapping, model_metrics = load_model_artifacts()

class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    is_spammer: bool
    risk_factors: List[Dict[str, str]]

class ModelInfo(BaseModel):
    metrics: Dict[str, float]
    feature_mapping: Dict[str, Dict[str, Any]]
    categories: List[str]

@app.get("/")
async def root():
    return {"message": "Spammer Detection API"}

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the trained model and available features"""
    # Get unique categories from feature mapping
    categories = sorted(list(set(info['category'] for info in feature_mapping.values())))
    
    return {
        "metrics": model_metrics,
        "feature_mapping": feature_mapping,
        "categories": categories
    }

def identify_risk_factors(features: Dict[str, float], feature_mapping: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Identify potential risk factors based on feature values and return detailed information
    """
    risk_factors = []
    
    # Define high-risk thresholds for specific features
    for feature_name, value in features.items():
        if feature_name not in feature_mapping:
            continue
            
        feature_info = feature_mapping[feature_name]
        description = feature_info.get('description', '')
        category = feature_info.get('category', '')
        
        # Check categorical features (typically boolean flags)
        if feature_info['type'] == 'categorical':
            if value == 1:
                risk_factors.append({
                    'feature': feature_name,
                    'description': description,
                    'category': category,
                    'value': str(value),
                    'reason': f"Flagged for {description.lower()}"
                })
        
        # Check numeric features against thresholds
        elif feature_info['type'] == 'numeric':
            # Add specific thresholds for numeric features
            if feature_name == 'X1' and value < 86400:  # Account age less than 24 hours
                risk_factors.append({
                    'feature': feature_name,
                    'description': description,
                    'category': category,
                    'value': str(value),
                    'reason': 'Very new account (less than 24 hours old)'
                })
            elif feature_name == 'X8' and value > 50:  # High number of recent logins
                risk_factors.append({
                    'feature': feature_name,
                    'description': description,
                    'category': category,
                    'value': str(value),
                    'reason': 'Unusually high number of recent logins'
                })
    
    return risk_factors

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction for a single user"""
    try:
        # Validate features
        print(feature_mapping.keys())
        print(request.features.keys())
        missing_features = [f for f in feature_mapping.keys() if f not in request.features]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}"
            )
        
        # Convert features to array in correct order
        features_array = np.array([request.features[f] for f in feature_mapping.keys()])
        
        # Scale features
        features_scaled = scaler.transform(features_array.reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0, 1]
        
        # Identify risk factors with detailed information
        risk_factors = identify_risk_factors(request.features, feature_mapping)
        
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
            # Validate features
            missing_features = [f for f in feature_mapping.keys() if f not in request.features]
            if missing_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required features: {missing_features}"
                )
            
            # Convert features to array in correct order
            features_array = np.array([request.features[f] for f in feature_mapping.keys()])
            
            # Scale features
            features_scaled = scaler.transform(features_array.reshape(1, -1))
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0, 1]
            
            # Identify risk factors with detailed information
            risk_factors = identify_risk_factors(request.features, feature_mapping)
            
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