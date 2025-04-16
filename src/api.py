from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import json
import os
from typing import List, Dict, Any, Optional
from huggingface_hub import hf_hub_download
import tempfile
from pathlib import Path
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from src.opentelemetry.config import setup_tracing, instrument_fastapi, get_tracer
from monitoring.feedback_collector import ModelMonitor, FeedbackEntry
from datetime import datetime
from sklearn.linear_model import SGDClassifier
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
from src.train_model import preprocess_data, handle_class_imbalance, evaluate_model, save_model_artifacts
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Spammer Detection API",
             description="API for detecting spammers using machine learning",
             version="1.0.0")

# Initialize OpenTelemetry
tracer = setup_tracing()
instrument_fastapi(app)

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

class FeedbackRequest(BaseModel):
    prediction_id: str
    is_correct: bool
    actual_label: Optional[bool] = None
    feedback_notes: Optional[str] = None

class OnlineLearningConfig(BaseModel):
    learning_rate: float = 0.01
    batch_size: int = 100
    min_samples: int = 1000
    retrain_interval: int = 3600  # 1 hour in seconds

# Initialize online learning components
online_learning_config = OnlineLearningConfig()
feedback_buffer = []
feedback_lock = threading.Lock()
last_retrain_time = datetime.now()

# Add model version tracking
class ModelVersion:
    def __init__(self):
        self.version = 1
        self.last_updated = datetime.now()
        self.metrics = {}

model_version = ModelVersion()
model_lock = threading.Lock()

def get_model_version() -> dict:
    """Get current model version information"""
    return {
        "version": model_version.version,
        "last_updated": model_version.last_updated.isoformat(),
        "metrics": model_version.metrics
    }

def collect_feedback(feedback: FeedbackRequest):
    """Collect feedback and store in buffer"""
    with feedback_lock:
        feedback_buffer.append({
            'timestamp': datetime.now(),
            'prediction_id': feedback.prediction_id,
            'is_correct': feedback.is_correct,
            'actual_label': feedback.actual_label,
            'feedback_notes': feedback.feedback_notes
        })

def should_retrain():
    """Check if model should be retrained"""
    with feedback_lock:
        if len(feedback_buffer) < online_learning_config.min_samples:
            return False
        time_since_last_retrain = (datetime.now() - last_retrain_time).total_seconds()
        return time_since_last_retrain >= online_learning_config.retrain_interval

def retrain_model():
    """Retrain model with collected feedback"""
    global model, last_retrain_time, model_version
    
    with feedback_lock:
        if len(feedback_buffer) < online_learning_config.min_samples:
            return
        
        try:
            # Convert feedback buffer to DataFrame
            feedback_df = pd.DataFrame(feedback_buffer)
            
            # Extract features and labels
            X = feedback_df['features'].values
            y = feedback_df['actual_label'].values
            
            # Preprocess data using the same logic as train_model.py
            X_scaled, y, _, scaler, feature_names = preprocess_data(
                pd.DataFrame(X, columns=feature_mapping.keys())
            )
            
            # Handle class imbalance
            X_resampled, y_resampled = handle_class_imbalance(X_scaled, y)
            
            # Initialize new model with same parameters as train_model.py
            new_model = RandomForestClassifier(
                n_estimators=500,
                class_weight={0: 1, 1: 5},
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Train in batches
            batch_size = online_learning_config.batch_size
            for i in range(0, len(X_resampled), batch_size):
                batch_X = X_resampled[i:i + batch_size]
                batch_y = y_resampled[i:i + batch_size]
                new_model.fit(batch_X, batch_y)
            
            # Evaluate the model
            metrics, trained_model = evaluate_model(
                new_model, 
                X_resampled, 
                X_scaled, 
                y_resampled, 
                y, 
                feature_names,
                threshold=0.3
            )
            
            # Update model and artifacts with thread safety
            with model_lock:
                model = trained_model
                last_retrain_time = datetime.now()
                model_version.version += 1
                model_version.last_updated = last_retrain_time
                model_version.metrics = metrics
            
            # Save updated model artifacts
            save_model_artifacts(
                model,
                scaler,
                feature_names,
                metrics,
                threshold=0.3
            )
            
            # Clear feedback buffer
            feedback_buffer.clear()
            
            # Log retraining success
            print(f"Model retrained successfully with {len(X)} samples")
            print(f"New metrics: {metrics}")
            
        except Exception as e:
            print(f"Error during model retraining: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrain model: {str(e)}"
            )

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
            if feature_name == 'X19' and value > 5:  # High urgent message count
                risk_factors.append({
                    'feature': feature_name,
                    'description': description,
                    'category': category,
                    'value': str(value),
                    'reason': 'High number of urgent messages'
                })
            elif feature_name == 'X8' and value > 10:  # High number of verified reviews
                risk_factors.append({
                    'feature': feature_name,
                    'description': description,
                    'category': category,
                    'value': str(value),
                    'reason': 'Unusually high number of verified reviews'
                })
            elif feature_name == 'X16' and value > 15:  # High daily message count
                risk_factors.append({
                    'feature': feature_name,
                    'description': description,
                    'category': category,
                    'value': str(value),
                    'reason': 'Excessive daily messaging activity'
                })
            elif feature_name == 'X18' and value > 8:  # High incomplete orders
                risk_factors.append({
                    'feature': feature_name,
                    'description': description,
                    'category': category,
                    'value': str(value),
                    'reason': 'High number of incomplete orders'
                })
    
    return risk_factors

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction for a single user"""
    try:
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
        
        # Make prediction with thread safety
        with model_lock:
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0, 1]
        
        # Identify risk factors with detailed information
        risk_factors = identify_risk_factors(request.features, feature_mapping)
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "is_spammer": bool(prediction),
            "risk_factors": risk_factors,
            "model_version": model_version.version
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

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": app.version,
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for model improvement"""
    try:
        collect_feedback(feedback)
        
        # Check if retraining is needed
        if should_retrain():
            with ThreadPoolExecutor() as executor:
                executor.submit(retrain_model)
        
        return {"status": "success", "message": "Feedback received"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/feedback/stats")
async def get_feedback_stats():
    """Get feedback statistics"""
    with feedback_lock:
        total_feedback = len(feedback_buffer)
        correct_predictions = sum(1 for f in feedback_buffer if f['is_correct'])
        accuracy = correct_predictions / total_feedback if total_feedback > 0 else 0
        
        return {
            "total_feedback": total_feedback,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "last_retrain": last_retrain_time.isoformat()
        }

@app.get("/model/version")
async def get_model_version_info():
    """Get current model version information"""
    return get_model_version()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 