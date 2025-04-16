import mlflow
import mlflow.sklearn
from datetime import datetime
import os

def setup_mlflow():
    """Initialize MLflow tracking"""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("spam-detection")

def log_model_metrics(model, metrics: dict, features: list):
    """Log model metrics and parameters to MLflow"""
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "model_type": type(model).__name__,
            "n_features": len(features),
            "timestamp": datetime.now().isoformat()
        })
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="spam-detection-model"
        )
        
        # Log feature importance if available
        if hasattr(model, "feature_importances_"):
            mlflow.log_dict(
                dict(zip(features, model.feature_importances_.tolist())),
                "feature_importance.json"
            )

def load_production_model():
    """Load the production model from MLflow"""
    model_uri = f"models:/spam-detection-model/production"
    return mlflow.sklearn.load_model(model_uri) 