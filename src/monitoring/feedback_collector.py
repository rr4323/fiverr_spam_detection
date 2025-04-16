import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import json
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
import logging
from datetime import timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackEntry(BaseModel):
    prediction_id: str
    true_label: int
    model_prediction: int
    model_confidence: float
    features: Dict[str, float]
    feedback_source: str
    timestamp: datetime
    additional_comments: Optional[str] = None

class ModelMonitor:
    def __init__(self, model_path: str, feedback_path: str, retrain_threshold: int = 100):
        self.model_path = Path(model_path)
        self.feedback_path = Path(feedback_path)
        self.feedback_data: List[FeedbackEntry] = []
        self.retrain_threshold = retrain_threshold
        self.performance_metrics = {
            'accuracy': [],
            'false_positives': [],
            'false_negatives': [],
            'drift_score': []
        }
        
        # Initialize MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("spam_detection_monitoring")
        
        # Load initial model
        self.current_model = self._load_model()
        self.feature_names = self._load_feature_names()
        
        # Create feedback directory if it doesn't exist
        self.feedback_path.mkdir(parents=True, exist_ok=True)
        
    def _load_model(self) -> RandomForestClassifier:
        """Load the current production model"""
        return joblib.load(self.model_path / 'spammer_detector.pkl')
    
    def _load_feature_names(self) -> List[str]:
        """Load feature names from model info"""
        with open(self.model_path / 'model_info.json', 'r') as f:
            model_info = json.load(f)
        return model_info['feature_names']
    
    def add_feedback(self, feedback: FeedbackEntry) -> None:
        """Add new feedback entry and trigger evaluation"""
        self.feedback_data.append(feedback)
        self._save_feedback(feedback)
        self._evaluate_performance()
        
        # Check if retraining is needed
        if len(self.feedback_data) >= self.retrain_threshold:
            self._trigger_retraining()
    
    def _save_feedback(self, feedback: FeedbackEntry) -> None:
        """Save feedback to disk"""
        feedback_file = self.feedback_path / f"feedback_{feedback.prediction_id}.json"
        with open(feedback_file, 'w') as f:
            json.dump(feedback.dict(), f)
    
    def _evaluate_performance(self) -> Dict[str, float]:
        """Evaluate current model performance based on feedback"""
        if not self.feedback_data:
            return {}
        
        recent_feedback = self.feedback_data[-100:]  # Look at last 100 feedbacks
        
        y_true = [f.true_label for f in recent_feedback]
        y_pred = [f.model_prediction for f in recent_feedback]
        
        # Calculate metrics
        accuracy = np.mean([t == p for t, p in zip(y_true, y_pred)])
        false_positives = sum([1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1])
        false_negatives = sum([1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0])
        
        # Calculate drift score (simplified version)
        drift_score = self._calculate_drift_score(recent_feedback)
        
        metrics = {
            'accuracy': accuracy,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'drift_score': drift_score
        }
        
        # Log metrics to MLflow
        with mlflow.start_run(run_name=f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_metrics(metrics)
        
        # Update performance history
        for key, value in metrics.items():
            self.performance_metrics[key].append(value)
        
        return metrics
    
    def _calculate_drift_score(self, feedback_entries: List[FeedbackEntry]) -> float:
        """Calculate feature drift score"""
        if not feedback_entries:
            return 0.0
        
        # Extract features from feedback
        recent_features = pd.DataFrame([f.features for f in feedback_entries])
        
        # Calculate drift score based on feature distributions
        drift_score = 0.0
        for feature in self.feature_names:
            if feature in recent_features:
                current_mean = recent_features[feature].mean()
                current_std = recent_features[feature].std()
                # Compare with original model's feature distribution
                # This is a simplified version - you might want to use more sophisticated drift detection
                drift_score += abs(current_mean - 0) / (current_std + 1e-6)
        
        return drift_score / len(self.feature_names)
    
    def _trigger_retraining(self) -> None:
        """Trigger model retraining if needed"""
        logger.info("Starting model retraining process...")
        
        # Prepare training data from feedback
        X = []
        y = []
        for feedback in self.feedback_data:
            features = [feedback.features[f] for f in self.feature_names]
            X.append(features)
            y.append(feedback.true_label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train new model
        new_model = RandomForestClassifier(
            n_estimators=500,
            class_weight={0: 1, 1: 5},
            max_depth=15,
            min_samples_split=3,
            random_state=42
        )
        
        with mlflow.start_run(run_name=f"retraining_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            new_model.fit(X, y)
            
            # Log model and metrics
            mlflow.sklearn.log_model(new_model, "model")
            mlflow.log_params(new_model.get_params())
            
            # Calculate and log performance metrics
            train_accuracy = new_model.score(X, y)
            mlflow.log_metric("train_accuracy", train_accuracy)
        
        # Save new model
        joblib.dump(new_model, self.model_path / 'spammer_detector_new.pkl')
        
        # Clear feedback data after retraining
        self.feedback_data = []
        logger.info("Model retraining completed. New model saved as spammer_detector_new.pkl")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        if not self.performance_metrics['accuracy']:
            return {"message": "No performance data available yet"}
        
        return {
            "last_24h": {
                "accuracy": np.mean(self.performance_metrics['accuracy'][-24:]),
                "false_positives": sum(self.performance_metrics['false_positives'][-24:]),
                "false_negatives": sum(self.performance_metrics['false_negatives'][-24:]),
                "drift_score": np.mean(self.performance_metrics['drift_score'][-24:])
            },
            "total": {
                "accuracy": np.mean(self.performance_metrics['accuracy']),
                "false_positives": sum(self.performance_metrics['false_positives']),
                "false_negatives": sum(self.performance_metrics['false_negatives']),
                "drift_score": np.mean(self.performance_metrics['drift_score'])
            }
        } 