import pytest
import requests
import json
from fastapi.testclient import TestClient
from src.api import app, FeedbackRequest, OnlineLearningConfig
import numpy as np
from datetime import datetime, timedelta

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint():
    test_data = {
        "X1": 1000,  # account age
        "X2": 113,   # messages sent
        "X3": 0,     # seller level
        "X4": 0,     # verification status
        "X5": 0,     # profile completion
        "X6": 0,     # portfolio items
        "X7": 0,     # active gigs
        "X8": 0,     # reviews received
        "X9": 0,     # rating tier
        "X10": 0,    # country tier
        "X11": 0,    # message frequency
        "X12": 0,    # urgent message count
        "X13": 0,    # link count
        "X14": 0,    # incomplete orders
        "X15": 0,    # risk score
        "X16": 0,    # message in day
        "X17": 0,    # message in week
        "X18": 0,    # message in month
        "X19": 0,    # urgent in message count
        "X20": 0,    # link in message
        "X21": 0,    # incomplete order
        "X22": 0,    # seller level flag
        "X23": 0,    # verification status flag
        "X24": 0,    # profile completion flag
        "X25": 0,    # portfolio items flag
        "X26": 0,    # active gigs flag
        "X27": 0,    # reviews received flag
        "X28": 0,    # rating tier flag
        "X29": 0,    # country tier flag
        "X30": 0,    # message frequency flag
        "X31": 0,    # urgent message count flag
        "X32": 0,    # link count flag
        "X33": 0,    # incomplete orders flag
        "X34": 0,    # risk score flag
        "X35": 0,    # message in day flag
        "X36": 0,    # message in week flag
        "X37": 0,    # message in month flag
        "X38": 0,    # urgent in message count flag
        "X39": 0,    # link in message flag
        "X40": 0,    # incomplete order flag
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert "probability" in result
    assert "risk_factors" in result
    assert "prediction_id" in result

def test_feedback_endpoint():
    # First make a prediction to get a prediction_id
    test_data = {
        "X1": 1000,
        "X2": 113,
        "X3": 0,
        "X4": 0,
        "X5": 0,
        "X6": 0,
        "X7": 0,
        "X8": 0,
        "X9": 0,
        "X10": 0,
        "X11": 0,
        "X12": 0,
        "X13": 0,
        "X14": 0,
        "X15": 0,
        "X16": 0,
        "X17": 0,
        "X18": 0,
        "X19": 0,
        "X20": 0,
        "X21": 0,
        "X22": 0,
        "X23": 0,
        "X24": 0,
        "X25": 0,
        "X26": 0,
        "X27": 0,
        "X28": 0,
        "X29": 0,
        "X30": 0,
        "X31": 0,
        "X32": 0,
        "X33": 0,
        "X34": 0,
        "X35": 0,
        "X36": 0,
        "X37": 0,
        "X38": 0,
        "X39": 0,
        "X40": 0,
    }
    
    pred_response = client.post("/predict", json=test_data)
    prediction_id = pred_response.json()["prediction_id"]
    
    # Submit feedback
    feedback_data = {
        "prediction_id": prediction_id,
        "is_correct": True,
        "actual_label": None,
        "feedback_notes": "Test feedback"
    }
    
    response = client.post("/feedback", json=feedback_data)
    assert response.status_code == 200
    assert response.json() == {"status": "feedback received"}

def test_feedback_stats_endpoint():
    response = client.get("/model/feedback/stats")
    assert response.status_code == 200
    stats = response.json()
    assert "total_feedback" in stats
    assert "correct_predictions" in stats
    assert "accuracy" in stats
    assert "last_retrain" in stats

def test_online_learning_config():
    config = OnlineLearningConfig()
    assert config.learning_rate == 0.01
    assert config.batch_size == 100
    assert config.min_samples == 1000
    assert config.retrain_interval == 3600

def test_should_retrain():
    # Test with empty feedback buffer
    assert not app.state.should_retrain()
    
    # Add enough feedback to trigger retraining
    for _ in range(1000):
        feedback = FeedbackRequest(
            prediction_id="test_id",
            is_correct=True,
            actual_label=None,
            feedback_notes="Test feedback"
        )
        app.state.feedback_buffer.append(feedback)
    
    assert app.state.should_retrain()

def test_retrain_model():
    # Add some feedback data
    for _ in range(1000):
        feedback = FeedbackRequest(
            prediction_id="test_id",
            is_correct=True,
            actual_label=None,
            feedback_notes="Test feedback"
        )
        app.state.feedback_buffer.append(feedback)
    
    # Test retraining
    app.state.retrain_model()
    
    # Verify model was updated
    assert app.state.model is not None
    assert app.state.last_retrain is not None
    assert len(app.state.feedback_buffer) == 0 