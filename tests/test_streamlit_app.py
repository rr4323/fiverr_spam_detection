import pytest
import streamlit as st
from src.streamlit_app import create_feedback_section, display_feedback_stats
import requests
from unittest.mock import patch, MagicMock

def test_create_feedback_section():
    # Mock the prediction result
    prediction_result = {
        "prediction": "spam",
        "probability": 0.85,
        "risk_factors": ["high message frequency", "multiple links"],
        "prediction_id": "test_id_123"
    }
    
    # Mock the API response
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"status": "feedback received"}
        
        # Test with correct prediction
        feedback_data = {
            "prediction_id": "test_id_123",
            "is_correct": True,
            "actual_label": None,
            "feedback_notes": "Test feedback"
        }
        
        result = create_feedback_section(prediction_result)
        assert result is not None
        
        # Test with incorrect prediction
        feedback_data = {
            "prediction_id": "test_id_123",
            "is_correct": False,
            "actual_label": "not_spam",
            "feedback_notes": "Test feedback"
        }
        
        result = create_feedback_section(prediction_result)
        assert result is not None

def test_display_feedback_stats():
    # Mock the API response
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "total_feedback": 100,
            "correct_predictions": 85,
            "accuracy": 0.85,
            "last_retrain": "2024-03-20T12:00:00Z"
        }
        
        result = display_feedback_stats()
        assert result is not None
        
        # Test with empty stats
        mock_get.return_value.json.return_value = {
            "total_feedback": 0,
            "correct_predictions": 0,
            "accuracy": 0.0,
            "last_retrain": None
        }
        
        result = display_feedback_stats()
        assert result is not None

def test_feedback_form_validation():
    # Test valid feedback submission
    valid_feedback = {
        "prediction_id": "test_id_123",
        "is_correct": True,
        "actual_label": None,
        "feedback_notes": "Test feedback"
    }
    
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        result = create_feedback_section({"prediction_id": "test_id_123"})
        assert result is not None
    
    # Test invalid feedback submission (missing prediction_id)
    invalid_feedback = {
        "is_correct": True,
        "actual_label": None,
        "feedback_notes": "Test feedback"
    }
    
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 400
        result = create_feedback_section({})
        assert result is not None

def test_feedback_stats_error_handling():
    # Test API error
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 500
        result = display_feedback_stats()
        assert result is not None
    
    # Test network error
    with patch('requests.get') as mock_get:
        mock_get.side_effect = requests.exceptions.RequestException
        result = display_feedback_stats()
        assert result is not None

def test_feedback_ui_elements():
    # Test feedback form elements
    with patch('streamlit.form') as mock_form:
        mock_form.return_value = MagicMock()
        result = create_feedback_section({"prediction_id": "test_id_123"})
        assert result is not None
    
    # Test feedback stats display elements
    with patch('streamlit.metric') as mock_metric:
        mock_metric.return_value = MagicMock()
        result = display_feedback_stats()
        assert result is not None 