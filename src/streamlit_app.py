import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go
from typing import Dict, Any

# Set page config
st.set_page_config(
    page_title="Fiverr Spammer Detection",
    page_icon="🛡️",
    layout="wide"
)

# API configuration
API_URL = "http://localhost:8000"

@st.cache_resource
def load_model_and_mapping():
    """Load model artifacts and feature mapping"""
    try:
        feature_mapping = joblib.load('../models/feature_mapping.pkl')
        with open('../models/model_metrics.json', 'r') as f:
            model_metrics = json.load(f)
        return feature_mapping, model_metrics
    except Exception as e:
        st.error(f"Failed to load model artifacts: {str(e)}")
        return {}, {}

def create_feature_input(feature_name: str, feature_info: Dict[str, Any]) -> Any:
    """Create appropriate input widget based on feature type and mapping"""
    if feature_info['type'] == 'boolean':
        return st.checkbox(
            feature_info['description'],
            value=False,
            help=feature_info.get('description', '')
        )
    
    elif feature_info['type'] == 'numeric':
        # For AccountAge_Seconds, convert to days for better understanding
        if feature_name == 'AccountAge_Seconds':
            days = st.number_input(
                "Account Age (in days)",
                min_value=0,
                max_value=int(feature_info['max_value'] / 86400),  # Convert seconds to days
                value=0,
                step=1,
                help="Age of the account in days"
            )
            return days * 86400  # Convert back to seconds for model
        
        # For MessagesSent_Total, use integer steps
        elif feature_name == 'MessagesSent_Total':
            return st.number_input(
                feature_info['description'],
                min_value=int(feature_info['min_value']),
                max_value=int(feature_info['max_value']),
                value=int(feature_info['min_value']),
                step=1,
                help=feature_info.get('description', '')
            )
        
        # For other numeric fields
        else:
            return st.number_input(
                feature_info['description'],
                min_value=float(feature_info['min_value']),
                max_value=float(feature_info['max_value']),
                value=float(feature_info['min_value']),
                step=0.1,
                help=feature_info.get('description', '')
            )
    
    elif feature_info['type'] == 'categorical':
        if 'value_mapping' in feature_info:
            # Create a selectbox with mapped values
            options = list(feature_info['value_mapping'].values())
            selected_label = st.selectbox(
                feature_info['description'],
                options=options,
                help=feature_info.get('description', '')
            )
            # Return the corresponding encoded value
            return next(k for k, v in feature_info['value_mapping'].items() if v == selected_label)
        else:
            # For Country_Encoded, show a more user-friendly input
            if feature_name == 'Country_Encoded':
                return st.number_input(
                    "Country Code",
                    min_value=1,
                    max_value=50000,
                    value=1,
                    step=1,
                    help="Enter the country code (1-50000)"
                )
            # For other encoded features without mapping
            return st.number_input(
                feature_info['description'],
                value=0,
                help=feature_info.get('description', '')
            )

def validate_user_input(user_input: Dict[str, Any], feature_mapping: Dict[str, Any]) -> bool:
    """Validate that all input values are within expected ranges"""
    for feature_name, value in user_input.items():
        feature_info = feature_mapping[feature_name]
        
        if feature_info['type'] == 'numeric':
            if not (feature_info['min_value'] <= value <= feature_info['max_value']):
                st.error(f"Invalid value for {feature_info['description']}. Must be between {feature_info['min_value']} and {feature_info['max_value']}")
                return False
        
        elif feature_info['type'] == 'categorical':
            if 'value_mapping' in feature_info:
                if value not in feature_info['value_mapping']:
                    st.error(f"Invalid value for {feature_info['description']}")
                    return False
    
    return True

def create_gauge_chart(probability):
    """Create a gauge chart for spam probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Spam Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    return fig

def predict_spammer(features: Dict[str, float]) -> Dict[str, Any]:
    """Make a prediction using the API"""
    try:
        response = requests.post(f"{API_URL}/predict", json={"features": features})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

def main():
    st.title("🛡️ Fiverr Spammer Detection System")
    st.write("Enter user details to predict if the account is likely to be a spammer.")
    
    # Load feature mapping and model metrics
    feature_mapping, model_metrics = load_model_and_mapping()
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    # Initialize user_input dictionary
    user_input = {}
    
    # Group features by type for better organization
    numeric_features = {k: v for k, v in feature_mapping.items() if v['type'] == 'numeric'}
    categorical_features = {k: v for k, v in feature_mapping.items() if v['type'] == 'categorical'}
    boolean_features = {k: v for k, v in feature_mapping.items() if v['type'] == 'boolean'}
    
    # Create input fields with proper organization
    with col1:
        st.subheader("Account Metrics")
        for field, props in numeric_features.items():
            user_input[field] = create_feature_input(field, props)
    
    with col2:
        st.subheader("Account Information")
        for field, props in categorical_features.items():
            user_input[field] = create_feature_input(field, props)
    
    with col3:
        st.subheader("Behavioral Flags")
        for field, props in boolean_features.items():
            user_input[field] = create_feature_input(field, props)
    
    # Add a predict button
    if st.button("Predict", type="primary"):
        try:
            # Validate input before prediction
            if not validate_user_input(user_input, feature_mapping):
                st.error("Please correct the input values before proceeding.")
                return
            
            # Make prediction using API
            result = predict_spammer(user_input)
            
            if result:
                # Display results
                st.header("Prediction Results")
                
                # Create columns for results
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.subheader("Prediction")
                    if result["is_spammer"]:
                        st.error("⚠️ Potential Spammer Detected!")
                        
                        if result["risk_factors"]:
                            st.warning("Risk Factors Detected:")
                            for factor in result["risk_factors"]:
                                st.markdown(f"- {factor}")
                    else:
                        st.success("✅ Legitimate User")
                
                with result_col2:
                    st.plotly_chart(create_gauge_chart(result['probability']))
                
                # Display feature importance
                st.subheader("Feature Importance")
                try:
                    feature_importance = pd.read_csv('../reports/feature_importance.csv')
                    st.bar_chart(feature_importance.set_index('feature'))
                except Exception as e:
                    st.warning("Could not load feature importance visualization")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Batch prediction section
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Check if all required features are present
            missing_features = [f for f in feature_mapping.keys() if f not in df.columns]
            if missing_features:
                st.error(f"Missing features in CSV: {missing_features}")
            else:
                if st.button("Predict Batch"):
                    # Prepare batch request
                    batch_data = df[list(feature_mapping.keys())].to_dict('records')
                    batch_request = [{"features": row} for row in batch_data]
                    
                    try:
                        # Make batch prediction
                        response = requests.post(f"{API_URL}/predict/batch", json=batch_request)
                        response.raise_for_status()
                        results = response.json()
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame(results)
                        results_df = pd.concat([df, results_df], axis=1)
                        
                        # Display results
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            csv,
                            "batch_predictions.csv",
                            "text/csv"
                        )
                    except Exception as e:
                        st.error(f"Batch prediction failed: {str(e)}")
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
    
    # Add information about the model
    with st.expander("About the Model"):
        st.write("""
        This spammer detection model uses Random Forest, trained on Fiverr user data.
        The model takes into account various user behaviors and account characteristics to identify potential spammers.
        
        Features are grouped into three categories:
        - Account Metrics: Numerical values about the account
        - Account Information: Categorical values about the account
        - Behavioral Flags: Indicators of suspicious behavior
        """)
    
    # Add footer
    st.markdown("---")
    st.markdown("Built with Streamlit • Model: Random Forest • Data: Fiverr User Behavior")

if __name__ == "__main__":
    main() 