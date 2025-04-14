import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List

# Set page config
st.set_page_config(
    page_title="Fiverr Spammer Detection",
    page_icon="🛡️",
    layout="wide"
)

# API configuration
API_URL = "http://localhost:8000"

@st.cache_resource
def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_URL}/model/info")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to get model information: {str(e)}")
        return {"metrics": {}, "feature_mapping": {}, "categories": []}

def create_feature_input(feature_name: str, feature_info: Dict[str, Any]) -> Any:
    """Create appropriate input widget based on feature type and mapping"""
    if feature_info['type'] == 'categorical':
        return st.checkbox(
            feature_info['description'],
            value=False,
            help=feature_info.get('description', '')
        )
    
    elif feature_info['type'] == 'numeric':
        # For AccountAge_Seconds, convert to days for better understanding
        if feature_name == 'X1':
            days = st.number_input(
                "Account Age (in days)",
                min_value=0,
                max_value=365,  # 1 year max
                value=0,
                step=1,
                help="Age of the account in days"
            )
            return days * 86400  # Convert back to seconds for model
        
        # For other numeric fields
        else:
            return st.number_input(
                feature_info['description'],
                min_value=float(feature_info.get('min_value', 0)),
                max_value=float(feature_info.get('max_value', 100)),
                value=float(feature_info.get('min_value', 0)),
                step=0.1,
                help=feature_info.get('description', '')
            )

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

def display_risk_factors(risk_factors: List[Dict[str, str]]):
    """Display risk factors in an organized way"""
    if not risk_factors:
        st.success("✅ No significant risk factors detected")
        return
    
    st.warning("⚠️ Risk Factors Detected:")
    
    # Group risk factors by category
    risk_factors_by_category = {}
    for factor in risk_factors:
        category = factor['category']
        if category not in risk_factors_by_category:
            risk_factors_by_category[category] = []
        risk_factors_by_category[category].append(factor)
    
    # Display risk factors by category
    for category, factors in risk_factors_by_category.items():
        with st.expander(f"🔍 {category.title()}"):
            for factor in factors:
                st.markdown(f"""
                - **{factor['description']}**  
                  Value: {factor['value']}  
                  Reason: {factor['reason']}
                """)

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
    
    # Get model information from API
    model_info = get_model_info()
    feature_mapping = model_info.get("feature_mapping", {})
    model_metrics = model_info.get("metrics", {})
    categories = model_info.get("categories", [])
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    # Initialize user_input dictionary
    user_input = {}
    
    # Group features by category for better organization
    features_by_category = {}
    for feature_name, feature_info in feature_mapping.items():
        category = feature_info.get('category', 'other')
        if category not in features_by_category:
            features_by_category[category] = []
        features_by_category[category].append((feature_name, feature_info))
    
    # Create input fields with proper organization
    with col1:
        st.subheader("Account Characteristics")
        for category in categories:
            if category in features_by_category:
                with st.expander(f"📊 {category.title()}"):
                    for feature_name, feature_info in features_by_category[category]:
                        user_input[feature_name] = create_feature_input(feature_name, feature_info)
    
    # Add a predict button
    if st.button("Predict", type="primary"):
        try:
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
                    else:
                        st.success("✅ Legitimate User")
                    
                    # Display risk factors
                    display_risk_factors(result["risk_factors"])
                
                with result_col2:
                    st.plotly_chart(create_gauge_chart(result['probability']))
                
                # Display model metrics
                st.subheader("Model Performance")
                metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index', columns=['Value'])
                st.dataframe(metrics_df)
        
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
                    try:
                        # Convert DataFrame to list of dictionaries
                        records = df.to_dict('records')
                        
                        # Make batch prediction using API
                        response = requests.post(f"{API_URL}/predict/batch", json=records)
                        response.raise_for_status()
                        results = response.json()
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="prediction_results.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Batch prediction failed: {str(e)}")
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")

if __name__ == "__main__":
    main() 