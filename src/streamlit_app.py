import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List
import time
import uuid
from datetime import datetime

# Set page config with dark theme for better visibility
st.set_page_config(
    page_title="Fiverr Spammer Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Fiverr Spammer Detection System - Powered by ML"
    }
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Center align headers */
    h1, h2, h3 {
        text-align: center;
        padding: 20px;
        color: #1f77b4;
    }
    
    /* Improved button styling */
    .stButton>button {
        width: 100%;
        margin-top: 20px;
        border-radius: 10px;
        height: 3em;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #135c8d;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
    
    /* Expander styling */
    .stExpander {
        background-color: rgba(31, 119, 180, 0.1);
        border-radius: 10px;
        margin: 10px 0;
    }
    
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1em;
        color: #1f77b4;
    }
    
    /* Improved spacing */
    div[data-testid="stVerticalBlock"] {
        gap: 25px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #0e1117;
        padding: 0px 10px;
        border-radius: 15px;
        overflow-x: auto;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: nowrap;
        background-color: transparent;
        border-radius: 15px;
        color: #4a9eff;
        padding: 10px 20px;
        font-size: 14px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(31, 119, 180, 0.2);
        color: white;
    }

    /* Hide scrollbar for tabs while allowing scroll */
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        -ms-overflow-style: none;
        scrollbar-width: none;
    }
    
    /* Input field styling */
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Warning and success message styling */
    .stAlert {
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Description text styling */
    .description-text {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin: 20px 0;
        padding: 20px;
        background-color: rgba(31, 119, 180, 0.1);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_URL = "http://localhost:8000"

# Cache model info for 1 hour
@st.cache_data(ttl=3600)
def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_URL}/model/info")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to get model information: {str(e)}")
        return {"metrics": {}, "feature_mapping": {}, "categories": []}

# Cache gauge chart creation
@st.cache_data
def create_gauge_chart(probability):
    """Create a gauge chart for spam probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Spam Probability", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
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
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'}
    )
    return fig

def create_feature_input(feature_name: str, feature_info: Dict[str, Any]) -> Any:
    """Create appropriate input widget based on feature type and mapping"""
    help_text = feature_info.get('description', '')
    if feature_info.get('importance', 0) > 0.05:
        help_text += " (High Impact Feature)"
    
    if feature_info['type'] == 'categorical':
        if feature_name == 'X6':  # Country Tier
            return st.selectbox(
                "Country Tier",
                options=[2, 3, 4],
                format_func=lambda x: {
                    2: "Tier 1 (High Trust)",
                    3: "Tier 2 (Medium Risk)",
                    4: "Tier 3 (High Risk)"
                }.get(x, str(x)),
                help="Select the country tier level",
                key=f"input_{feature_name}"
            )
        elif feature_name == 'X3':  # Seller Category
            return st.selectbox(
                "Seller Category",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "New Seller",
                    1: "Level 1",
                    2: "Level 2",
                    3: "Top Rated"
                }.get(x, str(x)),
                help="Select the seller category level",
                key=f"input_{feature_name}"
            )
        else:
            return st.number_input(
                feature_info['description'],
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help=help_text,
                key=f"input_{feature_name}"
            )
    
    elif feature_info['type'] == 'boolean':
        if feature_name in ['X15', 'X12']:  # Email verified, Link in message
            label = "Email Verified" if feature_name == 'X15' else "Contains Links in Messages"
            help_text = "Whether the email is verified" if feature_name == 'X15' else "Whether messages contain links"
            return st.checkbox(
                label,
                value=False,
                help=help_text,
                key=f"input_{feature_name}"
            )
    
    elif feature_info['type'] == 'numeric':
        if feature_name == 'X1':  # Account Age
            total_seconds = st.slider(
                "Account Age (second)",
                min_value=0,
                max_value=100000,
                value=0,
                help="Account age in seconds ",
                key=f"input_{feature_name}"
            )
            

            
            return total_seconds
        
        elif feature_name == 'X18':  # Incomplete Order
            return st.slider(
                "Incomplete Orders",
                min_value=3,
                max_value=13,
                value=3,
                help="Number of incomplete orders (range: 3-13)",
                key=f"input_{feature_name}"
            )
        elif feature_name == 'X17':  # Profile Completion Score
            return st.slider(
                "Profile Completion Score",
                min_value=1,
                max_value=8,
                value=1,
                help="Profile completion score (range: 1-8)",
                key=f"input_{feature_name}"
            )
        elif feature_name == 'X16':  # Message in Day
            return st.slider(
                "Messages per Day",
                min_value=3,
                max_value=18,
                value=3,
                help="Average messages sent per day (range: 3-18)",
                key=f"input_{feature_name}"
            )
        elif feature_name == 'X8':  # Total Verified Review
            value = st.slider(
                "Total Verified Reviews",
                min_value=4,
                max_value=25,
                value=4,
                help="Number of verified reviews (range: 4-25). Note: 5-10 reviews may indicate higher spam risk",
                key=f"input_{feature_name}"
            )
            if 5 <= value <= 10:
                st.warning("⚠️ This review count range (5-10) is associated with higher spam risk")
            return value
        else:
            return st.slider(
                feature_info['description'],
                min_value=0,
                max_value=100,
                value=0,
                help=help_text,
                key=f"input_{feature_name}"
            )

def display_risk_factors(risk_factors: List[Dict[str, str]]):
    """Display risk factors in an organized way"""
    if not risk_factors:
        st.success("✅ No significant risk factors detected")
        return
    
    st.warning("⚠️ Risk Factors Detected")
    
    # Group risk factors by category
    risk_factors_by_category = {}
    for factor in risk_factors:
        category = factor['category']
        if category not in risk_factors_by_category:
            risk_factors_by_category[category] = []
        risk_factors_by_category[category].append(factor)
    
    # Display risk factors by category with improved styling
    for category, factors in risk_factors_by_category.items():
        with st.expander(f"🔍 {category.title()}", expanded=True):
            for factor in factors:
                # Determine risk level color
                if 'high' in factor['reason'].lower():
                    color = 'rgba(255,0,0,0.1)'
                elif 'medium' in factor['reason'].lower():
                    color = 'rgba(255,255,0,0.1)'
                else:
                    color = 'rgba(255,255,255,0.05)'
                
                st.markdown(f"""
                <div style='background-color: {color}; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                    <strong>{factor['description']}</strong><br/>
                    Value: {factor['value']}<br/>
                    Reason: {factor['reason']}
                </div>
                """, unsafe_allow_html=True)

@st.cache_data(ttl=60)
def predict_spammer(features: Dict[str, float]) -> Dict[str, Any]:
    """Make a prediction using the API"""
    try:
        cleaned_features = {k: float(v if v is not None else 0) for k, v in features.items()}
        response = requests.post(f"{API_URL}/predict", json={"features": cleaned_features})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

def create_risk_matrix(risk_factors: List[Dict[str, str]]) -> pd.DataFrame:
    """Create a risk matrix DataFrame"""
    if not risk_factors:
        return pd.DataFrame()
    
    risk_df = pd.DataFrame(risk_factors)
    risk_matrix = pd.pivot_table(
        risk_df,
        values='feature',
        index='category',
        aggfunc='count',
        fill_value=0
    ).rename(columns={'feature': 'Risk Count'})
    
    # Add tooltips for risk levels
    risk_matrix['Risk Level'] = risk_matrix['Risk Count'].apply(
        lambda x: 'High' if x > 5 else 'Medium' if x > 2 else 'Low'
    )
    
    return risk_matrix

def create_feedback_section(prediction_result: Dict[str, Any]):
    """Create feedback collection UI"""
    st.markdown("### Feedback")
    st.markdown("Help improve the model by providing feedback on this prediction")
    
    # Store prediction ID in session state
    if 'prediction_id' not in st.session_state:
        st.session_state.prediction_id = str(uuid.uuid4())
    
    # Feedback form
    with st.form("feedback_form"):
        is_correct = st.radio(
            "Was this prediction correct?",
            ["Yes", "No"],
            horizontal=True
        )
        
        actual_label = None
        if is_correct == "No":
            actual_label = st.radio(
                "What was the correct classification?",
                ["Legitimate User", "Spammer"],
                horizontal=True
            )
        
        feedback_notes = st.text_area(
            "Additional notes (optional)",
            help="Provide any additional context about this prediction"
        )
        
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            try:
                feedback_data = {
                    "prediction_id": st.session_state.prediction_id,
                    "is_correct": is_correct == "Yes",
                    "actual_label": actual_label == "Spammer" if actual_label else None,
                    "feedback_notes": feedback_notes
                }
                
                response = requests.post(
                    f"{API_URL}/feedback",
                    json=feedback_data
                )
                response.raise_for_status()
                
                st.success("Thank you for your feedback! It will help improve the model.")
                
                # Clear prediction ID for next prediction
                del st.session_state.prediction_id
                
            except Exception as e:
                st.error(f"Failed to submit feedback: {str(e)}")

def display_feedback_stats():
    """Display feedback statistics"""
    try:
        response = requests.get(f"{API_URL}/model/feedback/stats")
        response.raise_for_status()
        stats = response.json()
        
        st.markdown("### Model Feedback Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Feedback", stats["total_feedback"])
        
        with col2:
            st.metric(
                "Accuracy",
                f"{stats['accuracy']:.1%}",
                help="Based on user feedback"
            )
        
        with col3:
            last_retrain = datetime.fromisoformat(stats["last_retrain"])
            st.metric(
                "Last Retrain",
                last_retrain.strftime("%Y-%m-%d %H:%M"),
                help="Last model retraining time"
            )
        
    except Exception as e:
        st.error(f"Failed to load feedback statistics: {str(e)}")

def main():
    # Center-aligned title with icon and subtitle
    st.markdown("""
        <h1 style='text-align: center; color: #4a9eff; margin-bottom: 10px; font-size: 2.5em;'>
            🛡️ Fiverr Spammer Detection System
        </h1>
        <div class="description-text">
            Enter user details below to analyze potential spam activity
        </div>
    """, unsafe_allow_html=True)
    
    # Get model information from API with loading indicator
    with st.spinner("Loading model information..."):
        model_info = get_model_info()
    
    feature_mapping = model_info.get("feature_mapping", {})
    model_metrics = model_info.get("metrics", {})
    categories = model_info.get("categories", [])
    
    # Create two equal columns for form and predictions
    form_col, pred_col = st.columns(2)
    
    # Initialize session state for predictions if not exists
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    # Group features by category and sort by importance
    features_by_category = {}
    for feature_name, feature_info in feature_mapping.items():
        category = feature_info.get('category', 'other')
        if category not in features_by_category:
            features_by_category[category] = []
        features_by_category[category].append((feature_name, feature_info))
    
    # Sort features by importance within each category
    for category in features_by_category:
        features_by_category[category].sort(
            key=lambda x: x[1].get('importance', 0),
            reverse=True
        )
    
    # Create input fields with proper organization
    with form_col:
        st.markdown("<h2 style='text-align: center; color: #4a9eff; margin-bottom: 20px;'>Account Characteristics</h2>", unsafe_allow_html=True)
        
        # Create tabs with better naming
        tab_names = {
            "account_behavior": "👤 Account Behavior",
            "account_information": "ℹ️ Basic Info",
            "account_metrics": "📊 Metrics",
            "bot_detection": "🤖 Bot Detection",
            "composite_risk_score": "⚠️ Risk Score",
            "message_behavior": "💬 Messages",
            "other_flags": "🏁 Other Flags",
            "profile_features": "👥 Profile",
            "security_flags": "🔒 Security"
        }
        
        tabs = st.tabs([tab_names.get(category.lower().replace(" ", "_"), category.title()) 
                       for category in categories if category in features_by_category])
        
        user_input = {}
        for tab, category in zip(tabs, [cat for cat in categories if cat in features_by_category]):
            with tab:
                for feature_name, feature_info in features_by_category[category]:
                    user_input[feature_name] = create_feature_input(feature_name, feature_info)
        
        # Add predict button with loading state
        if st.button("Predict", type="primary", key="predict_button"):
            with st.spinner("Analyzing account..."):
                result = predict_spammer(user_input)
                if result:
                    st.session_state.prediction_made = True
                    st.session_state.prediction_result = result
    
    # Display results in prediction column
    with pred_col:
        if st.session_state.get('prediction_made', False):
            result = st.session_state.prediction_result
            st.subheader("Prediction Results")
            
            # Display prediction status with animation
            if result["is_spammer"]:
                st.error("⚠️ Potential Spammer Detected!")
            else:
                st.success("✅ Legitimate User")
            
            # Display probability gauge
            st.plotly_chart(
                create_gauge_chart(result['probability']),
                use_container_width=True,
                config={'displayModeBar': False}
            )
            
            # Create tabs for different result sections
            risk_tab, matrix_tab, metrics_tab = st.tabs(["Risk Factors", "Risk Matrix", "Model Metrics"])
            
            with risk_tab:
                display_risk_factors(result["risk_factors"])
            
            with matrix_tab:
                risk_matrix = create_risk_matrix(result["risk_factors"])
                if not risk_matrix.empty:
                    st.dataframe(
                        risk_matrix.style.background_gradient(cmap='YlOrRd'),
                        use_container_width=True
                    )
                else:
                    st.info("No risk factors to display in matrix")
            
            with metrics_tab:
                metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index', columns=['Value'])
                st.dataframe(
                    metrics_df.style.format("{:.4f}"),
                    use_container_width=True
                )
            
            # Add feedback section
            create_feedback_section(result)
            
            # Add feedback stats section
            display_feedback_stats()

if __name__ == "__main__":
    main() 