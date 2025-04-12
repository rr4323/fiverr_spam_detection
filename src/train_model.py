import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import json
from typing import Dict, Any

def load_data(file_path):
    """
    Load and provide initial analysis of the dataset
    """
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print("\nSample of first few rows:")
    print(df.head())
    print("\nClass distribution:")
    print(df['label'].value_counts(normalize=True))
    
    print("\nMissing values summary:")
    print(df.isnull().sum())
    return df

def preprocess_data(df):
    """
    Preprocess the data by handling missing values and scaling features
    """
    print("\nPreprocessing data...")
    # Store user_ids for later use
    user_ids = df['user_id']
    
    # Separate features and target
    X = df.drop(['label', 'user_id'], axis=1)
    y = df['label']
    
    # Handle missing values using median imputation
    print("\nHandling missing values...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Scale features to have zero mean and unit variance
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Quality check for NaN values
    if np.isnan(X_scaled).any():
        raise ValueError("There are still NaN values after preprocessing!")
    
    return X_scaled, y, user_ids, scaler, X.columns

def handle_class_imbalance(X_train, y_train):
    """
    Handle class imbalance using SMOTE
    """
    print("\nHandling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """
    Evaluate the model and return metrics
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('../reports/confusion_matrix.png')
    plt.close()
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    feature_importance.head(20).plot(kind='bar', x='feature', y='importance')
    plt.title('Top 20 Feature Importances')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../reports/feature_importance.png')
    plt.close()
    
    # Save feature importance to CSV
    feature_importance.to_csv('../reports/feature_importance.csv', index=False)
    
    return metrics, model

def save_model_artifacts(model, scaler, feature_names, metrics):
    """
    Save model artifacts for production
    """
    # Create models directory if it doesn't exist
    models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    joblib.dump(model, os.path.join(models_dir, 'spammer_detector.pkl'))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    
    # Save metrics
    with open(os.path.join(models_dir, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

def create_feature_mapping(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Create feature mapping based on dataset characteristics"""
    feature_mapping = {}
    
    # Define mappings for encoded values
    account_level_mapping = {
        36: "Level 2 Seller",
        71: "Level 1 Seller",
        51: "New Seller",
        28: "Top Rated Seller",
        13: "Rising Talent"
    }
    
    country_tier_mapping = {
        9: "Tier 1 (High Trust)",
        11: "Tier 2",
        8: "Tier 3",
        16: "Tier 4 (Low Trust)"
    }
    
    verification_level_mapping = {
        21: "Basic Verification",
        7: "ID Verified",
        15: "Payment Verified",
        2: "Phone Verified"
    }
    
    # Account metrics (numeric)
    numeric_features = [
        'AccountAge_Seconds',
        'MessagesSent_Total',
        'LoginCount_Recent',
        'SkillsListed_Count',
        'PortfolioItems_Count',
        'GigsActive_Count',
        'ReviewsReceived_Count',
        'OrdersCompleted_Count'
    ]
    
    # Encoded categorical features with mappings
    categorical_features = {
        'AccountLevel_Encoded': {
            'mapping': account_level_mapping,
            'description': 'Seller level on Fiverr'
        },
        'ReferrerType_Encoded': {
            'mapping': {
                1: "Direct",
                2: "Search",
                3: "Social Media"
            },
            'description': 'How the user found Fiverr'
        },
        'CountryTier_Encoded': {
            'mapping': country_tier_mapping,
            'description': 'Country trust tier'
        },
        'ProfileCompletionTier_Encoded': {
            'mapping': {
                1: "Basic",
                2: "Complete"
            },
            'description': 'Profile completion level'
        },
        'VerificationLevel_Encoded': {
            'mapping': verification_level_mapping,
            'description': 'Account verification status'
        },
        'Country_Encoded': {
            'description': 'Country code',
            'is_encoded': True  # We'll keep this as encoded since there are many countries
        },
        'AvgRating_Encoded': {
            'mapping': {
                10: "5.0 Stars",
                9: "4.5-4.9 Stars",
                8: "4.0-4.4 Stars",
                7: "3.5-3.9 Stars",
                6: "3.0-3.4 Stars",
                5: "2.5-2.9 Stars",
                4: "2.0-2.4 Stars",
                3: "1.5-1.9 Stars",
                2: "1.0-1.4 Stars",
                1: "0.5-0.9 Stars"
            },
            'description': 'Average seller rating'
        }
    }
    
    # Boolean features with descriptions
    boolean_features = {
        'HasProfilePic': 'User has uploaded a profile picture',
        'HasDescription': 'User has filled out their profile description',
        'EmailVerified': 'Email address is verified',
        'PhoneVerified': 'Phone number is verified',
        'LoginFromSuspiciousIP': 'Recent login from suspicious IP address',
        'SentLink_InMsg': 'Sent external links in messages',
        'MentionsOffPlatformApp': 'Mentioned communication outside Fiverr',
        'FlaggedByOtherUser': 'Reported by other users',
        'UsernameHasNumbers': 'Username contains numbers',
        'UsernameHasExcessiveSpecialChars': 'Username has unusual characters',
        'UsedDisposableEmail': 'Used temporary email service',
        'PaymentVerified_Flag': 'Payment method is verified',
        'AsksForEmail_InMsg': 'Requested email address in messages',
        'AsksForPayment_OffPlatform': 'Requested payment outside Fiverr',
        'SentShortenedLink_InMsg': 'Sent shortened URLs in messages',
        'AsksToOpenLink_Urgent': 'Urgently requested to open links',
        'MentionsAttachment_InMsg': 'Mentioned attachments in messages',
        'UsedUrgentLanguage_InMsg': 'Used urgent/scare tactics in messages',
        'ImpersonationAttempt_InMsg': 'Attempted to impersonate someone',
        'AsksForFinancialDetails_OffPlatform': 'Requested financial information',
        'AsksForCredentials_OffPlatform': 'Requested login credentials',
        'MentionsOffPlatformPaymentMethod': 'Mentioned alternative payment methods',
        'UsedTemporaryOnlinePhoneNumber': 'Used temporary phone number',
        'VeryShortInitialMessage': 'Sent very short initial messages',
        'UsedGenericSpamTemplate': 'Used generic spam message templates',
        'AsksForPersonalInfo': 'Requested personal information',
        'ContactedUnsolicited': 'Contacted users without prior interaction',
        'RapidMessagingDetected': 'Sent many messages in short time',
        'AttemptedSuspiciousAction': 'Attempted suspicious actions',
        'VagueJobDescriptionPosted': 'Posted vague job descriptions',
        'IndiscriminateApplicationsSent': 'Applied to many jobs without reading',
        'IsKnownBotOrHeadlessBrowser': 'Detected bot-like behavior',
        'SuspectedRobotUser': 'Suspected automated account',
        'CaptchaDefeatedByBot': 'Bypassed security checks',
        'OtherBehaviorFlag_5': 'Other suspicious behavior (5)',
        'OtherBehaviorFlag_6': 'Other suspicious behavior (6)'
    }
    
    # Add numeric features
    for feature in numeric_features:
        if feature in df.columns:  # Check if feature exists in dataframe
            feature_mapping[feature] = {
                'type': 'numeric',
                'description': f'{feature.replace("_", " ")}',
                'min_value': float(df[feature].min()),
                'max_value': float(df[feature].max())
            }
    
    # Add categorical features with mappings
    for feature, info in categorical_features.items():
        if feature in df.columns:  # Check if feature exists in dataframe
            feature_mapping[feature] = {
                'type': 'categorical',
                'description': info['description'],
                'categories': sorted(df[feature].unique().tolist())
            }
            if 'mapping' in info:
                feature_mapping[feature]['value_mapping'] = info['mapping']
            if 'is_encoded' in info:
                feature_mapping[feature]['is_encoded'] = info['is_encoded']
    
    # Add boolean features with descriptions
    for feature, description in boolean_features.items():
        if feature in df.columns:  # Check if feature exists in dataframe
            feature_mapping[feature] = {
                'type': 'boolean',
                'description': description
            }
    
    return feature_mapping

def main():
    # Load and preprocess data
    df = load_data('../data/fiverr_data.csv')
    
    # Preprocess data
    X_scaled, y, user_ids, scaler, feature_names = preprocess_data(df)
    
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)
    
    # Initialize and train the best model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=42
    )
    
    # Evaluate model
    metrics, trained_model = evaluate_model(
        model, X_train_resampled, X_test, y_train_resampled, y_test, feature_names
    )
    
    # Save model artifacts
    save_model_artifacts(trained_model, scaler, feature_names, metrics)
    
    # Print final metrics
    print("\nFinal Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Create feature mapping
    feature_mapping = create_feature_mapping(df)
    
    # Save feature mapping
    models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(feature_mapping, os.path.join(models_dir, 'feature_mapping.pkl'))
    
    # Convert feature_names Index to list before saving
    feature_names_list = feature_names.tolist()
    
    # Save feature names
    with open(os.path.join(models_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names_list, f)

if __name__ == "__main__":
    main() 