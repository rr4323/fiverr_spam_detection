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

def create_feature_mapping():
    """Create a mapping of encoded features to their actual names and categories."""
    return {
        'X1': {
            'name': 'AccountAge_Seconds',
            'type': 'numeric',
            'category': 'Account Metrics',
            'description': 'Account age in seconds',
            'importance': 0.0215
        },
        'X2': {
            'name': 'MessagesSent_Total',
            'type': 'numeric',
            'category': 'Account Metrics',
            'description': 'Total number of messages sent',
            'importance': 0.0364
        },
        'X3': {
            'name': 'AccountLevel_Encoded',
            'type': 'categorical',
            'category': 'Account Information',
            'description': 'Account level encoded',
            'mapping': {
                '36': 'Level 2 Seller',
                '71': 'Level 1 Seller',
                '51': 'New Seller',
                '28': 'Top Rated Seller',
                '13': 'Rising Talent'
            },
            'importance': 0.0192
        },
        'X4': {
            'name': 'ReferrerType_Encoded',
            'type': 'categorical',
            'category': 'Account Information',
            'description': 'Type of referrer encoded',
            'mapping': {
                '1': 'Direct',
                '2': 'Search',
                '3': 'Social Media'
            },
            'importance': 0.0450
        },
        'X5': {
            'name': 'CountryTier_Encoded',
            'type': 'categorical',
            'category': 'Account Information',
            'description': 'Country tier encoded',
            'mapping': {
                '9': 'Tier 1 (High Trust)',
                '11': 'Tier 2',
                '8': 'Tier 3',
                '16': 'Tier 4 (Low Trust)'
            },
            'importance': 0.0456
        },
        'X6': {
            'name': 'ProfileCompletionTier_Encoded',
            'type': 'categorical',
            'category': 'Account Information',
            'description': 'Profile completion tier encoded',
            'mapping': {
                '1': 'Basic',
                '2': 'Complete'
            },
            'importance': 0.0611
        },
        'X7': {
            'name': 'VerificationLevel_Encoded',
            'type': 'categorical',
            'category': 'Account Information',
            'description': 'Level of verification encoded',
            'mapping': {
                '21': 'Basic Verification',
                '7': 'ID Verified',
                '15': 'Payment Verified',
                '2': 'Phone Verified'
            },
            'importance': 0.0649
        },
        'X8': {
            'name': 'LoginCount_Recent',
            'type': 'numeric',
            'category': 'Account Metrics',
            'description': 'Number of recent logins',
            'importance': 0.0555
        },
        'X9': {
            'name': 'Country_Encoded',
            'type': 'categorical',
            'category': 'Account Information',
            'description': 'Country encoded',
            'mapping': {
                '1': 'United States',
                '2': 'United Kingdom',
                '3': 'Canada',
                '4': 'Australia',
                '5': 'Germany',
                '6': 'France',
                '7': 'India',
                '8': 'Pakistan',
                '9': 'Bangladesh',
                '10': 'Philippines',
                '11': 'Nigeria',
                '12': 'Kenya',
                '13': 'South Africa',
                '14': 'Other European',
                '15': 'Other Asian',
                '16': 'Other'
            },
            'importance': 0.0223
        },
        'X10': {
            'name': 'HasProfilePic',
            'type': 'categorical',
            'category': 'Profile Features',
            'description': 'Whether user has profile picture',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X11': {
            'name': 'HasDescription',
            'type': 'categorical',
            'category': 'Profile Features',
            'description': 'Whether user has profile description',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X12': {
            'name': 'EmailVerified',
            'type': 'categorical',
            'category': 'Profile Features',
            'description': 'Whether email is verified',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X13': {
            'name': 'PhoneVerified',
            'type': 'categorical',
            'category': 'Profile Features',
            'description': 'Whether phone is verified',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X14': {
            'name': 'SkillsListed_Count',
            'type': 'numeric',
            'category': 'Profile Features',
            'description': 'Number of skills listed',
            'importance': 0.0
        },
        'X15': {
            'name': 'PortfolioItems_Count',
            'type': 'numeric',
            'category': 'Profile Features',
            'description': 'Number of portfolio items',
            'importance': 0.0
        },
        'X16': {
            'name': 'GigsActive_Count',
            'type': 'numeric',
            'category': 'Account Metrics',
            'description': 'Number of active gigs',
            'importance': 0.0207
        },
        'X17': {
            'name': 'ReviewsReceived_Count',
            'type': 'numeric',
            'category': 'Account Metrics',
            'description': 'Number of reviews received',
            'importance': 0.0375
        },
        'X18': {
            'name': 'OrdersCompleted_Count',
            'type': 'numeric',
            'category': 'Account Metrics',
            'description': 'Number of completed orders',
            'importance': 0.0231
        },
        'X19': {
            'name': 'AvgRating_Encoded',
            'type': 'categorical',
            'category': 'Account Information',
            'description': 'Average rating encoded',
            'mapping': {
                '10': '5.0 Stars',
                '9': '4.5-4.9 Stars',
                '8': '4.0-4.4 Stars',
                '7': '3.5-3.9 Stars',
                '6': '3.0-3.4 Stars',
                '5': '2.5-2.9 Stars',
                '4': '2.0-2.4 Stars',
                '3': '1.5-1.9 Stars',
                '2': '1.0-1.4 Stars',
                '1': '0.5-0.9 Stars'
            },
            'importance': 0.1773
        },
        'X20': {
            'name': 'LoginFromSuspiciousIP',
            'type': 'categorical',
            'category': 'Security Flags',
            'description': 'Login detected from suspicious IP',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X21': {
            'name': 'SentLink_InMsg',
            'type': 'categorical',
            'category': 'Message Behavior',
            'description': 'Sent links in messages',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0288
        },
        'X22': {
            'name': 'MentionsOffPlatformApp',
            'type': 'categorical',
            'category': 'Message Behavior',
            'description': 'Mentions off-platform applications',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0835
        },
        'X23': {
            'name': 'FlaggedByOtherUser',
            'type': 'categorical',
            'category': 'Security Flags',
            'description': 'User was flagged by other users',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0309
        },
        'X24': {
            'name': 'UsernameHasNumbers',
            'type': 'categorical',
            'category': 'Account Behavior',
            'description': 'Username contains numbers',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X25': {
            'name': 'UsernameHasExcessiveSpecialChars',
            'type': 'categorical',
            'category': 'Account Behavior',
            'description': 'Username contains excessive special characters',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0334
        },
        'X26': {
            'name': 'UsedDisposableEmail',
            'type': 'categorical',
            'category': 'Account Behavior',
            'description': 'Used a disposable email service',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X27': {
            'name': 'PaymentVerified_Flag',
            'type': 'categorical',
            'category': 'Account Information',
            'description': 'Payment method is verified',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X28': {
            'name': 'AsksForEmail_InMsg',
            'type': 'categorical',
            'category': 'Message Behavior',
            'description': 'Asks for email in messages',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X29': {
            'name': 'AsksForPayment_OffPlatform',
            'type': 'categorical',
            'category': 'Message Behavior',
            'description': 'Asks for payment outside platform',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X30': {
            'name': 'SentShortenedLink_InMsg',
            'type': 'categorical',
            'category': 'Message Behavior',
            'description': 'Sent shortened links in messages',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X31': {
            'name': 'AsksToOpenLink_Urgent',
            'type': 'categorical',
            'category': 'Message Behavior',
            'description': 'Urgently asks to open links',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X32': {
            'name': 'MentionsAttachment_InMsg',
            'type': 'categorical',
            'category': 'Message Behavior',
            'description': 'Mentions attachments in messages',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X33': {
            'name': 'UsedUrgentLanguage_InMsg',
            'type': 'categorical',
            'category': 'Message Behavior',
            'description': 'Uses urgent language in messages',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X34': {
            'name': 'ImpersonationAttempt_InMsg',
            'type': 'categorical',
            'category': 'Message Behavior',
            'description': 'Attempts impersonation in messages',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X35': {
            'name': 'AsksForFinancialDetails_OffPlatform',
            'type': 'categorical',
            'category': 'Security Flags',
            'description': 'Requested financial information',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X36': {
            'name': 'AsksForCredentials_OffPlatform',
            'type': 'categorical',
            'category': 'Security Flags',
            'description': 'Requested login credentials',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0350
        },
        'X37': {
            'name': 'MentionsOffPlatformPaymentMethod',
            'type': 'categorical',
            'category': 'Security Flags',
            'description': 'Mentioned alternative payment methods',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X38': {
            'name': 'UsedTemporaryOnlinePhoneNumber',
            'type': 'categorical',
            'category': 'Account Behavior',
            'description': 'Used temporary phone number',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X39': {
            'name': 'VeryShortInitialMessage',
            'type': 'categorical',
            'category': 'Message Behavior',
            'description': 'Sent very short initial messages',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0206
        },
        'X40': {
            'name': 'UsedGenericSpamTemplate',
            'type': 'categorical',
            'category': 'Message Behavior',
            'description': 'Used generic spam message templates',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X41': {
            'name': 'AsksForPersonalInfo',
            'type': 'categorical',
            'category': 'Security Flags',
            'description': 'Requested personal information',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X42': {
            'name': 'ContactedUnsolicited',
            'type': 'categorical',
            'category': 'Message Behavior',
            'description': 'Contacted users without prior interaction',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X43': {
            'name': 'RapidMessagingDetected',
            'type': 'categorical',
            'category': 'Message Behavior',
            'description': 'Sent many messages in short time',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X44': {
            'name': 'AttemptedSuspiciousAction',
            'type': 'categorical',
            'category': 'Security Flags',
            'description': 'Attempted suspicious actions',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X45': {
            'name': 'VagueJobDescriptionPosted',
            'type': 'categorical',
            'category': 'Message Behavior',
            'description': 'Posted vague job descriptions',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X46': {
            'name': 'IndiscriminateApplicationsSent',
            'type': 'categorical',
            'category': 'Message Behavior',
            'description': 'Applied to many jobs without reading',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X47': {
            'name': 'IsKnownBotOrHeuristic',
            'type': 'categorical',
            'category': 'Bot Detection',
            'description': 'Detected bot-like behavior',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X48': {
            'name': 'HeadlessBrowser',
            'type': 'categorical',
            'category': 'Bot Detection',
            'description': 'Used headless browser',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X49': {
            'name': 'SuspectedRobotUser',
            'type': 'categorical',
            'category': 'Bot Detection',
            'description': 'Suspected automated account',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0210
        },
        'X50': {
            'name': 'CaptchaDefeatedByBot',
            'type': 'categorical',
            'category': 'Bot Detection',
            'description': 'Bypassed security checks',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        },
        'X51': {
            'name': 'OtherBehaviorFlag_5',
            'type': 'categorical',
            'category': 'Other Flags',
            'description': 'Other suspicious behavior (5)',
            'ui': {
                'type': 'selectbox',
                'options': ['No', 'Yes'],
                'default': 'No'
            },
            'importance': 0.0
        }
    }

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
    feature_mapping = create_feature_mapping()
    
    # Save feature mapping
    models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(feature_mapping, os.path.join(models_dir, 'feature_mapping.pkl'))
    
    print("Feature mapping created and saved successfully!")
    print("\nFeature mapping structure:")
    for encoded_col, info in feature_mapping.items():
        print(f"\n{encoded_col} -> {info['description']}")
        print(f"Type: {info['type']}")
        if 'min_value' in info:
            print(f"Range: {info['min_value']} to {info['max_value']}")
        if 'categories' in info:
            print(f"Categories: {info['categories']}")

if __name__ == "__main__":
    main() 