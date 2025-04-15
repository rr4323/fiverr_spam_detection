import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib
import os
import json
from typing import Dict, Any, Tuple

def create_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create composite features to enhance spam detection
    """
    print("\nCreating composite features...")
    
    # Create a copy to avoid modifying the original dataframe
    df_composite = df.copy()
    
    # Account risk score
    df_composite['X52'] = (
        (df_composite['X1'] < 86400).astype(int) * 5 +  # New account (< 1 day)
        (df_composite['X2'] > 100).astype(int) * 3 +    # High message count
        (df_composite['X19'] == 1).astype(int) * 5 +    # Lowest rating
        (df_composite['X20'] == 3).astype(int) * 3      # Highest risk level
    )
    
    # Message behavior risk
    df_composite['X53'] = (
        (df_composite['X21'] > 10).astype(int) * 2 +    # Many links
        (df_composite['X22'] > 10).astype(int) * 2 +    # Many off-platform mentions
        df_composite[['X28', 'X29', 'X30', 'X31', 'X33']].sum(axis=1)  # Suspicious message behaviors
    )
    
    # Verification risk
    df_composite['X54'] = (
        (df_composite['X12'] == 0).astype(int) +        # Email not verified
        (df_composite['X13'] == 0).astype(int) +        # Phone not verified
        (df_composite['X27'] == 0).astype(int)          # Payment not verified
    )
    
    return df_composite

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and provide initial analysis of the dataset
    """
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Add composite features
    df = create_composite_features(df)
    
    print(f"Dataset shape: {df.shape}")
    print("\nClass distribution:")
    print(df['label'].value_counts(normalize=True))
    
    print("\nMissing values summary:")
    print(df.isnull().sum())
    return df

def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, StandardScaler, pd.Index]:
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
    
    # Scale features using MinMaxScaler for better handling of extreme values
    print("Scaling features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Quality check for NaN values
    if np.isnan(X_scaled).any():
        raise ValueError("There are still NaN values after preprocessing!")
    
    return X_scaled, y, user_ids, scaler, X.columns

def handle_class_imbalance(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle class imbalance using SMOTE-Tomek
    """
    print("\nHandling class imbalance with SMOTE-Tomek...")
    smt = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def evaluate_model(model: RandomForestClassifier, 
                  X_train: np.ndarray, 
                  X_test: np.ndarray, 
                  y_train: np.ndarray, 
                  y_test: np.ndarray, 
                  feature_names: pd.Index,
                  threshold: float = 0.3) -> Tuple[Dict[str, float], RandomForestClassifier]:
    """
    Evaluate the model and return metrics
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions with custom threshold
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Decision Threshold': threshold
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

def save_model_artifacts(model: RandomForestClassifier, 
                        scaler: StandardScaler, 
                        feature_names: pd.Index, 
                        metrics: Dict[str, float], 
                        threshold: float = 0.3) -> None:
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
    
    # Save metrics and threshold
    model_info = {
        'metrics': metrics,
        'threshold': threshold,
        'feature_names': list(feature_names)
    }
    with open(os.path.join(models_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=4)

def create_feature_mapping():
    """Create a mapping of encoded features to their actual names and categories."""
    mapping = {
        'X1': {
            'name': 'AccountAge_Seconds',
            'type': 'numeric',
            'category': 'Account Metrics',
            'description': 'Account age in seconds',
            'importance': 0.0761
        },
        'X2': {
            'name': 'MessagesSent_Total',
            'type': 'numeric',
            'category': 'Account Metrics',
            'description': 'Total number of messages sent',
            'importance': 0.0680
        },
        'X3': {
            'name': 'SellerCategory',
            'type': 'categorical',
            'category': 'Account Information',
            'description': 'Seller category encoded',
            'importance': 0.0192
        },
        'X4': {
            'name': 'ReferrerType_Encoded',
            'type': 'categorical',
            'category': 'Account Information',
            'description': 'Type of referrer encoded',
            'importance': 0.0450
        },
        'X5': {
            'name': 'CountryTier_Encoded',
            'type': 'categorical',
            'category': 'Account Information',
            'description': 'Country tier encoded',
            'mapping': {
                '2': 'Tier 1 (High Trust)',
                '3': 'Tier 2 (Medium Risk)',
                '4': 'Tier 3 (High Risk)'
            },
            'importance': 0.0456
        },
        'X6': {
            'name': 'CountryTier',
            'type': 'categorical',
            'category': 'Account Information',
            'description': 'Country risk tier',
            'mapping': {
                '2': 'Tier 1 (High Trust)',
                '3': 'Tier 2 (Medium Risk)',
                '4': 'Tier 3 (High Risk)'
            },
            'importance': 0.0611
        },
        'X7': {
            'name': 'VerificationLevel_Encoded',
            'type': 'categorical',
            'category': 'Account Information',
            'description': 'Level of verification encoded',
            'importance': 0.0649
        },
        'X8': {
            'name': 'TotalVerifiedReview',
            'type': 'numeric',
            'category': 'Account Metrics',
            'description': 'Total number of verified reviews',
            'importance': 0.0555
        },
        'X9': {
            'name': 'Country_Encoded',
            'type': 'categorical',
            'category': 'Account Information',
            'description': 'Country encoded',
            'importance': 0.0223
        },
        'X10': {
            'name': 'ProfilePic_Indicator',
            'type': 'boolean',
            'category': 'Profile Features',
            'description': 'Whether user has profile picture',
            'importance': 0.0
        },
        'X11': {
            'name': 'AskedPersonalInfo',
            'type': 'boolean',
            'category': 'Message Behavior',
            'description': 'Whether user asked for personal information',
            'importance': 0.0
        },
        'X12': {
            'name': 'LinkInMessage',
            'type': 'boolean',
            'category': 'Message Behavior',
            'description': 'Whether message contains links',
            'importance': 0.0
        },
        'X13': {
            'name': 'PhoneVerified_Flag',
            'type': 'boolean',
            'category': 'Profile Features',
            'description': 'Whether phone is verified',
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
            'name': 'EmailVerified',
            'type': 'boolean',
            'category': 'Profile Features',
            'description': 'Whether email is verified',
            'importance': 0.0
        },
        'X16': {
            'name': 'MessageInDay',
            'type': 'numeric',
            'category': 'Account Metrics',
            'description': 'Number of messages sent in a day',
            'importance': 0.0207
        },
        'X17': {
            'name': 'ProfileCompletionScore',
            'type': 'numeric',
            'category': 'Profile Features',
            'description': 'Profile completion score (1-8)',
            'importance': 0.0375
        },
        'X18': {
            'name': 'IncompleteOrder',
            'type': 'numeric',
            'category': 'Account Metrics',
            'description': 'Number of incomplete orders',
            'importance': 0.0231
        },
        'X19': {
            'name': 'UrgentInMessageCount',
            'type': 'numeric',
            'category': 'Message Behavior',
            'description': 'Count of urgent messages',
            'importance': 0.2261
        },
        'X20': {
            'name': 'Activity_Risk_Encoded',
            'type': 'categorical',
            'category': 'Security Flags',
            'description': 'Activity risk level encoded',
            'importance': 0.0
        },
        'X21': {
            'name': 'SentLink_InMsg_Count',
            'type': 'numeric',
            'category': 'Message Behavior',
            'description': 'Count of links sent in messages',
            'importance': 0.0592
        },
        'X22': {
            'name': 'MentionsOffPlatformApp_EncodedCount',
            'type': 'numeric',
            'category': 'Message Behavior',
            'description': 'Count of off-platform app mentions',
            'importance': 0.0620
        },
        'X23': {
            'name': 'FlaggedByOtherUser_Count',
            'type': 'numeric',
            'category': 'Security Flags',
            'description': 'Count of flags by other users',
            'importance': 0.0309
        },
        'X24': {
            'name': 'UsernameHasNumbers_Flag',
            'type': 'boolean',
            'category': 'Account Behavior',
            'description': 'Username contains numbers',
            'importance': 0.0
        },
        'X25': {
            'name': 'UsernameHasExcessiveSpecialChars_Flag',
            'type': 'boolean',
            'category': 'Account Behavior',
            'description': 'Username has excessive special characters',
            'importance': 0.0334
        },
        'X26': {
            'name': 'UsedDisposableEmail_Flag',
            'type': 'boolean',
            'category': 'Account Behavior',
            'description': 'Used a disposable email service',
            'importance': 0.0
        },
        'X27': {
            'name': 'PaymentVerified_Flag',
            'type': 'boolean',
            'category': 'Account Information',
            'description': 'Payment method is verified',
            'importance': 0.0
        },
        'X28': {
            'name': 'AsksForEmail_InMsg_Flag',
            'type': 'boolean',
            'category': 'Message Behavior',
            'description': 'Asks for email in messages',
            'importance': 0.0
        },
        'X29': {
            'name': 'AsksForPayment_OffPlatform_Flag',
            'type': 'boolean',
            'category': 'Message Behavior',
            'description': 'Asks for payment outside platform',
            'importance': 0.0
        },
        'X30': {
            'name': 'SentShortenedLink_InMsg_Flag',
            'type': 'boolean',
            'category': 'Message Behavior',
            'description': 'Sent shortened links in messages',
            'importance': 0.0
        },
        'X31': {
            'name': 'AsksToOpenLink_Urgent_Flag',
            'type': 'boolean',
            'category': 'Message Behavior',
            'description': 'Urgently asks to open links',
            'importance': 0.0
        },
        'X32': {
            'name': 'MentionsAttachment_InMsg_Count',
            'type': 'numeric',
            'category': 'Message Behavior',
            'description': 'Count of attachment mentions',
            'importance': 0.0
        },
        'X33': {
            'name': 'UsedUrgentLanguage_InMsg_Flag',
            'type': 'boolean',
            'category': 'Message Behavior',
            'description': 'Uses urgent language in messages',
            'importance': 0.0
        },
        'X34': {
            'name': 'ImpersonationAttempt_InMsg_Flag',
            'type': 'boolean',
            'category': 'Message Behavior',
            'description': 'Attempts impersonation in messages',
            'importance': 0.0
        },
        'X35': {
            'name': 'AsksForFinancialDetails_OffPlatform_Flag',
            'type': 'boolean',
            'category': 'Security Flags',
            'description': 'Requested financial information',
            'importance': 0.0
        },
        'X36': {
            'name': 'AsksForCredentials_OffPlatform_Flag',
            'type': 'boolean',
            'category': 'Security Flags',
            'description': 'Requested login credentials',
            'importance': 0.0350
        },
        'X37': {
            'name': 'MentionsOffPlatformPaymentMethod_Flag',
            'type': 'boolean',
            'category': 'Security Flags',
            'description': 'Mentioned alternative payment methods',
            'importance': 0.0
        },
        'X38': {
            'name': 'UsedTemporaryOnlinePhoneNumber_Flag',
            'type': 'boolean',
            'category': 'Account Behavior',
            'description': 'Used temporary phone number',
            'importance': 0.0
        },
        'X39': {
            'name': 'VeryShortInitialMessage_Flag',
            'type': 'boolean',
            'category': 'Message Behavior',
            'description': 'Sent very short initial messages',
            'importance': 0.0206
        },
        'X40': {
            'name': 'UsedGenericSpamTemplate_Flag',
            'type': 'boolean',
            'category': 'Message Behavior',
            'description': 'Used generic spam message templates',
            'importance': 0.0
        },
        'X41': {
            'name': 'AsksForPersonalInfo_Flag',
            'type': 'boolean',
            'category': 'Security Flags',
            'description': 'Requested personal information',
            'importance': 0.0
        },
        'X42': {
            'name': 'ContactedUnsolicited_Flag',
            'type': 'boolean',
            'category': 'Message Behavior',
            'description': 'Contacted users without prior interaction',
            'importance': 0.0
        },
        'X43': {
            'name': 'RapidMessagingDetected_Flag',
            'type': 'boolean',
            'category': 'Message Behavior',
            'description': 'Sent many messages in short time',
            'importance': 0.0
        },
        'X44': {
            'name': 'AttemptedSuspiciousAction_Flag',
            'type': 'boolean',
            'category': 'Security Flags',
            'description': 'Attempted suspicious actions',
            'importance': 0.0
        },
        'X45': {
            'name': 'VagueJobDescriptionPosted_Flag',
            'type': 'boolean',
            'category': 'Message Behavior',
            'description': 'Posted vague job descriptions',
            'importance': 0.0
        },
        'X46': {
            'name': 'IndiscriminateApplicationsSent_Flag',
            'type': 'boolean',
            'category': 'Message Behavior',
            'description': 'Applied to many jobs without reading',
            'importance': 0.0
        },
        'X47': {
            'name': 'IsKnownBotOrHeuristic_Flag',
            'type': 'boolean',
            'category': 'Bot Detection',
            'description': 'Detected bot-like behavior',
            'importance': 0.0
        },
        'X48': {
            'name': 'HeadlessBrowser_Flag',
            'type': 'boolean',
            'category': 'Bot Detection',
            'description': 'Used headless browser',
            'importance': 0.0
        },
        'X49': {
            'name': 'SuspectedRobotUser_Flag',
            'type': 'boolean',
            'category': 'Bot Detection',
            'description': 'Suspected automated account',
            'importance': 0.0210
        },
        'X50': {
            'name': 'CaptchaDefeatedByBot_Flag',
            'type': 'boolean',
            'category': 'Bot Detection',
            'description': 'Bypassed security checks',
            'importance': 0.0
        },
        'X51': {
            'name': 'OtherBehaviorFlag_5',
            'type': 'boolean',
            'category': 'Other Flags',
            'description': 'Other suspicious behavior (5)',
            'importance': 0.0
        }
    }
    
    # Add composite features to the mapping
    mapping.update({
        'X52': {
            'name': 'AccountRiskScore',
            'type': 'numeric',
            'category': 'Composite Risk Score',
            'description': 'Combined risk score based on account age, message count, rating, and risk level',
            'importance': 0.0  # Will be updated after training
        },
        'X53': {
            'name': 'MessageBehaviorRiskScore',
            'type': 'numeric',
            'category': 'Composite Risk Score',
            'description': 'Combined risk score based on link count, off-platform mentions, and suspicious message behaviors',
            'importance': 0.0  # Will be updated after training
        },
        'X54': {
            'name': 'VerificationRiskScore',
            'type': 'numeric',
            'category': 'Composite Risk Score',
            'description': 'Combined risk score based on email, phone, and payment verification status',
            'importance': 0.0  # Will be updated after training
        }
    })
    
    return mapping

def main():
    # Load and preprocess data
    df = load_data('../data/fiverr_data.csv')
    
    # Add composite features
    df = create_composite_features(df)
    
    # Preprocess data
    X_scaled, y, user_ids, scaler, feature_names = preprocess_data(df)
    
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)
    
    # Initialize and train the model with optimized parameters
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=500,
        class_weight={0: 1, 1: 5},  # Increased weight for spam class
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Evaluate model with lower threshold
    metrics, trained_model = evaluate_model(
        model, X_train_resampled, X_test, y_train_resampled, y_test, feature_names, threshold=0.3
    )
    
    # Save model artifacts
    save_model_artifacts(trained_model, scaler, feature_names, metrics, threshold=0.3)
    
    # Print final metrics
    print("\nFinal Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Create feature mapping
    feature_mapping = create_feature_mapping()
    
    # Update importance scores for composite features
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': trained_model.feature_importances_
    })
    
    for i, feature in enumerate(feature_names):
        if feature in feature_mapping:
            feature_mapping[feature]['importance'] = feature_importance.loc[feature_importance['feature'] == feature, 'importance'].values[0]
    
    # Save feature mapping
    models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(feature_mapping, os.path.join(models_dir, 'feature_mapping.pkl'))
    
    print("\nFeature mapping created and saved successfully!")
    print("\nFeature mapping structure:")
    for encoded_col, info in feature_mapping.items():
        print(f"\n{encoded_col} -> {info['description']}")
        print(f"Type: {info['type']}")
        if 'mapping' in info:
            print(f"Categories: {info['mapping']}")
        print(f"Importance: {info['importance']}")

if __name__ == "__main__":
    main() 