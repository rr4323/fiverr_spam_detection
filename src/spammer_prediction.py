# Import necessary libraries
# Data manipulation and analysis
import pandas as pd
import numpy as np
import os
import json

# Machine learning tools
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import various machine learning models
# Ensemble learning models
from sklearn.ensemble import (
    RandomForestClassifier,  # Combines multiple decision trees
    GradientBoostingClassifier,  # Builds trees sequentially
    AdaBoostClassifier,  # Adaptive Boosting
    ExtraTreesClassifier,  # More randomized tree building
    BaggingClassifier,  # Builds multiple models on data subsets
    VotingClassifier,  # Combines multiple models through voting
    StackingClassifier  # Combines models using meta-learner
)

# Linear models
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier

# Other classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier  # Extreme Gradient Boosting
from lightgbm import LGBMClassifier  # Light Gradient Boosting
from catboost import CatBoostClassifier  # Categorical Boosting
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier  # Neural Network
from sklearn.svm import SVC

from scipy import stats
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def load_data(file_path):
    """
    Load and provide initial analysis of the dataset
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Print dataset information
    print(f"Dataset shape: {df.shape}")
    print("\nSample of first few rows:")
    print(df.head())
    print("\nClass distribution:")
    print(df['label'].value_counts(normalize=True))
    
    # Check for missing values
    print("\nMissing values summary:")
    print(df.isnull().sum())
    return df

def preprocess_data(df):
    """
    Preprocess the data by handling missing values and scaling features
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        tuple: (scaled features, target variable, user IDs, scaler object)
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
    
    return X_scaled, y, user_ids, scaler

def handle_class_imbalance(X, y, method='smote'):
    """Handle class imbalance using various techniques"""
    print(f"\nHandling class imbalance using {method}...")
    
    if method == 'smote':
        # SMOTE: Synthetic Minority Over-sampling Technique
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    elif method == 'undersample':
        # Random under-sampling
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
    elif method == 'hybrid':
        # Hybrid approach: SMOTE + Random under-sampling
        smote = SMOTE(sampling_strategy=0.5, random_state=42)
        rus = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)
    else:
        # No resampling
        X_resampled, y_resampled = X, y
    
    print(f"Class distribution after {method}:")
    print(pd.Series(y_resampled).value_counts(normalize=True))
    
    return X_resampled, y_resampled

def save_model_results(model, model_name, metrics, y_test, y_pred, y_test_proba=None, feature_names=None):
    """Save detailed model results to a structured format"""
    # Create results directory if it doesn't exist
    results_dir = '../reports/model_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create model-specific directory
    model_dir = os.path.join(results_dir, model_name.lower().replace(" ", "_"))
    os.makedirs(model_dir, exist_ok=True)
    
    # Save metrics to JSON
    metrics_file = os.path.join(model_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_file = os.path.join(model_dir, 'classification_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Save confusion matrix as CSV
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, 
                        index=['Actual Negative', 'Actual Positive'],
                        columns=['Predicted Negative', 'Predicted Positive'])
    cm_df.to_csv(os.path.join(model_dir, 'confusion_matrix.csv'))
    
    # Save predictions and probabilities if available
    predictions_df = pd.DataFrame({
        'y_test': y_test,
        'y_pred': y_pred
    })
    if y_test_proba is not None:
        predictions_df['y_proba'] = y_test_proba
    predictions_df.to_csv(os.path.join(model_dir, 'predictions.csv'), index=False)
    
    # Save feature importance if available
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importance.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        feature_importance.head(20).plot(kind='bar', x='feature', y='importance')
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'feature_importance.png'))
        plt.close()
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    plt.close()

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, feature_names=None):
    """Evaluate a model and return metrics"""
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Try to get probability predictions if available
    y_test_proba = None
    try:
        y_test_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_test_proba)
    except (AttributeError, NotImplementedError):
        try:
            y_test_proba = model.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, y_test_proba)
        except (AttributeError, NotImplementedError):
            roc_auc = None
    
    # Create metrics dictionary
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc
    }
    
    # Save detailed results
    save_model_results(model, model_name, metrics, y_test, y_pred, y_test_proba, feature_names)
    
    return metrics

def feature_importance_analysis(model, feature_names, model_name):
    """Analyze and plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances - {model_name}')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'../plots/model_plots/feature_importances_{model_name}.png')
        plt.close()
        
        # Print top features
        print(f"\nTop 10 Important Features for {model_name}:")
        for i in range(10):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def main():
    """
    Main function to orchestrate the entire model training and evaluation process
    """
    # Load data
    df = load_data('../data/fiverr_data.csv')
    
    # Preprocess data
    X_scaled, y, user_ids, scaler = preprocess_data(df)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test, user_ids_train, user_ids_test = train_test_split(
        X_scaled, y, user_ids, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train, method='smote')
    
    # Define base models with class weights
    base_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    base_gb = GradientBoostingClassifier(random_state=42)
    base_xgb = XGBClassifier(scale_pos_weight=36, random_state=42)
    base_lr = LogisticRegression(class_weight='balanced', random_state=42)
    
    # Define dictionary of models to evaluate
    models = {
        # Tree-based Models
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'Random Forest (300 trees)': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        
        # Gradient Boosting Variants
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(scale_pos_weight=36, random_state=42),
        'LightGBM': LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(class_weights=[1, 36], random_state=42, verbose=False),
        
        # Linear Models
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
        'Ridge Classifier': RidgeClassifier(class_weight='balanced', random_state=42),
        'SGD Classifier': SGDClassifier(class_weight='balanced', max_iter=1000, random_state=42),
        
        # Instance-based Learning
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'KNN (k=10)': KNeighborsClassifier(n_neighbors=10),
        
        # Naive Bayes Variants
        'Gaussian NB': GaussianNB(),
        'Bernoulli NB': BernoulliNB(),
        
        # Neural Network
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        
        # Ensemble Methods
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Bagging': BaggingClassifier(estimator=base_rf, random_state=42),
        
        # Voting Classifier
        'Voting (Hard)': VotingClassifier(
            estimators=[
                ('rf', base_rf),
                ('gb', base_gb),
                ('xgb', base_xgb)
            ],
            voting='hard'
        ),
        
        'Voting (Soft)': VotingClassifier(
            estimators=[
                ('rf', base_rf),
                ('gb', base_gb),
                ('lr', base_lr)
            ],
            voting='soft'
        ),
        
        # Stacking Classifier
        'Stacking': StackingClassifier(
            estimators=[
                ('rf', base_rf),
                ('gb', base_gb),
                ('xgb', base_xgb)
            ],
            final_estimator=LogisticRegression(class_weight='balanced'),
            cv=5
        )
    }
    
    # Get feature names
    feature_names = df.drop(['label', 'user_id'], axis=1).columns
    
    # Evaluate all models
    results = []
    best_model = None
    best_test_f1 = 0
    
    print("\nTraining and evaluating models...")
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        result = evaluate_model(model, X_train_resampled, X_test, y_train_resampled, y_test, model_name, feature_names)
        results.append(result)
        
        # Track best model based on test F1 score
        if result['F1 Score'] > best_test_f1:
            best_test_f1 = result['F1 Score']
            best_model = (model_name, model)
        
        # Analyze feature importance for tree-based models
        if model_name in ['Random Forest', 'Random Forest (300 trees)', 'Extra Trees', 'XGBoost', 'LightGBM', 'CatBoost']:
            feature_names = df.drop(['label', 'user_id'], axis=1).columns
            feature_importance_analysis(model, feature_names, model_name)
    
    # Save and visualize results
    results_df = pd.DataFrame(results)
    results_df.to_csv('../reports/model_comparison.csv', index=False)
    print("\nModel comparison results saved to '../reports/model_comparison.csv'")
    
    # Print best model
    best_model_name, best_model = best_model
    print(f"\nBest performing model: {best_model_name}")
    print("\nDetailed results for all models:")
    print(results_df.to_string(index=False))

# Entry point of the script
if __name__ == "__main__":
    main() 