# Fiverr Spammer Detection Project Summary

## Project Overview
This project aims to detect potential spammers on the Fiverr platform using machine learning techniques. The implementation includes comprehensive data analysis, model comparison, and prediction generation to identify users who might become spammers based on their behavior patterns.

## Dataset Description
- **Source**: Kaggle Competition - Predict Potential Spammers on Fiverr
- **Features**: 51 anonymized parameters (X1-X51) related to user behavior
- **Target Variable**: Binary classification (1 = spammer, 0 = non-spammer)
- **Identifier**: user_id for each entry
- **Privacy**: Feature names are anonymized for privacy concerns

## Data Analysis

### 1. Exploratory Data Analysis (EDA)
The EDA process includes:

#### Basic Statistics
- Dataset shape and dimensions
- Summary statistics for all features
- Class distribution analysis
- Missing values assessment

#### Feature Analysis
- Correlation analysis between features
- Identification of highly correlated feature pairs (>0.8)
- Statistical significance testing (t-tests)
- Feature importance ranking

#### Visualization Outputs
- Class distribution plots
- Correlation heatmaps
- Feature distribution plots
- Missing value analysis plots

### 2. Data Preprocessing
1. Missing Value Handling
   - Median imputation strategy
   - Quality checks for remaining NaN values

2. Feature Scaling
   - StandardScaler implementation
   - Zero mean and unit variance transformation

## Implementation Details

### Models Implemented

#### Tree-based Models
- Random Forest (100 & 300 trees)
- Extra Trees
- Decision Tree

#### Gradient Boosting Variants
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

#### Linear Models
- Logistic Regression
- Ridge Classifier
- SGD Classifier

#### Instance-based Learning
- KNN (k=5)
- KNN (k=10)

#### Probabilistic Models
- Gaussian Naive Bayes
- Bernoulli Naive Bayes

#### Neural Network
- Multi-layer Perceptron (100, 50 hidden layers)

#### Advanced Ensemble Methods
- AdaBoost
- Bagging with Random Forest
- Voting Classifier (Hard & Soft)
- Stacking Classifier

### Evaluation Metrics
For each model, we calculate and compare:
- Accuracy (Train & Test)
- Precision (Train & Test)
- Recall (Train & Test)
- F1 Score (Train & Test)

### Output Files Generated

#### 1. Analysis Outputs
- `eda_plots/class_distribution.png`
- `eda_plots/correlation_matrix.png`
- `eda_plots/missing_values.png`
- Feature distribution plots for significant features

#### 2. Model Evaluation Outputs
- `model_comparison_results.csv`
- `comparison_train_accuracy.png`
- `comparison_train_precision.png`
- `comparison_train_recall.png`
- `comparison_train_f1.png`
- `best_model_train_confusion_matrix.png`
- `best_model_test_confusion_matrix.png`

#### 3. Predictions
- `spammer_predictions.csv`: Final predictions file with user_id and prediction columns

## Dependencies
```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
```

## Running the Project
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script:
   ```bash
   python spammer_prediction.py
   ```

## Model Selection Strategy
- Models are evaluated based on their F1 score on the test set
- The best performing model is retrained on the full dataset
- Multiple model variants are tested to ensure robust comparison
- Ensemble methods combine the strengths of multiple base models

## Key Features of Implementation
1. **Comprehensive Data Analysis**: Detailed EDA with visualizations
2. **Robust Preprocessing**: Handling missing values and scaling
3. **Multiple Model Comparison**: Wide range of algorithms tested
4. **Advanced Ensemble Methods**: Voting and stacking implementations
5. **Detailed Evaluation**: Multiple metrics and visualizations
6. **Production-Ready**: Final model can be used for predictions

## Best Practices Implemented
1. Feature scaling for normalized input
2. Cross-validation in stacking classifier
3. Multiple evaluation metrics
4. Clear visualization of results
5. Proper train-test splitting
6. Comprehensive model comparison
7. Final model retraining on full dataset
8. Statistical validation of features

## Future Improvements
1. Hyperparameter tuning for each model
2. Feature selection based on importance
3. Cross-validation for all models
4. More advanced ensemble techniques
5. Deep learning approaches
6. Feature engineering based on domain knowledge
7. Model interpretability analysis
8. Real-time prediction capabilities
9. Model monitoring and updating system
10. API development for integration

## Overfitting Analysis

During the evaluation of various models, we observed signs of overfitting in some models, particularly those with high complexity. Here's a summary of the findings:

### Models with Overfitting Signs
- **Random Forest**: High train accuracy (0.9998) but lower test accuracy (0.9867) and a significant drop in F1 score from train (0.9957) to test (0.6911).
- **Extra Trees**: Similar pattern with train accuracy (0.9998) and test accuracy (0.9864), and F1 score dropping from train (0.9960) to test (0.6787).
- **Decision Tree**: Train accuracy (0.9998) vs. test accuracy (0.9759), with F1 score dropping from train (0.9960) to test (0.5640).
- **Stacking**: Train accuracy (0.9993) vs. test accuracy (0.9872), with F1 score dropping from train (0.9862) to test (0.7149).

### Models with Balanced Performance
- **Gradient Boosting**: Train accuracy (0.9837) and test accuracy (0.9838) with balanced F1 scores (train: 0.6123, test: 0.6029).
- **XGBoost**: Train accuracy (0.9878) and test accuracy (0.9867) with F1 scores (train: 0.7394, test: 0.7061).
- **CatBoost**: Train accuracy (0.9896) and test accuracy (0.9876) with F1 scores (train: 0.7773, test: 0.7249).

### Recommendations to Mitigate Overfitting
1. **Regularization**: Implement techniques like pruning for decision trees or L1/L2 regularization for linear models.
2. **Hyperparameter Tuning**: Adjust parameters such as tree depth, learning rate, and number of estimators.
3. **Cross-Validation**: Use k-fold cross-validation to ensure model robustness and generalization.
4. **Feature Selection**: Remove irrelevant or redundant features to simplify the model.

These strategies can help improve the generalization of models and reduce the risk of overfitting, leading to better performance on unseen data.

## Project Structure
```
├── spammer_prediction.py    # Main implementation file
├── requirements.txt         # Project dependencies
├── eda_plots/              # EDA visualization outputs
├── model_comparison_results.csv  # Model evaluation results
└── spammer_predictions.csv      # Final predictions
``` 