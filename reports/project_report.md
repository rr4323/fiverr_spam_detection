# Fiverr Spammer Detection System

## 1. Project Overview
A machine learning system to detect and prevent spam activities on Fiverr platform using user behavior analysis and pattern recognition.

## 2. Problem & Solution
  ### Problem:
   Recently attackers are using freelance job sites such as Fiverr to distribute malware disguised as job offers. These job offers contain attachments that pretend to be the job brief but are actually installers for keyloggers such as Agent Tesla or Remote Access Trojan (RATs). Due to this many users lost their earnings, bidding fees and fake client projects, also some users lost their accounts too. Many of the LinkedIn connections faced it and some of them lost their professional growth, side income and stability.

  This project is about to understand how data science people will solve this problem by using their different methods and techniques. 

  Columns in the Training Set
      • label - indicates whether or not the user became a spammer. A "1" indicates the user became a spammer, a "0" indicates the user did not become a spammer.
      • user_id - the unique ID of the user
      • Columns X1 through X51 are different parameters that a user took before or after registering to the platform. This could be things like "whether or not the username contains an underscore" or "the number of characters in the users email" or "whether a user came from a valid referrer (i.e. google, bing, or another site)." Due to privacy issues, columns for all of these parameters have been named X with a following number.
### Solution: 
ML-based detection system using Random Forest classifier with 300 trees
### Key Features**: 
51 behavioral and account metrics

## 3. Exploratory Data Analysis (EDA)
### 3.1 Data Characteristics
- **Total Samples**: 458,798
- **Total Features**: 51
- **Class Distribution**:
  - Spammer: 2.69%
  - Non-Spammer: 97.31%
- **Missing Values**: 6 total missing values (only in X13)

### 3.2 Feature Analysis
- **Top Important Features** (Based on Model Analysis):
  1. X19 (UrgentInMessageCount): 0.2261 - Count of urgent messages
  2. X1 (MessagesSent_Total): 0.0761 - Total number of messages sent
  3. X2 (MessagesSent_Last30Days): 0.0680 - Messages sent in last 30 days
  4. X7 (VerificationLevel_Encoded): 0.0649 - Level of account verification
  5. X22 (OffPlatformApp_Mentions): 0.0620 - Mentions of off-platform communication
  6. X21 (LinksInMessages_Count): 0.0592 - Count of links in messages

- **Feature Interpretation Approach**:
  - Using LLM to analyze spammer behavior patterns
  - Hypothesis-driven feature importance interpretation
  - Focus on model performance rather than feature meaning
  - Continuous validation through model metrics

- **Highly Correlated Feature Pairs** (>0.8):
  - X18 ↔ X16: 0.844
  - X20 ↔ X10: 0.862
  - X35 ↔ X34: 0.972
  - X44 ↔ X41: 0.931

### 3.3 Key Patterns
- Severe class imbalance (2.69% vs 97.31%)
- Strong feature discrimination capability
- Multiple highly correlated feature pairs
- Minimal missing data (only 6 values in X13)

## 4. Model Development
### 4.1 Data Preprocessing
- **Class Imbalance Handling**:
  - SMOTE: Applied to training data
  - Class Weights: 'balanced' for most models
  - XGBoost: scale_pos_weight=36
  - Stratified Sampling: Used in train-test split

- **Feature Processing**:
  - Median imputation for missing values
  - StandardScaler for feature scaling
  - Feature correlation analysis
  - Dimensionality reduction

### 4.2 Model Selection & Evaluation
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest (300 trees) | 0.985 | 0.761 | 0.636 | 0.693 | 0.951 |
| Random Forest | 0.985 | 0.760 | 0.633 | 0.691 | 0.948 |
| Bagging Classifier | 0.984 | 0.726 | 0.663 | 0.693 | 0.956 |
| Extra Trees | 0.984 | 0.731 | 0.646 | 0.686 | 0.946 |
| Stacking | 0.985 | 0.780 | 0.598 | 0.677 | 0.856 |
| Voting (Soft) | 0.968 | 0.440 | 0.752 | 0.555 | 0.950 |
| Voting (Hard) | 0.963 | 0.404 | 0.762 | 0.528 | - |
| Neural Network | 0.957 | 0.354 | 0.720 | 0.474 | 0.930 |
| Decision Tree | 0.971 | 0.469 | 0.597 | 0.525 | 0.790 |
| Gradient Boosting | 0.955 | 0.342 | 0.747 | 0.469 | 0.947 |
| LightGBM | 0.980 | 0.629 | 0.642 | 0.635 | 0.957 |
| AdaBoost | 0.931 | 0.247 | 0.773 | 0.375 | 0.938 |
| KNN (k=5) | 0.937 | 0.261 | 0.737 | 0.386 | 0.873 |
| KNN (k=10) | 0.926 | 0.232 | 0.762 | 0.355 | 0.893 |
| XGBoost | 0.916 | 0.227 | 0.878 | 0.361 | 0.958 |
| CatBoost | 0.881 | 0.172 | 0.903 | 0.289 | 0.958 |
| Logistic Regression | 0.865 | 0.148 | 0.842 | 0.251 | 0.925 |
| SGD Classifier | 0.867 | 0.149 | 0.844 | 0.254 | 0.923 |
| Ridge Classifier | 0.818 | 0.113 | 0.846 | 0.199 | 0.905 |
| Bernoulli NB | 0.779 | 0.089 | 0.779 | 0.159 | 0.880 |
| Gaussian NB | 0.487 | 0.047 | 0.944 | 0.090 | 0.880 |

### 4.3 Best Model: Random Forest (300 trees)
- **Why Selected**:
  1. Best balance between precision (0.761) and recall (0.636)
  2. High ROC-AUC (0.951)
  3. Good interpretability with feature importance
  4. Robust to overfitting
  5. Handles class imbalance well
  6. Highest accuracy among all models (0.985)
  7. Consistent performance across all metrics

- **Key Parameters**:
  - n_estimators: 300
  - class_weight: balanced
  - max_depth: 20
  - min_samples_split: 5

- **Confusion Matrix**:
  ```
  [[11962   238]
   [  364   636]]
  ```

- **Alternative Strong Performers**:
  1. Bagging Classifier (F1: 0.693, ROC-AUC: 0.956)
  2. Extra Trees (F1: 0.686, ROC-AUC: 0.946)
  3. Stacking (F1: 0.677, ROC-AUC: 0.856)

## 5. Results & Impact
- **Performance Metrics**:
  - F1 Score: 0.693
  - ROC-AUC: 0.951
  - Precision: 0.761
  - Recall: 0.636
  - Accuracy: 0.985

- **Business Impact**:
  - Reduced false positives through high precision
  - Good spam detection rate with balanced recall
  - Improved user experience
  - Reduced manual moderation workload

## 6. Feature Analysis
### 6.1 Feature Categories and Importance
- **Top Important Features** (Based on Model Analysis):
  1. X19 (UrgentInMessageCount): 0.2261 - Count of urgent messages
  2. X1 (MessagesSent_Total): 0.0761 - Total number of messages sent
  3. X2 (MessagesSent_Last30Days): 0.0680 - Messages sent in last 30 days
  4. X7 (VerificationLevel_Encoded): 0.0649 - Level of account verification
  5. X22 (OffPlatformApp_Mentions): 0.0620 - Mentions of off-platform communication
  6. X21 (LinksInMessages_Count): 0.0592 - Count of links in messages

- **Features with Zero Importance**:
  - X33 (UsedUrgentLanguage_InMsg)
  - X30 (SentShortenedLink_InMsg)
  - X46 (IndiscriminateApplicationsSent)
  - X47 (IsKnownBotOrHeuristic)
  - X48 (HeadlessBrowser)
  - X29 (AsksForPayment_OffPlatform)
  - X27 (PaymentVerified_Flag)

### 6.2 Feature Categories
- **Account Metrics**:
  - X1 (MessagesSent_Total): Total number of messages sent
  - X2 (MessagesSent_Last30Days): Messages sent in last 30 days
  - X16 (MessageInDay_Count): Messages per day (3-18)
  - X18 (IncompleteOrder_Count): Number of incomplete orders (3-13)
  - X19 (UrgentInMessageCount): Count of urgent messages

- **Account Information**:
  - X3 (SellerCategory_Encoded): Seller category classification
  - X6 (CountryTier_Encoded): Country risk tier (2-4)
  - X7 (VerificationLevel_Encoded): Verification level
  - X8 (TotalVerifiedReview_Count): Number of verified reviews (4-25)
  - X17 (ProfileCompletionScore): Profile completion score (1-8)

- **Profile Features**:
  - X10 (HasProfilePic_Flag): Profile picture presence
  - X11 (AskedPersonalInfo_Flag): Personal info requests
  - X12 (LinkInMessage_Flag): Link presence in messages
  - X13 (PhoneVerified_Flag): Phone verification status
  - X14 (SkillsListed_Flag): Skills listed status
  - X15 (EmailVerified_Flag): Email verification status

- **Message Behavior**:
  - X19 (UrgentInMessageCount): Count of urgent messages
  - X21 (LinksInMessages_Count): Count of links in messages
  - X22 (OffPlatformApp_Mentions): Off-platform communication mentions
  - X28 (AskedForEmail_Flag): Email requests in messages
  - X29 (AskedForOffPlatformPayment_Flag): Off-platform payment requests
  - X30 (SentShortenedLinks_Flag): Shortened link usage
  - X31 (UrgentLinkRequests_Flag): Urgent link requests
  - X32 (AttachmentMentions_Flag): Attachment mentions
  - X33 (UsedUrgentLanguage_Flag): Urgent language usage
  - X34 (ImpersonationAttempt_Flag): Impersonation attempts
  - X39 (VeryShortMessages_Flag): Very short messages
  - X40 (UsedSpamTemplate_Flag): Spam template usage
  - X41 (AskedForPersonalInfo_Flag): Personal info requests
  - X42 (UnsolicitedContact_Flag): Unsolicited contact
  - X43 (RapidMessaging_Flag): Rapid messaging detection
  - X45 (VagueJobDescriptions_Flag): Vague job descriptions

- **Security Flags**:
  - X20 (RiskLevel_Encoded): Risk level classification
  - X23 (UserFlags_Count): User flag count
  - X35 (AskedForFinancialDetails_Flag): Financial details requests
  - X36 (AskedForCredentials_Flag): Credential requests
  - X37 (OffPlatformPaymentMentions_Flag): Off-platform payment mentions
  - X44 (SuspiciousActions_Flag): Suspicious actions

- **Account Behavior**:
  - X24 (UsernameHasNumbers_Flag): Username contains numbers
  - X25 (UsernameHasSpecialChars_Flag): Username has special characters
  - X26 (UsedDisposableEmail_Flag): Disposable email usage
  - X38 (UsedTemporaryPhone_Flag): Temporary phone usage

- **Bot Detection**:
  - X47 (BotLikeBehavior_Flag): Bot-like behavior
  - X48 (HeadlessBrowser_Flag): Headless browser detection
  - X49 (SuspectedRobot_Flag): Robot suspicion
  - X50 (CaptchaDefeated_Flag): Captcha defeat

### 6.3 Feature Encoding
- **Country Tier** (X6):
  - 2: Tier 1 (High Trust)
  - 3: Tier 2 (Medium Risk)
  - 4: Tier 3 (High Risk, e.g., Pakistan)

- **Profile Completion** (X17):
  - 8: Complete
  - 7: Advanced
  - 6: Intermediate
  - 5: Basic
  - 4: Minimal
  - 3: Very Basic
  - 2: Incomplete
  - 1: Empty

- **Verified Reviews** (X8):
  - 4-25: Range of verified reviews
  - 5-10: Higher spammer probability range

- **Daily Messages** (X16):
  - 3-18: Normal range of messages per day

- **Incomplete Orders** (X18):
  - 3-13: Range of incomplete orders

### 6.4 Key Patterns
- **Spammer Characteristics**:
  - High urgent message count (X19)
  - Personal info requests (X11)
  - Link presence in messages (X12)
  - Incomplete orders (X18)
  - Low profile completion (X17)
  - Verified reviews in suspicious range (X8: 5-10)
  - Higher country risk tier (X6: 3-4)

- **Non-Spammer Characteristics**:
  - Low urgent message count
  - No personal info requests
  - No links in messages
  - Complete orders
  - High profile completion
  - Verified reviews outside suspicious range
  - Lower country risk tier

### 6.5 Feature Importance Analysis
- **Top Features**:
  1. Urgent Message Count (X19): Most important feature, indicating that urgent messaging patterns are strong indicators of spam behavior
  2. Total Messages Sent (X1): Shows overall activity level
  3. Recent Messages (X2): Indicates current activity patterns
  4. Verification Level (X7): Important for establishing account legitimacy
  5. Off-Platform Mentions (X22): Shows attempts to move communication off-platform
  6. Link Count (X21): Indicates potential malicious content sharing

- **Risk Indicators**:
  - High urgent message count
  - Personal info requests
  - Link presence in messages
  - Incomplete orders
  - Low profile completion
  - Verified reviews in suspicious range (5-10)
  - Higher country risk tier

This analysis suggests that while individual message behaviors are important, the combination of urgent messaging patterns, verification status, and profile completeness is most significant for spam detection.

## 7. Implementation
### 7.1 Trained Model
- **Model Artifacts**:
  - Random Forest model (.pkl)
  - StandardScaler for feature scaling
  - Feature names and importance
  - Model version tracking system
  - Performance metrics history

- **Prediction Process**:
  1. Load trained model and scaler
  2. Scale input features
  3. Make prediction
  4. Return probability, class, and model version

- **Model Versioning**:
  - Automatic version tracking
  - Version history with metrics
  - Last update timestamp
  - Performance metrics per version
  - Thread-safe model updates

### 7.2 Streamlit Application
- **Core Features**:
  - User input form for 51 features
  - Real-time prediction display
  - Probability score visualization
  - Risk factor analysis
  - Model version information
  - Feedback submission system

- **User Interface**:
  - Clean and intuitive design
  - Feature input validation
  - Result visualization
  - Error handling
  - Version information display
  - Feedback collection form

### 7.3 Technical Stack
- Python 3.10
- scikit-learn
- pandas
- Streamlit
- joblib
- FastAPI
- OpenTelemetry
- Hugging Face Hub

### 7.4 Online Learning System
- **Feedback Collection**:
  - Prediction feedback mechanism
  - Feedback buffer management
  - Thread-safe feedback storage
  - Feedback statistics tracking
  - Automated retraining triggers

- **Model Updates**:
  - Automatic retraining process
  - Version increment on updates
  - Performance metrics tracking
  - Model artifact management
  - Thread-safe model switching

- **Configuration**:
  ```python
  class OnlineLearningConfig:
      learning_rate: float = 0.01
      batch_size: int = 100
      min_samples: int = 1000
      retrain_interval: int = 3600  # 1 hour
  ```

## 8. Current Status and Future Improvements
### 8.1 Implemented Features
- **Online Learning**:
  - ✓ Incremental learning for model updates
  - ✓ Real-time adaptation to new spam patterns
  - ✓ Continuous model retraining with new data
  - ✓ Automated model versioning
  - ✓ Performance monitoring during updates
  - ✓ Thread-safe model updates
  - ✓ Feedback-based retraining
  - ✓ Version tracking and metrics

- **Model Configuration**:
  - ✓ OnlineLearningConfig for retraining parameters
  - ✓ Batch processing support
  - ✓ Performance metrics tracking
  - ✓ Model artifact management
  - ✓ Automated version increment

### 8.2 Future Improvements
- **Model Enhancements**:
  - A/B testing for model versions
  - Version performance comparison dashboard
  - Automated rollback on performance degradation
  - Enhanced feature importance tracking
  - Model ensemble techniques
  - Automated hyperparameter optimization
  - Cross-validation improvements

- **Application Improvements**:
  - **User Feedback System** (Partially Implemented):
    - ✓ Basic feedback collection
    - ✓ Feedback statistics tracking
    - Planned: Advanced feedback analytics dashboard
    - Planned: User trust scoring system
    - Planned: Feedback-based feature importance updates

  - **Performance Monitoring** (Partially Implemented):
    - ✓ Basic metrics tracking
    - ✓ Version performance logging
    - ✓ Error rate monitoring
    - Planned: Advanced model drift detection
    - Planned: Resource utilization monitoring
    - Planned: Prediction latency tracking
    - Planned: Automated alerting system

  - **Enhanced Visualization**:
    - Planned: Interactive performance dashboards
    - Planned: Real-time prediction visualization
    - Planned: Feature importance trends
    - Planned: User behavior patterns
    - Planned: Spam pattern evolution tracking

  - **Batch Processing** (Partially Implemented):
    - ✓ Basic batch predictions
    - ✓ Scheduled model retraining
    - Planned: Automated report generation
    - Planned: Batch performance analysis
    - Planned: Bulk feedback processing

### 8.3 Feature Understanding
- **Domain Expert Collaboration**:
  - Planned: Regular review sessions with moderators
  - Planned: Expert validation of feature importance
  - Planned: New feature suggestions
  - Planned: Pattern validation
  - Planned: Rule-based system integration

- **Feature Validation**:
  - Planned: A/B testing for new features
  - Planned: Feature impact analysis
  - Planned: Correlation studies
  - Planned: Feature stability monitoring
  - Planned: Feature drift detection

- **Continuous Improvement**:
  - ✓ Regular model performance reviews
  - ✓ Feature importance updates
  - Planned: Pattern recognition updates
  - Planned: Spam behavior evolution tracking
  - Planned: Model adaptation to new spam tactics

### 8.4 Current Limitations
- Model retraining requires minimum sample size
- Limited real-time performance monitoring
- Basic feedback collection system
- No automated alerting system
- Limited visualization capabilities
- No advanced feature validation
- Basic batch processing support

### 8.5 Next Steps
1. Implement advanced monitoring and alerting
2. Develop comprehensive visualization dashboard
3. Enhance feedback collection system
4. Implement automated feature validation
5. Develop advanced batch processing
6. Create expert collaboration system
7. Implement automated rollback mechanism

## 9. Resources
### 9.1 Dataset Source
- **Primary Dataset**: Fiverr User Behavior Dataset
  - Source: Internal Fiverr platform data
  - Size: 458,798 samples
  - Features: 51 behavioral and account metrics
  - Class Distribution: 2.69% spammer, 97.31% non-spammer
  - Access: Internal Fiverr data warehouse

- **Data Collection Process**:
  - User behavior tracking
  - Account activity monitoring
  - Message content analysis
  - Profile verification data
  - Historical spam reports

### 9.2 Software Stack
- **Core ML Libraries**:
  - scikit-learn 1.3.0
  - pandas 2.0.3
  - numpy 1.24.3
  - imbalanced-learn 0.11.0

- **Development Tools**:
  - Python 3.10
  - VS Code
  - Git
  - pytest 8.1.1
  - pytest-mock 3.12.0
  - requests-mock 1.12.0

- **Deployment**:
  - FastAPI 0.104.1
  - Streamlit 1.28.0
  - joblib 1.3.2
  - uvicorn 0.24.0
  - OpenTelemetry 1.21.0
  - Hugging Face Hub 0.22.2

### 9.3 Monitoring and Observability
- **Metrics Collection**:
  - Model performance metrics
  - Version tracking
  - Feedback statistics
  - Prediction latency
  - Error rates
  - Resource utilization

- **Alerting System**:
  - Performance degradation alerts
  - Error rate thresholds
  - Resource usage alerts
  - Model drift detection
  - Version comparison alerts

## 10. Individual Details
### Project Team
- **Name**: Rajeev Ranjan
- **Email**: ranjanrajeev886@gmail.com
- **Phone**: 9829159481
- **Role**: Senior Backend Developer 