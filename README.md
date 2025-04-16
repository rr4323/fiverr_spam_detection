##Problem statement - 

Recently attackers are using freelance job sites such as Fiverr to distribute malware disguised as job offers. These job offers contain attachments that pretend to be the job brief but are actually installers for keyloggers such as Agent Tesla or Remote Access Trojan (RATs). Due to this many users lost their earnings, bidding fees and fake client projects, also some users lost their accounts too. Many of the LinkedIn connections faced it and some of them lost their professional growth, side income and stability.

This project is about to understand how data science people will solve this problem by using their different methods and techniques. 

Columns in the Training Set
    • label - indicates whether or not the user became a spammer. A "1" indicates the user became a spammer, a "0" indicates the user did not become a spammer.
    • user_id - the unique ID of the user
    • Columns X1 through X51 are different parameters that a user took before or after registering to the platform. This could be things like "whether or not the username contains an underscore" or "the number of characters in the users email" or "whether a user came from a valid referrer (i.e. google, bing, or another site)." Due to privacy issues, columns for all of these parameters have been named X with a following number.

Objective : -  Find a good fit algorithm which will have Potential to Predict Spammers. 

Dataset link: https://www.kaggle.com/competitions/predict-potential-spammers-on-fiverr

## Model Deployment

The trained model has been deployed to Hugging Face for easy access and integration. You can find the model at: [blackrajeev/fiverr_spam_detection](https://huggingface.co/blackrajeev/fiverr_spam_detection)

### Hugging Face Profile
- Profile: [blackrajeev](https://huggingface.co/blackrajeev)
- Repository: [fiverr_spam_detection](https://huggingface.co/blackrajeev/fiverr_spam_detection)

### Deploying Models to Hugging Face

To deploy the model files to Hugging Face, follow these steps:

1. Install the Hugging Face CLI:
```bash
pip install -U "huggingface_hub[cli]"
```

2. Login with your Hugging Face credentials:
```bash
huggingface-cli login
```

3. Push your model files:
```bash
huggingface-cli upload blackrajeev/fiverr_spam_detection .
```

Make sure you have the following files in your directory before uploading:
- spammer_detector.pkl
- scaler.pkl
- feature_mapping.pkl
- model_metrics.json

### Running the Streamlit App

The project includes a Streamlit web application for interactive model predictions and data visualization. To run the app:

1. Navigate to the src directory:
```bash
cd src
```

2. Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501` in your web browser.

### API Usage

The project includes a FastAPI-based REST API for making predictions. The API can be used in two ways:

1. **Local Deployment**:
   - Clone this repository
   - Install dependencies: `pip install -r requirements.txt`
   - Run the API: `python src/api.py`
   - The API will be available at `http://localhost:8000`

2. **Using Hugging Face Model**:
   - The API is configured to use the model from Hugging Face
   - No need to download model files locally
   - Simply make requests to the API endpoints

### API Endpoints

- `GET /`: Basic API information
- `GET /model/info`: Get model metrics and feature mapping
- `GET /model/version`: Get current model version and metrics
- `POST /predict`: Make a single prediction
- `POST /predict/batch`: Make multiple predictions at once
- `POST /feedback`: Submit feedback for model improvement
- `GET /model/feedback/stats`: Get feedback statistics

Example prediction request:
```json
{
    "features": {
        "X1": 0.5,
        "X2": 1.0,
        // ... other features
    }
}
```

Example prediction response:
```json
{
    "prediction": 1,
    "probability": 0.85,
    "is_spammer": true,
    "risk_factors": [
        {
            "feature": "X19",
            "description": "Urgent message count",
            "category": "Message Behavior",
            "value": "7",
            "reason": "High number of urgent messages"
        }
    ],
    "model_version": 2
}
```

### Model Versioning and Updates

The system includes automatic model versioning and updates:

1. **Version Tracking**:
   - Each model retraining increments the version number
   - Version information includes:
     - Version number
     - Last update timestamp
     - Current performance metrics

2. **Automatic Updates**:
   - Model is automatically retrained when:
     - Enough feedback is collected (configurable threshold)
     - Minimum time has passed since last retraining
   - New predictions automatically use the latest model version
   - No manual intervention needed for updates

3. **Version Information**:
   - Check current model version:
     ```bash
     curl http://localhost:8000/model/version
     ```
   - Response includes:
     ```json
     {
         "version": 2,
         "last_updated": "2024-03-20T12:00:00Z",
         "metrics": {
             "accuracy": 0.92,
             "precision": 0.89,
             "recall": 0.94
         }
     }
     ```

4. **Feedback Collection**:
   - Submit feedback after predictions:
     ```bash
     curl -X POST http://localhost:8000/feedback \
          -H "Content-Type: application/json" \
          -d '{
              "prediction_id": "pred_123",
              "is_correct": true,
              "actual_label": null,
              "feedback_notes": "Correct prediction"
          }'
     ```

5. **Feedback Statistics**:
   - Monitor feedback collection:
     ```bash
     curl http://localhost:8000/model/feedback/stats
     ```
   - Response includes:
     ```json
     {
         "total_feedback": 150,
         "correct_predictions": 135,
         "accuracy": 0.90,
         "last_retrain": "2024-03-20T12:00:00Z"
     }
     ```

### Development

When adding new features or making changes:

1. Write tests for new functionality
2. Run the test suite to ensure existing functionality works
3. Make your changes
4. Run tests again to verify changes
5. Update documentation if needed

The test suite helps maintain code quality and catch potential issues early in development.

### Model Retraining Configuration

The system's retraining behavior can be configured through the `OnlineLearningConfig`:

```python
class OnlineLearningConfig:
    learning_rate: float = 0.01
    batch_size: int = 100
    min_samples: int = 1000
    retrain_interval: int = 3600  # 1 hour in seconds
```

Adjust these parameters based on your needs:
- `min_samples`: Minimum feedback samples needed for retraining
- `retrain_interval`: Minimum time between retraining attempts
- `batch_size`: Number of samples processed in each training batch
- `learning_rate`: Learning rate for the model (if applicable)

