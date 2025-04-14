Problem statement - 

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
- `POST /predict`: Make a single prediction
- `POST /predict/batch`: Make multiple predictions at once

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

For detailed API documentation, visit `http://localhost:8000/docs` when running the API locally.

