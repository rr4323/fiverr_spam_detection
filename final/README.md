# Fiverr Spammer Detection System

## Overview
A machine learning-based system designed to detect and prevent spam activities on the Fiverr platform. The system utilizes advanced user behavior analysis and pattern recognition techniques to identify potential spammers with high accuracy.

## Key Features
- Machine Learning-based spam detection using Random Forest classifier
- High precision (76.1%) and recall (63.6%) in spam detection
- Analysis of 51 behavioral and account metrics
- Real-time user behavior monitoring
- Automated spam probability scoring
- Robust handling of class imbalance

## Technical Details
- **Model**: Random Forest with 300 trees
- **Performance Metrics**:
  - F1 Score: 0.693
  - ROC-AUC: 0.951
  - Precision: 0.761
  - Recall: 0.636
  - Accuracy: 0.985

## Requirements
- Python 3.8+
- Key Dependencies:
  - scikit-learn==1.2.2
  - pandas==2.2.3
  - numpy==1.26.4
  - FastAPI==0.115.12
  - Streamlit==1.44.1
  - XGBoost==3.0.0
  - CatBoost==1.2.7
  - LightGBM==4.6.0

## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
.
├── src/           # Source code
├── reports/       # Analysis reports and documentation
├── README.md     # Project documentation
└── requirements.txt  # Python dependencies
```

## Features Analyzed
The system analyzes various user behavior patterns including:
- Account age and activity metrics
- Message patterns and frequency
- Profile completeness
- Verification status
- User interaction patterns
- Suspicious activity indicators

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[MIT License](LICENSE)

## Authors
- Project Team

For detailed technical information and analysis, please refer to the `project_report.md` file.
