
# Breast Cancer Malignancy Prediction Model

## Model Overview
This model predicts breast cancer malignancy using quantitative features from fine needle aspirate (FNA) samples.

## Model Performance
- **Accuracy**: 0.9912
- **Sensitivity (Recall)**: 0.9762
- **Specificity**: 1.0000
- **ROC-AUC**: 0.9970
- **Brier Score**: 0.0153

## Files in Deployment Package
- `breast_cancer_model.pkl`: Trained logistic regression model
- `breast_cancer_scaler.pkl`: Feature scaler for preprocessing
- `model_metadata.pkl`: Model metadata and performance metrics
- `model_info.json`: Human-readable model information
- `deployment_validation.py`: Validation script for model deployment
- `requirements.txt`: Python package dependencies

## Usage
```python
import joblib

# Load model and scaler
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('breast_cancer_scaler.pkl')

# Make predictions
predictions = model.predict(scaled_features)
probabilities = model.predict_proba(scaled_features)[:, 1]
```

## Clinical Notes
- Model should be used as decision support tool only
- Requires human oversight and clinical judgment
- Regular monitoring and updates recommended
- Compliance with healthcare regulations required

## Version: 1.0.0
## Training Date: 2025-10-31T09:18:19.451596
