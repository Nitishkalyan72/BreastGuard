
# Breast Cancer Model Deployment Validation Script
# Generated on: 2025-11-01T13:28:34.865621

import joblib
import numpy as np
import pandas as pd

def load_model():
    '''Load the trained model and scaler'''
    model = joblib.load('breast_cancer_model_deployment\breast_cancer_model.pkl')
    scaler = joblib.loadcoord('breast_cancer_model_deployment\breast_cancer_scaler.pkl')
    metadata = joblib.load('breast_cancer_model_deployment\model_metadata.pkl')
    return model, scaler, metadata

def predict_breast_cancer(features):
    '''
    Predict breast cancer malignancy for given features

    Parameters:
    features: numpy array or pandas DataFrame with shape (n_samples, 10)
              Features should be in the same order as training data

    Returns:
    predictions: numpy array with predictions (0=Benign, 1=Malignant)
    probabilities: numpy array with prediction probabilities
    '''
    model, scaler, metadata = load_model()

    # Ensure features are in correct format
    if isinstance(features, pd.DataFrame):
        features = features.values

    # Apply scaling
    features_scaled = scaler.transform(features)

    # Make predictions
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[:, 1]

    return predictions, probabilities

def get_model_info():
    '''Get model information and performance metrics'''
    _, _, metadata = load_model()
    return metadata

# Example usage:
if __name__ == "__main__":
    # Load model info
    info = get_model_info()
    print("Model Information:")
    print(f"Version: {info['version']}")
    print(f"Training Date: {info['training_date']}")
    print(f"Accuracy: {info['performance_metrics']['accuracy']:.4f}")
    print(f"Sensitivity: {info['performance_metrics']['recall']:.4f}")
    print(f"Specificity: {info['clinical_validation']['specificity']:.4f}")
