"""
Training Data Generation Script for SHAP Explainability
--------------------------------------------------------
This script generates the training background data required for SHAP explainability.
It ensures the data matches the model's preprocessing pipeline and feature order.

Run this script whenever:
- The model is retrained
- The preprocessing pipeline changes
- train_data.pkl is missing or corrupted

Usage:
    python generate_training_data.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import sys
import os

# Add deployment directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'breast_cancer_model_deployment'))
from preprocessor import BreastCancerPreprocessor


def generate_training_data():
    """
    Generate and save training data for SHAP explainability.
    
    This function:
    1. Loads the breast cancer dataset
    2. Selects the same 10 features used by the model
    3. Applies the same train/test split (random_state=42)
    4. Applies preprocessing (log transformation)
    5. Scales the data using the saved scaler
    6. Saves the result as train_data.pkl
    """
    
    print("=" * 70)
    print("SHAP Training Data Generation")
    print("=" * 70)
    
    # Load the breast cancer dataset
    print("\n1. Loading breast cancer dataset...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    print(f"   ✓ Loaded {len(df)} samples")
    
    # Map to expected column names
    # The dataset uses different naming conventions, so we map them
    feature_mapping = {
        'mean compactness': 'compactness_mean',
        'mean concavity': 'concavity_mean',
        'worst radius': 'radius_worst',
        'worst texture': 'texture_worst',
        'worst perimeter': 'perimeter_worst',
        'worst area': 'area_worst',
        'worst smoothness': 'smoothness_worst',
        'worst concave points': 'concave_points_worst',
        'worst symmetry': 'symmetry_worst',
        'worst fractal dimension': 'fractal_dimension_worst'
    }
    
    # Select the 10 features used by the model
    print("\n2. Selecting model features...")
    selected_cols = list(feature_mapping.keys())
    X = df[selected_cols].values
    y = data.target
    print(f"   ✓ Selected {len(selected_cols)} features")
    
    # Split the data (MUST match the split used during model training)
    print("\n3. Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✓ Training set: {X_train.shape[0]} samples")
    print(f"   ✓ Test set: {X_test.shape[0]} samples")
    
    # Apply preprocessing (log transformation)
    print("\n4. Applying preprocessing (log transformation)...")
    preprocessor = BreastCancerPreprocessor()
    X_train_preprocessed = preprocessor.transform(X_train)
    print(f"   ✓ Log transformation applied to {len(preprocessor.log_transform_features)} features")
    
    # Load the scaler and scale the preprocessed data
    print("\n5. Loading scaler and scaling data...")
    scaler_path = os.path.join('breast_cancer_model_deployment', 'breast_cancer_scaler.pkl')
    scaler = joblib.load(scaler_path)
    X_train_scaled = scaler.transform(X_train_preprocessed)
    print(f"   ✓ Data scaled to shape {X_train_scaled.shape}")
    
    # Validate the data
    print("\n6. Validating training data...")
    if X_train_scaled.shape[1] != 10:
        raise ValueError(f"Expected 10 features, got {X_train_scaled.shape[1]}")
    if X_train_scaled.shape[0] != 455:
        print(f"   ⚠ Warning: Expected 455 samples, got {X_train_scaled.shape[0]}")
    print(f"   ✓ Validation passed")
    
    # Save the scaled training data for SHAP
    print("\n7. Saving training data...")
    output_path = os.path.join('breast_cancer_model_deployment', 'train_data.pkl')
    joblib.dump(X_train_scaled, output_path)
    print(f"   ✓ Saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("✓ SHAP Training Data Generation Complete")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  - Shape: {X_train_scaled.shape}")
    print(f"  - Features: {X_train_scaled.shape[1]}")
    print(f"  - Samples: {X_train_scaled.shape[0]}")
    print(f"  - File: {output_path}")
    print(f"\nYou can now restart the Flask app to enable SHAP explainability.")


if __name__ == "__main__":
    try:
        generate_training_data()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
