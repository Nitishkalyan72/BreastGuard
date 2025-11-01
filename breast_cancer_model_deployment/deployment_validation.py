"""
Deployment Validation Script for Breast Cancer Prediction Model
Tests the complete preprocessing and prediction pipeline
"""
import joblib
import numpy as np
import sys
import os

# Add current directory to path for preprocessor import
sys.path.insert(0, os.path.dirname(__file__))
from preprocessor import BreastCancerPreprocessor


def load_model_components():
    """Load all model components"""
    model = joblib.load('breast_cancer_model.pkl')
    scaler = joblib.load('breast_cancer_scaler.pkl')
    metadata = joblib.load('model_metadata.pkl')
    preprocessor = BreastCancerPreprocessor()
    return model, scaler, metadata, preprocessor


def test_prediction(model, scaler, preprocessor, features, description):
    """Test a single prediction"""
    # Apply preprocessing pipeline
    preprocessed = preprocessor.transform([features])
    scaled = scaler.transform(preprocessed)
    
    # Make prediction
    prediction = model.predict(scaled)[0]
    probabilities = model.predict_proba(scaled)[0]
    
    result = "Malignant (Cancerous)" if prediction == 1 else "Benign (Non-Cancerous)"
    confidence = probabilities[prediction] * 100
    
    print(f"\n{description}")
    print(f"  Input: {features}")
    print(f"  Result: {result}")
    print(f"  Confidence: {confidence:.2f}%")
    print(f"  Probabilities: [Benign: {probabilities[0]:.4f}, Malignant: {probabilities[1]:.4f}]")
    
    return prediction, probabilities


def main():
    print("=" * 70)
    print("Breast Cancer Model Deployment Validation")
    print("=" * 70)
    
    # Load components
    print("\n1. Loading model components...")
    model, scaler, metadata, preprocessor = load_model_components()
    print(f"   ✓ Model loaded: {model.__class__.__name__}")
    print(f"   ✓ Preprocessor loaded with {len(preprocessor.log_transform_features)} log-transform features")
    print(f"   ✓ Feature count: {len(metadata['features'])}")
    
    # Display features
    print("\n2. Model features (in order):")
    for i, feature in enumerate(metadata['features'], 1):
        log_marker = " [LOG]" if feature in preprocessor.log_transform_features else ""
        print(f"   {i}. {feature}{log_marker}")
    
    # Test cases
    print("\n3. Running validation tests...")
    print("-" * 70)
    
    # Test 1: Typical benign case
    benign_test = [0.08, 0.03, 12.0, 15.0, 80.0, 500.0, 0.08, 0.02, 0.16, 0.055]
    pred1, prob1 = test_prediction(model, scaler, preprocessor, benign_test, 
                                   "TEST 1: Typical Benign Case (Expected: Benign)")
    
    # Test 2: Typical malignant case
    malignant_test = [0.35, 0.40, 28.0, 35.0, 190.0, 2500.0, 0.25, 0.25, 0.35, 0.15]
    pred2, prob2 = test_prediction(model, scaler, preprocessor, malignant_test,
                                   "TEST 2: Typical Malignant Case (Expected: Malignant)")
    
    # Test 3: Borderline case
    borderline_test = [0.15, 0.10, 18.0, 22.0, 120.0, 1000.0, 0.12, 0.08, 0.22, 0.08]
    pred3, prob3 = test_prediction(model, scaler, preprocessor, borderline_test,
                                   "TEST 3: Borderline Case (Could be either)")
    
    # Test 4: Another benign case
    benign_test_2 = [0.10, 0.05, 14.0, 18.0, 95.0, 650.0, 0.09, 0.03, 0.18, 0.06]
    pred4, prob4 = test_prediction(model, scaler, preprocessor, benign_test_2,
                                   "TEST 4: Another Benign Case (Expected: Benign)")
    
    # Validation summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    tests_passed = 0
    if pred1 == 0 and prob1[0] > 0.7:
        print("✓ TEST 1 PASSED: Correctly predicted Benign")
        tests_passed += 1
    else:
        print("✗ TEST 1 FAILED: Did not predict Benign as expected")
    
    if pred2 == 1 and prob2[1] > 0.7:
        print("✓ TEST 2 PASSED: Correctly predicted Malignant")
        tests_passed += 1
    else:
        print("✗ TEST 2 FAILED: Did not predict Malignant as expected")
    
    print(f"○ TEST 3: Borderline case - Predicted {['Benign', 'Malignant'][pred3]}")
    tests_passed += 0.5
    
    if pred4 == 0 and prob4[0] > 0.7:
        print("✓ TEST 4 PASSED: Correctly predicted Benign")
        tests_passed += 1
    else:
        print("✗ TEST 4 FAILED: Did not predict Benign as expected")
    
    print(f"\nOVERALL RESULT: {tests_passed}/4 tests passed")
    
    if tests_passed >= 3:
        print("✓ MODEL VALIDATION SUCCESSFUL")
        return 0
    else:
        print("✗ MODEL VALIDATION FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
