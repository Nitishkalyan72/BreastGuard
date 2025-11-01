"""
Test script to verify SHAP and Gemini AI integration
Run this to test both features with sample medical data
"""

import requests
import json

# Test cases from the validation script
test_cases = {
    "benign_case": {
        "compactness_mean": 0.08,
        "concavity_mean": 0.03,
        "radius_worst": 12.0,
        "texture_worst": 15.0,
        "perimeter_worst": 80.0,
        "area_worst": 500.0,
        "smoothness_worst": 0.08,
        "concave_points_worst": 0.02,
        "symmetry_worst": 0.16,
        "fractal_dimension_worst": 0.055
    },
    "malignant_case": {
        "compactness_mean": 0.35,
        "concavity_mean": 0.40,
        "radius_worst": 28.0,
        "texture_worst": 35.0,
        "perimeter_worst": 190.0,
        "area_worst": 2500.0,
        "smoothness_worst": 0.25,
        "concave_points_worst": 0.25,
        "symmetry_worst": 0.35,
        "fractal_dimension_worst": 0.15
    }
}

def test_prediction(case_name, data):
    """Test a prediction case"""
    print(f"\n{'='*70}")
    print(f"Testing: {case_name.replace('_', ' ').title()}")
    print(f"{'='*70}")
    
    # Make prediction request
    url = "http://127.0.0.1:5000/predict"
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Prediction: {result['prediction']}")
        print(f"✓ Confidence: {result['confidence']}")
        print(f"✓ Probabilities: Benign={result['probabilities']['benign']}, Malignant={result['probabilities']['malignant']}")
        
        # Check SHAP results
        if 'shap_analysis' in result:
            print(f"\n✓ SHAP Analysis Available")
            print(f"  - Top features: {len(result['shap_analysis']['top_features'])}")
            print(f"  - Visualizations: {'Yes' if result['shap_analysis']['visualizations_available'] else 'No'}")
            
            print("\n  Top Contributing Features:")
            for i, feat in enumerate(result['shap_analysis']['top_features'], 1):
                print(f"    {i}. {feat['feature']} → {feat['contribution']}")
        else:
            print("\n⚠ SHAP Analysis not available")
        
        # Check Gemini results
        if 'health_insight' in result:
            print(f"\n✓ Gemini Health Insight Generated")
            print(f"\n{result['health_insight'][:200]}...")
        else:
            print("\n⚠ Gemini Health Insight not available")
            
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BreastGuard - SHAP & Gemini AI Integration Test")
    print("="*70)
    
    for case_name, data in test_cases.items():
        test_prediction(case_name, data)
    
    print(f"\n{'='*70}")
    print("Test Complete - Check the web UI to see visualizations!")
    print("="*70)
