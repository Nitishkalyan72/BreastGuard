# Breast Cancer Model Preprocessing Guide

## Overview
This document explains the complete preprocessing pipeline for the breast cancer prediction model, including the critical log transformation step that must be applied consistently during both training and deployment.

## Preprocessing Pipeline Steps

The preprocessing pipeline consists of the following steps **in this exact order**:

### 1. Feature Selection
**Purpose**: Remove weak predictors and redundant features to reduce model complexity.

**Removed Features**:
- **Weak predictors** (low correlation with diagnosis): `fractal_dimension_se`, `symmetry_se`, `texture_se`, `fractal_dimension_mean`, `smoothness_se`
- **Redundant features** (high multicollinearity): `radius_mean`, `radius_se`, `area_mean`, `area_se`, `concave points_mean`, and others

**Retained Features** (10 total):
1. compactness_mean
2. concavity_mean  
3. radius_worst
4. texture_worst
5. perimeter_worst
6. area_worst
7. smoothness_worst
8. concave points_worst
9. symmetry_worst
10. fractal_dimension_worst

### 2. Data Splitting
**Purpose**: Separate data into training and test sets before any preprocessing to prevent data leakage.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Important**: All subsequent preprocessing is fitted on training data only.

### 3. Class Imbalance Handling (Training Data Only)
**Purpose**: Address class imbalance to prevent bias toward the majority class.

**Method**: SMOTE (Synthetic Minority Oversampling Technique)
- Creates synthetic samples of the minority class (Malignant)
- Applied **only** to training data to maintain test set realism
- Original ratio: ~1.68:1 (Benign:Malignant)
- After SMOTE: 1:1 balanced

**Critical Note**: This step is only used during model training, not during deployment/prediction.

### 4. Log Transformation ⚠️ CRITICAL STEP
**Purpose**: Reduce skewness in heavily skewed features to improve model performance.

**Method**: Apply `np.log1p(x)` transformation to the following 7 features:
1. compactness_mean
2. concavity_mean
3. radius_worst
4. perimeter_worst
5. area_worst
6. symmetry_worst
7. fractal_dimension_worst

**Why log1p?**
- `log1p(x) = log(1 + x)` handles zero values gracefully
- Reduces right skewness and makes distributions more normal
- Improves gradient descent convergence for linear models

**DEPLOYMENT REQUIREMENT**: This transformation **MUST** be applied to all input data before scaling and prediction. The `preprocessor.py` module handles this automatically.

### 5. Feature Scaling
**Purpose**: Normalize features to the same scale for distance-based algorithms.

**Method**: StandardScaler (z-score normalization)
```python
scaled_feature = (feature - mean) / std
```

**Important**: 
- The scaler is fitted on log-transformed training data
- During deployment, data must be log-transformed **before** scaling
- The saved `breast_cancer_scaler.pkl` expects log-transformed input

## Deployment Pipeline

### Correct Order for Predictions:
```python
# 1. Load components
from preprocessor import BreastCancerPreprocessor
preprocessor = BreastCancerPreprocessor()
scaler = joblib.load('breast_cancer_scaler.pkl')
model = joblib.load('breast_cancer_model.pkl')

# 2. Get input features (10 values in correct order)
features = [...]  # User input

# 3. Apply log transformation
preprocessed = preprocessor.transform([features])

# 4. Apply scaling
scaled = scaler.transform(preprocessed)

# 5. Make prediction
prediction = model.predict(scaled)
probability = model.predict_proba(scaled)
```

### Common Pitfall ⚠️
**WRONG**: Applying scaler directly to raw input
```python
# This will produce incorrect predictions!
scaled = scaler.transform([features])  # Missing log transformation
prediction = model.predict(scaled)
```

**RIGHT**: Apply log transformation first
```python
# Correct approach
preprocessed = preprocessor.transform([features])
scaled = scaler.transform(preprocessed)
prediction = model.predict(scaled)
```

## Model Performance

With correct preprocessing pipeline:
- **Accuracy**: 96.49%
- **Precision**: 95.65%
- **Recall (Sensitivity)**: 95.65%
- **F1-Score**: 95.65%
- **ROC-AUC**: 0.963

## Input Validation

All input features must be:
- **Non-negative** (required for log transformation)
- **In the correct order** (as listed in the feature list)
- **Complete** (all 10 features required)

## Clinical Interpretation

### Label Encoding
- **0 = Benign** (Non-Cancerous)
- **1 = Malignant** (Cancerous)

### Feature Ranges (typical values before transformation)
- **Compactness**: 0.05 - 0.40
- **Concavity**: 0.0 - 0.50
- **Radius worst**: 7 - 36 mm
- **Texture worst**: 10 - 50
- **Perimeter worst**: 50 - 250 mm
- **Area worst**: 180 - 4000 mm²
- **Smoothness worst**: 0.07 - 0.25
- **Concave points worst**: 0.0 - 0.30
- **Symmetry worst**: 0.15 - 0.40
- **Fractal dimension worst**: 0.05 - 0.20

## Files Reference

- `preprocessor.py` - Implements the log transformation pipeline
- `breast_cancer_model.pkl` - Trained Logistic Regression model
- `breast_cancer_scaler.pkl` - Fitted StandardScaler (expects log-transformed input)
- `model_metadata.pkl` - Model metadata and feature names
- `deployment_validation.py` - Validation script to test the complete pipeline

## Testing

Run the validation script to verify the preprocessing pipeline:
```bash
cd breast_cancer_model_deployment
python deployment_validation.py
```

Expected output: All tests should pass with correct predictions for benign and malignant cases.

---

**Last Updated**: October 31, 2025  
**Version**: 1.0
