"""
Preprocessing pipeline for breast cancer prediction model.
This module ensures consistent preprocessing between training and deployment.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BreastCancerPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom preprocessor that applies log transformation to skewed features
    before scaling. This must match the preprocessing done during training.
    """
    
    def __init__(self):
        # Features that received log transformation during training
        self.log_transform_features = [
            'compactness_mean', 'concavity_mean', 'radius_worst',
            'perimeter_worst', 'area_worst', 'symmetry_worst',
            'fractal_dimension_worst'
        ]
        
        # All features in order (must match training order)
        self.feature_names = [
            'compactness_mean', 'concavity_mean', 'radius_worst',
            'texture_worst', 'perimeter_worst', 'area_worst',
            'smoothness_worst', 'concave points_worst', 'symmetry_worst',
            'fractal_dimension_worst'
        ]
        
        # Indices of features that need log transformation
        self.log_indices = [i for i, feat in enumerate(self.feature_names) 
                           if feat in self.log_transform_features]
        
    def fit(self, X, y=None):
        """Fit method (no-op, included for sklearn compatibility)"""
        return self
    
    def transform(self, X):
        """
        Apply log transformation to skewed features.
        
        Args:
            X: Input features (array-like or DataFrame)
            
        Returns:
            Transformed features with log1p applied to skewed features
        """
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X_transformed = X.values.copy()
        else:
            X_transformed = np.array(X).copy()
        
        # Apply log1p transformation to specified features
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X_transformed[:, idx])
        
        return X_transformed
    
    def get_feature_names(self):
        """Return the expected feature names in order"""
        return self.feature_names


# Test the preprocessor
if __name__ == "__main__":
    import joblib
    
    # Load the scaler
    scaler = joblib.load('breast_cancer_scaler.pkl')
    model = joblib.load('breast_cancer_model.pkl')
    
    # Create preprocessor
    preprocessor = BreastCancerPreprocessor()
    
    # Test with sample data (realistic benign case)
    test_benign = [[0.1, 0.05, 12, 15, 80, 500, 0.08, 0.02, 0.16, 0.055]]
    
    print("Testing preprocessor:")
    print(f"Original input: {test_benign[0]}")
    
    # Apply preprocessing
    transformed = preprocessor.transform(test_benign)
    print(f"After log transform: {transformed[0]}")
    
    # Scale
    scaled = scaler.transform(transformed)
    print(f"After scaling: {scaled[0]}")
    
    # Predict
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]
    
    print(f"\nPrediction: {prediction} (0=Benign, 1=Malignant)")
    print(f"Probability: {probability}")
