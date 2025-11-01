"""
SHAP Explainability Module for Breast Cancer Prediction
This module provides model explainability using SHAP (SHapley Additive exPlanations)
to help users understand which features contribute to cancer risk predictions.
"""

import shap
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
import base64
from io import BytesIO


class BreastCancerExplainer:
    """
    SHAP-based explainer for breast cancer prediction model.
    Provides feature importance and individual prediction explanations.
    """
    
    def __init__(self, model, X_train_scaled, feature_names):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Trained machine learning model (Logistic Regression or Random Forest)
            X_train_scaled: Scaled training data for background distribution
            feature_names: List of feature names for visualization
        """
        self.model = model
        self.feature_names = feature_names
        
        # Initialize SHAP explainer based on model type
        # LinearExplainer is faster and more accurate for linear models
        try:
            # Try LinearExplainer first (works for Logistic Regression)
            self.explainer = shap.LinearExplainer(model, X_train_scaled)
            self.explainer_type = "linear"
            print("✓ SHAP LinearExplainer initialized successfully")
        except:
            # Fallback to KernelExplainer for other model types (e.g., Random Forest)
            # Use a sample of training data as background for efficiency
            background = shap.sample(X_train_scaled, min(100, len(X_train_scaled)))
            self.explainer = shap.KernelExplainer(model.predict_proba, background)
            self.explainer_type = "kernel"
            print("✓ SHAP KernelExplainer initialized successfully")
    
    def explain_prediction(self, X_scaled):
        """
        Generate SHAP values for a single prediction.
        
        Args:
            X_scaled: Scaled input features (preprocessed and scaled)
            
        Returns:
            shap_values: SHAP values for the prediction
        """
        # Calculate SHAP values
        if self.explainer_type == "linear":
            shap_values = self.explainer.shap_values(X_scaled)
        else:
            # For KernelExplainer, we get probabilities for both classes
            shap_values = self.explainer.shap_values(X_scaled)
            # Use SHAP values for malignant class (index 1)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        return shap_values
    
    def get_top_features(self, shap_values, top_n=5):
        """
        Get the top N most important features for a prediction.
        
        Args:
            shap_values: SHAP values array for a single prediction
            top_n: Number of top features to return
            
        Returns:
            List of tuples (feature_name, shap_value, contribution_type)
        """
        # Flatten SHAP values if needed
        if shap_values.ndim > 1:
            shap_values = shap_values.flatten()
        
        # Get absolute values for ranking importance
        abs_shap = np.abs(shap_values)
        
        # Get indices of top features by absolute importance
        top_indices = np.argsort(abs_shap)[::-1][:top_n]
        
        # Build result list with feature names and their contributions
        top_features = []
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            shap_val = shap_values[idx]
            
            # Determine if feature pushes toward malignant or benign
            if shap_val > 0:
                contribution = "malignant"
            else:
                contribution = "benign"
            
            top_features.append({
                'feature': feature_name.replace('_', ' ').title(),
                'shap_value': float(shap_val),
                'contribution': contribution,
                'abs_value': float(abs_shap[idx])
            })
        
        return top_features
    
    def create_waterfall_plot(self, shap_values, X_scaled, prediction):
        """
        Create a SHAP waterfall plot showing feature contributions.
        
        Args:
            shap_values: SHAP values for the prediction
            X_scaled: Scaled input features
            prediction: Model prediction (0=Benign, 1=Malignant)
            
        Returns:
            Base64-encoded PNG image of the waterfall plot
        """
        try:
            # Flatten arrays if needed
            if shap_values.ndim > 1:
                shap_values = shap_values.flatten()
            if X_scaled.ndim > 1:
                X_scaled = X_scaled.flatten()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get top 8 features for clarity
            abs_shap = np.abs(shap_values)
            top_indices = np.argsort(abs_shap)[::-1][:8]
            
            # Sort by SHAP value for waterfall effect
            sorted_indices = sorted(top_indices, key=lambda i: shap_values[i])
            
            # Prepare data for plotting
            features = [self.feature_names[i].replace('_', ' ').title() for i in sorted_indices]
            values = [shap_values[i] for i in sorted_indices]
            colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in values]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(features))
            ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
            
            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=10)
            ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11, fontweight='bold')
            ax.set_title(f'Feature Contributions - Predicted: {"Malignant" if prediction == 1 else "Benign"}',
                        fontsize=13, fontweight='bold', pad=15)
            
            # Add vertical line at zero
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            
            # Add grid for readability
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#ff6b6b', alpha=0.8, label='Pushes toward Malignant'),
                Patch(facecolor='#4ecdc4', alpha=0.8, label='Pushes toward Benign')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            print(f"Error creating waterfall plot: {e}")
            return None
    
    def create_feature_importance_plot(self, shap_values):
        """
        Create a bar chart showing overall feature importance.
        
        Args:
            shap_values: SHAP values for the prediction
            
        Returns:
            Base64-encoded PNG image of the feature importance plot
        """
        try:
            # Flatten if needed
            if shap_values.ndim > 1:
                shap_values = shap_values.flatten()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Get absolute importance
            abs_importance = np.abs(shap_values)
            
            # Sort features by importance
            sorted_indices = np.argsort(abs_importance)[::-1]
            
            # Plot top 10 features
            top_n = min(10, len(sorted_indices))
            top_indices = sorted_indices[:top_n]
            
            features = [self.feature_names[i].replace('_', ' ').title() for i in top_indices]
            importance = [abs_importance[i] for i in top_indices]
            
            # Create bar chart
            bars = ax.bar(range(len(features)), importance, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
            
            # Customize plot
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Absolute SHAP Value', fontsize=11, fontweight='bold')
            ax.set_title('Top 10 Most Important Features', fontsize=13, fontweight='bold', pad=15)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            print(f"Error creating importance plot: {e}")
            return None
