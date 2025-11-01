from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import sys
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


app = Flask(__name__)

# Add the deployment directory to Python path for importing preprocessor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'breast_cancer_model_deployment'))
from preprocessor import BreastCancerPreprocessor

# Import SHAP explainability and Gemini AI modules
from shap_explainer import BreastCancerExplainer
from gemini_assistant import GeminiHealthAssistant

# Load model, scaler, metadata
model_path = os.path.join('breast_cancer_model_deployment', 'breast_cancer_model.pkl')
scaler_path = os.path.join('breast_cancer_model_deployment', 'breast_cancer_scaler.pkl')
metadata_path = os.path.join('breast_cancer_model_deployment', 'model_metadata.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
metadata = joblib.load(metadata_path)
preprocessor = BreastCancerPreprocessor()

feature_names = metadata["features"]
feature_names = [f.replace(' ', '_') for f in feature_names]
print("Loaded Features:", feature_names)
print("Preprocessing pipeline initialized with log transformation")


def generate_scientific_explanation(prediction, top_features):
    """
    Generate a concise scientific/medical explanation based on SHAP results.
    
    Args:
        prediction: 0 for Benign, 1 for Malignant
        top_features: List of top contributing features from SHAP
        
    Returns:
        Two-line scientific explanation
    """
    if not top_features or len(top_features) == 0:
        return None
    
    # Get the top 2-3 most important features
    top_malignant = [f for f in top_features[:3] if f['contribution'] == 'malignant']
    top_benign = [f for f in top_features[:3] if f['contribution'] == 'benign']
    
    if prediction == 1:  # Malignant
        # Focus on features pushing toward malignancy
        if top_malignant:
            feature_names_str = ", ".join([f['feature'].lower() for f in top_malignant[:2]])
            line1 = f"Higher {feature_names_str} indicate abnormal growth patterns suggesting malignancy."
        else:
            line1 = "Cellular features show characteristics associated with malignant tissue."
        
        line2 = "The model detected irregular cell morphology and increased cellular activity."
    
    else:  # Benign
        # Focus on features pushing toward benign
        if top_benign:
            feature_names_str = ", ".join([f['feature'].lower() for f in top_benign[:2]])
            line1 = f"Lower {feature_names_str} suggest normal, healthy tissue characteristics."
        else:
            line1 = "Cellular features show characteristics associated with benign tissue."
        
        line2 = "The model detected smooth and uniform cells suggesting non-cancerous tissue."
    
    return f"{line1}\n{line2}"

# Initialize SHAP explainer with training data
# CRITICAL: SHAP requires real training background data for accurate explanations
# Using synthetic/random data produces misleading feature importance that could
# misinform medical decisions - this is unacceptable in healthcare applications
try:
    # Try to load actual training data
    train_data_path = os.path.join('breast_cancer_model_deployment', 'train_data.pkl')
    
    if os.path.exists(train_data_path):
        # Load real training data
        X_train_scaled = joblib.load(train_data_path)
        print(f"✓ Loaded training data: {X_train_scaled.shape}")
        
        # Validate training data shape matches model expectations
        if X_train_scaled.shape[1] != len(feature_names):
            raise ValueError(f"Training data has {X_train_scaled.shape[1]} features but model expects {len(feature_names)}")
        
        # Initialize SHAP explainer with real background data
        shap_explainer = BreastCancerExplainer(model, X_train_scaled, feature_names)
        shap_available = True
        print("✓ SHAP Explainer initialized with real training data and ready")
    else:
        # FAIL SAFELY: Do not use synthetic data
        # In healthcare, showing misleading explanations is worse than showing none
        print("=" * 70)
        print("⚠ CRITICAL WARNING: Training data (train_data.pkl) not found!")
        print("⚠ SHAP explainability DISABLED to prevent misleading explanations")
        print("⚠ Using synthetic data would produce incorrect feature importance")
        print("⚠ To enable SHAP: Save training data as 'train_data.pkl'")
        print("=" * 70)
        shap_explainer = None
        shap_available = False

except Exception as e:
    print(f"⚠ ERROR: SHAP initialization failed: {e}")
    print("⚠ SHAP explainability will be unavailable for this session")
    shap_explainer = None
    shap_available = False

# Initialize Gemini AI Health Assistant
try:
    gemini_assistant = GeminiHealthAssistant()
    gemini_available = gemini_assistant.get_availability_status()
    if gemini_available:
        print("✓ Gemini AI Health Assistant ready")
    else:
        print("⚠ Gemini AI not available - check API key configuration")
except Exception as e:
    print(f"⚠ Warning: Gemini initialization failed: {e}")
    gemini_assistant = None
    gemini_available = False

@app.route('/')
def home():
    return render_template('home.html', feature_names=feature_names)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Collect features in the same order
#         print("Incoming JSON:", request.get_json())
#         features = [float(request.form.get(f)) for f in feature_names]
#         final_features = np.array([features])

#         # Transform & Predict
#         scaled = scaler.transform(final_features)
#         pred = model.predict(scaled)[0]
#         prob = model.predict_proba(scaled)[0][1]

#         result = "Malignant (Cancerous)" if pred == 1 else "Benign (Non-Cancerous)"
#         prob_percent = round(prob * 100, 2)

#         return render_template('home.html',
#                                feature_names=feature_names,
#                                prediction_text=f'Result: {result}',
#                                probability_text=f'Confidence: {prob_percent}%')
#     except Exception as e:
#         return render_template('home.html',
#                                feature_names=feature_names,
#                                prediction_text=f"Error: {str(e)}")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if request is JSON (Postman or API)
        if request.is_json:
            data = request.get_json()
            print("Incoming JSON:", data)
            features = [float(data.get(f)) for f in feature_names]
        else:
            # Otherwise get data from HTML form
            print("Incoming Form Data:", request.form)
            features = [float(request.form.get(f)) for f in feature_names]

        # Convert to array
        final_features = np.array([features])
        
        # Validate input: all features must be non-negative for log transformation
        if np.any(final_features < 0):
            raise ValueError("All feature values must be non-negative")

        # Apply preprocessing (log transformation) then scale
        preprocessed = preprocessor.transform(final_features)
        scaled = scaler.transform(preprocessed)
        
        # Predict
        pred = model.predict(scaled)[0]
        probabilities = model.predict_proba(scaled)[0]
        
        # Get confidence in the predicted class
        confidence = probabilities[pred] * 100
        
        # Also get individual probabilities for transparency
        prob_benign = probabilities[0] * 100
        prob_malignant = probabilities[1] * 100

        # Prepare result
        result = "Malignant (Cancerous)" if pred == 1 else "Benign (Non-Cancerous)"

        # ========== SHAP Explainability Integration ==========
        shap_data = None
        top_features = None
        waterfall_plot = None
        importance_plot = None
        
        if shap_available and shap_explainer:
            try:
                # Generate SHAP values for this prediction
                shap_values = shap_explainer.explain_prediction(scaled)
                
                # Get top contributing features
                top_features = shap_explainer.get_top_features(shap_values, top_n=5)
                
                # Create visualizations
                waterfall_plot = shap_explainer.create_waterfall_plot(shap_values, scaled, pred)
                importance_plot = shap_explainer.create_feature_importance_plot(shap_values)
                
                # Prepare SHAP data for response
                shap_data = {
                    'top_features': top_features,
                    'waterfall_plot': waterfall_plot,
                    'importance_plot': importance_plot
                }
                
                print("✓ SHAP analysis completed successfully")
            except Exception as e:
                print(f"⚠ SHAP analysis failed: {e}")
                shap_data = None
        
        # ========== Scientific Medical Explanation ==========
        scientific_explanation = None
        
        if top_features:
            try:
                # Generate scientific explanation from SHAP results
                scientific_explanation = generate_scientific_explanation(pred, top_features)
                print("✓ Scientific explanation generated from SHAP")
            except Exception as e:
                print(f"⚠ Scientific explanation generation failed: {e}")
        
        # ========== Gemini AI Health Insights ==========
        gemini_insight = None
        
        if gemini_available and gemini_assistant and top_features:
            try:
                # Generate personalized health insights using Gemini
                gemini_insight = gemini_assistant.generate_health_insight(
                    prediction=pred,
                    confidence=confidence,
                    top_features=top_features
                )
                print("✓ Gemini health insight generated successfully")
            except Exception as e:
                print(f"⚠ Gemini insight generation failed: {e}")
                gemini_insight = "AI Health Assistant is currently unavailable."
        
        # ========== Return Response ==========
        
        # If request came from API (JSON)
        if request.is_json:
            response_data = {
                "prediction": result,
                "confidence": f"{confidence:.2f}%",
                "probabilities": {
                    "benign": f"{prob_benign:.2f}%",
                    "malignant": f"{prob_malignant:.2f}%"
                }
            }
            
            # Add SHAP data if available
            if shap_data:
                response_data['shap_analysis'] = {
                    'top_features': top_features,
                    'visualizations_available': waterfall_plot is not None
                }
            
            # Add Gemini insight if available
            if gemini_insight:
                response_data['health_insight'] = gemini_insight
            
            return jsonify(response_data)
        # Save user input data
        form_data = {f: request.form.get(f) for f in feature_names}

        return render_template('home.html',
                            feature_names=feature_names,
                            form_data=form_data,  # 👈 Pass back filled data
                            prediction_text=f'Result: {result}',
                            probability_text=f'Confidence: {confidence:.2f}% (Benign: {prob_benign:.2f}%, Malignant: {prob_malignant:.2f}%)',
                            shap_available=shap_data is not None,
                            top_features=top_features,
                            waterfall_plot=waterfall_plot,
                            importance_plot=importance_plot,
                            scientific_explanation=scientific_explanation,
                            gemini_available=gemini_insight is not None,
                            gemini_insight=gemini_insight)

        # # Else return to web page (HTML)
        # return render_template('home.html',
        #                        feature_names=feature_names,
        #                        prediction_text=f'Result: {result}',
        #                        probability_text=f'Confidence: {confidence:.2f}% (Benign: {prob_benign:.2f}%, Malignant: {prob_malignant:.2f}%)',
        #                        # SHAP explainability data
        #                        shap_available=shap_data is not None,
        #                        top_features=top_features,
        #                        waterfall_plot=waterfall_plot,
        #                        importance_plot=importance_plot,
        #                        # Scientific explanation
        #                        scientific_explanation=scientific_explanation,
        #                        # Gemini AI insights
        #                        gemini_available=gemini_insight is not None,
        #                        gemini_insight=gemini_insight)

    except Exception as e:
        print("Error:", e)
        if request.is_json:
            return jsonify({"error": str(e)}), 400
        return render_template('home.html',
                               feature_names=feature_names,
                               prediction_text=f'Error: {e}',
                               probability_text='')


@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
