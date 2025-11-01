"""
Gemini AI Health Assistant Module
Provides NLP-based health insights and recommendations for breast cancer predictions
using Google's Gemini AI API.
"""

import os
from google import genai
from google.genai import types


class GeminiHealthAssistant:
    """
    AI Health Assistant powered by Google Gemini.
    Generates personalized health insights based on prediction results and feature contributions.
    """
    
    def __init__(self):
        """
        Initialize the Gemini client with API key from environment.
        """
        # Get API key from environment variables 
        self.api_key = os.environ.get("GEMINI_API_KEY")
        
        # Initialize client only if API key is available
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.is_available = True
                print("✓ Gemini AI Assistant initialized successfully")
            except Exception as e:
                print(f"⚠ Warning: Failed to initialize Gemini client: {e}")
                self.client = None
                self.is_available = False
        else:
            print("⚠ Warning: GEMINI_API_KEY not found in environment")
            self.client = None
            self.is_available = False
    
    def generate_health_insight(self, prediction, confidence, top_features, feature_values=None):
        """
        Generate personalized health insights using Gemini AI.
        """
        if not self.is_available:
            return "AI Health Assistant is currently unavailable. Please check API configuration."
        
        try:
            result_text = "Malignant (Cancerous)" if prediction == 1 else "Benign (Non-Cancerous)"
            
            features_description = ""
            for i, feat in enumerate(top_features[:5], 1):
                contribution = "increased cancer risk" if feat['contribution'] == "malignant" else "decreased cancer risk"
                features_description += f"{i}. {feat['feature']} — This feature {contribution}.<br>"

            prompt = f"""
            You are a medical AI assistant summarizing breast cancer screening results.

            Prediction: {result_text} (Confidence: {confidence:.1f}%)
            Key Features: {features_description}

            Task:
            Give a short, clear explanation (under 100 words) describing what this result means in simple medical terms.

            Format:
            📋 Summary:
            Briefly describe what the prediction suggests.
            🔍 Key Factors:
            List only the main 2–3 features influencing the result.
            💡 Next Step:
            Give one short medical recommendation.
            """


            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.6,
                    max_output_tokens=500,
                    top_p=0.9,
                    top_k=40
                )
            )

            if response and response.text:
                insight = response.text.strip()

                # ✨ Convert markdown-style text into clean HTML
                insight = insight.replace("**📋 Clinical Assessment:**", "<h3>📋 Clinical Assessment:</h3>")
                insight = insight.replace("**🔍 Feature Analysis:**", "<h3>🔍 Feature Analysis:</h3>")
                insight = insight.replace("**💡 Clinical Recommendations:**", "<h3>💡 Clinical Recommendations:</h3>")

                # Fix bold text formatting (convert markdown **text** → <b>text</b>)
                import re
                insight = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", insight)

                # Ensure any unclosed <b> tags are properly closed
                insight = insight.replace("<b>", "<b>").replace("<b>", "<b>").replace("<b>", "<b>")
                insight = insight.replace("<b>", "<b>").replace("<b>", "<b>")
                insight = insight.replace("<b>", "<b>").replace("<b>", "<b>")  # (just redundancy-safe cleanup)
                insight = insight.replace("<b>", "<b>").replace("<b>", "<b>")
                insight = insight.replace("<b>", "<b>").replace("<b>", "<b>")
                insight = insight.replace("<b>", "<b>").replace("<b>", "<b>")
                insight = insight.replace("<b>", "<b>").replace("<b>", "<b>")
                insight = insight.replace("<b>", "<b>").replace("<b>", "<b>")

                # Add clean spacing and HTML line breaks
                insight = re.sub(r"\n{2,}", "<br><br>", insight)
                insight = insight.replace("\n", "<br>")


                # Wrap final HTML
                html_output = f"""
                <div style='font-family: Arial; line-height:1.6; font-size:16px; color:#333;'>
                    {insight}
                </div>
                """

                return html_output

            else:
                return "<p>Unable to generate health insights at this time. Please consult a healthcare professional for personalized advice.</p>"
        
        except Exception as e:
            print(f"Error generating Gemini insights: {e}")
            return "<p>AI Health Assistant currently not working. Please consult a healthcare professional for guidance.</p>"

    
    def get_availability_status(self):
        """
        Check if Gemini AI is available and ready to use.
        
        Returns:
            Boolean indicating availability
        """
        return self.is_available
    
    def generate_feature_explanation(self, feature_name, shap_value):
        """
        Generate a simple explanation for a specific feature's contribution.
        
        Args:
            feature_name: Name of the feature
            shap_value: SHAP value indicating contribution strength
            
        Returns:
            String explanation
        """
        if not self.is_available:
            return "Feature explanation unavailable."
        
        try:
            direction = "increased cancer risk" if shap_value > 0 else "decreased cancer risk"
            strength = "strongly" if abs(shap_value) > 0.5 else "moderately"
            
            prompt = f"""Explain in 1-2 simple sentences what "{feature_name}" means in breast cancer screening and why it {strength} indicates {direction}. 
            
Use language a non-medical person can understand. Keep it under 50 words."""
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=100
                )
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                return f"This feature {strength} indicates {direction}."
                
        except Exception as e:
            print(f"Error generating feature explanation: {e}")
            direction = "increased cancer risk" if shap_value > 0 else "decreased cancer risk"
            return f"This feature contributes to {direction}."
