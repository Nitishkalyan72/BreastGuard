# 🩺 BreastGuard – AI-Powered Breast Cancer Prediction System  

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-black)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![AI Powered](https://img.shields.io/badge/🤖-AI%20Powered-blueviolet) 
![NLP Enabled](https://img.shields.io/badge/NLP-Integrated-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/Use-Educational-lightgrey)

---

## 🌸 Overview  
**BreastGuard** is an AI-based web application that predicts the likelihood of breast cancer using clinical cell nucleus data.  
It leverages a **Logistic Regression model** with explainable AI (SHAP) and **Gemini-powered NLP health insights** to make medical predictions transparent and human-understandable.

🌐 **Live Demo:** [https://breastguard-ai.onrender.com](https://breastguard-ai.onrender.com)

---

## 🧠 Key Features  

🔍 **Explainable AI (SHAP Integration)**  
- Highlights which features influence predictions  
- Generates feature contribution & importance plots  
- Makes decisions transparent and interpretable  

🤖 **Gemini AI Health Assistant**  
- Uses **Google Gemini API** for human-like explanations   

⚕️ **Scientific Medical Explanation**  
- Provides a concise, clinical summary based on SHAP insights  
- Differentiates between professional and friendly AI advice  

🎨 **Modern UI & Design**  
- Built with **Bootstrap 5.3**, fully responsive  
- Color-coded results:  
  - 🩸 *Malignant* → Red/Pink gradient  
  - 💧 *Benign* → Blue/Cyan gradient  
- Smooth user flow: *Input → Predict → Explain → Advise*  

📊 **Machine Learning Pipeline**  
- Multiple ML models tested (Logistic Regression, Random Forest, etc.)  
- **Logistic Regression** selected for deployment based on performance and interpretability.    
- Includes **log transformation** & **scaling** for accuracy and consistency  




---

## 🧩 Model Details  

### 🎯 Algorithm  
- **Logistic Regression** (Scikit-learn)  
The model uses 10 features for prediction:
1. Compactness Mean
2. Concavity Mean
3. Radius Worst
4. Texture Worst
5. Perimeter Worst
6. Area Worst
7. Smoothness Worst
8. Concave Points Worst
9. Symmetry Worst
10. Fractal Dimension Worst


### 📈 Model Performance  
| Metric | Score |
|:--|:--|
| Accuracy | 96.49% |
| Precision | 95.65% |
| Recall | 95.65% |
| F1-Score | 95.65% |
| ROC-AUC | 0.963 |

---

## 🧪 How It Works  
1️⃣ User enters medical feature values via form  
2️⃣ Data passes through **`preprocessor.py`** → applies `log1p` transformation & scaling  
3️⃣ Model predicts benign/malignant outcome  
4️⃣ **SHAP & Gemini AI** modules explain *why* that result occurred  

🌐 **Live Demo:** [https://breastguard-ai.onrender.com](https://breastguard-ai.onrender.com)

---

## ⚙️ Project Structure  

- `app.py` — Main Flask application  

### 📁 Templates  
- `templates/home.html` — Frontend HTML form for predictions  

### 🤖 Machine Learning & Explainability  
- `breast_cancer_model_deployment/` — Contains trained ML models and preprocessing  
  - `breast_cancer_model.pkl` — Trained Logistic Regression model  
  - `breast_cancer_scaler.pkl` — Feature scaler (expects log-transformed input)  
  - `model_metadata.pkl` — Metadata including feature names  
  - `preprocessor.py` — Handles log transformation and scaling  
  - `deployment_validation.py` — Validation and testing of preprocessing pipeline  
  - `PREPROCESSING_GUIDE.md` — Documentation for preprocessing steps  
- `shap_explainer.py` — SHAP Explainability module for feature contribution analysis  
- `gemini_assistant.py` — Google Gemini API integration for AI Health Assistant  
- `generate_training_data.py` — Script to generate or augment dataset  
- `test_prediction.py` — Script to locally test model predictions  
- `BreastGuard.ipynb` — Core notebook for model training, evaluation, and experimentation  

### ⚙️ Configuration  
- `requirements.txt` — Python dependencies  
- `README.md` — Main project documentation  


## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Nitishkalyan72/BreastGuard.git
cd BreastGuard
```
### 2️⃣ Create and Activate Conda Environment
```bash
conda create -p .venv python=3.11 -y
conda activate ./.venv
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### ▶️ Run the Application
```bash
python app.py
```

Then open your browser and navigate to:  
👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---


## 🧰 Tech Stack
| Layer | Technology |
|--------|-------------|
| Frontend | HTML, CSS, Bootstrap 5.3 |
| Backend | Flask (Python 3.11) |
| ML | Scikit-learn, SHAP |
| NLP | Google Gemini AI |
| Environment | Conda |
---

## ⚠️ Disclaimer
This project is for **research and educational purposes only**.  
It is **not a substitute** for professional medical diagnosis.  
Always consult **certified healthcare providers** for medical concerns.

---

## 💬 Contribute
Pull requests are welcome!  
For major changes, open an issue first to discuss what you’d like to modify.

---

## 🧑‍💻 Author
**Nitish Kalyan**  
*Data Scientist*  
