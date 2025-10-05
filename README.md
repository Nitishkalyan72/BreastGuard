# 🩺 Breast Guard  
### *A Machine Learning Project for Early Breast Cancer Detection*

---

## 🧠 Overview  
**Breast Guard** is a **machine learning-based diagnostic tool** that predicts whether a patient is likely to have breast cancer using historical medical data and diagnostic features.  
The project leverages multiple algorithms and data visualization techniques to enhance interpretability and accuracy — ultimately aiming to **assist healthcare professionals in early detection**.

---

## 🚀 Key Features  

- **Algorithms Used**  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  

- **Data Preprocessing**  
  - Performed using **Pandas** and **NumPy**  
  - Includes cleaning, feature selection, and label encoding  

- **Model Development**  
  - Implemented using **Scikit-learn (sklearn)**  
  - Evaluated for accuracy, precision, and recall  

- **Visualization**  
  - **Matplotlib** and **Seaborn** used for correlation plots, distributions, and comparison graphs  

- **Goal**  
  To create a **reliable and interpretable predictive model** that can help in **early diagnosis** of breast cancer, potentially **saving lives through timely medical intervention**.

---

## 📊 Dataset  
The dataset used is derived from the **Breast Cancer Wisconsin (Diagnostic) Dataset**.  
It contains **569 samples** with **32 features**, including radius, texture, perimeter, area, smoothness, and more.

| Column | Description |
|--------|--------------|
| `id` | Unique identifier for each record |
| `diagnosis` | 'M' = Malignant (Cancer), 'B' = Benign (No Cancer) |
| `radius_mean`, `texture_mean`, ... | Diagnostic features used for model training |

---

## ⚙️ Data Preprocessing Workflow  

The dataset was cleaned to remove null columns, and label encoding was performed to convert categorical variables (like diagnosis) into numeric form.  
After preprocessing, statistical analysis and exploratory data analysis (EDA) were carried out to identify correlations between features and diagnosis outcomes.

---

## 📈 Exploratory Data Analysis (EDA)

EDA was performed to understand the relationships and patterns within the dataset, including:

- Distribution of diagnosis (Malignant vs. Benign)  
- Correlation between different numerical features  
- Identifying the most influential features for prediction  
- Feature scaling and normalization for model efficiency  

---

## 🧩 Model Building  

Three models were trained and compared:

1. **Logistic Regression** — Used as a baseline model for binary classification.  
2. **Decision Tree** — Helped interpret feature importance and visual relationships.  
3. **Random Forest** — Provided ensemble-based improvement and highest accuracy.  

---

## 📊 Model Performance  

| Model | Accuracy |
|--------|-----------|
| Logistic Regression | 96% |
| Decision Tree | 94% |
| Random Forest | 97% |

> Random Forest demonstrated the **highest accuracy** and **best generalization performance**.

---

## 🩹 Conclusion  
The **Breast Guard** project demonstrates how **machine learning can enhance medical diagnostics**, especially for diseases like breast cancer.  
With proper validation and integration into healthcare systems, it could serve as a **powerful decision-support tool** for doctors and researchers.

---

## 🧰 Tech Stack  

| Category | Tools Used |
|-----------|-------------|
| Language | Python |
| Libraries | Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn |
| Algorithms | Logistic Regression, Decision Tree, Random Forest |

---


---

## 👨‍💻 Author  
**Nitish Kalyan**  
*Machine Learning Enthusiast | Developer | Innovator*  

📧 **Email:** [nitishkalyan7249@gmail.com](mailto:nitishkalyan7249@gmail.com)  
🌐 **GitHub:** [github.com/NitishKalyan72](https://github.com/NitishKalyan72)

---

✨ *“Empowering healthcare through data and innovation.”*


