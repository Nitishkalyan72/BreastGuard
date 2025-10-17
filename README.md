# 🧠 Breast Cancer Prediction using PCA & LDA

## 📘 Project Overview
This project focuses on analyzing the **Breast Cancer Wisconsin (Diagnostic) Dataset** to predict whether a tumor is **benign (B)** or **malignant (M)**.  
The main goal is to apply **dimensionality reduction techniques** — **Principal Component Analysis (PCA)** and **Linear Discriminant Analysis (LDA)** — to simplify the data, improve interpretability, and enhance model performance.

---

## 🎯 Objective
To explore how **PCA** and **LDA** can be used to:
- Reduce the dataset’s dimensionality while preserving essential information.
- Handle **class imbalance** using class weights.
- Visualize and interpret data patterns for better model understanding.
- Compare **unsupervised (PCA)** vs **supervised (LDA)** approaches in feature extraction.

---

## 📊 Dataset Information
The **Breast Cancer Wisconsin (Diagnostic) Dataset** contains **30 numerical features** computed from digitized images of fine needle aspirate (FNA) of breast masses.  
These describe characteristics like **radius**, **texture**, **smoothness**, **compactness**, and **concavity**.

| Feature Type | Example Features |
|---------------|------------------|
| Mean Features | radius_mean, texture_mean, perimeter_mean |
| Worst Features | radius_worst, area_worst, smoothness_worst |
| Standard Error Features | radius_se, texture_se, area_se |

---

## 🧩 Techniques Used

### 🌀 Principal Component Analysis (PCA)
PCA was applied to:
- **Reduce** the high-dimensional dataset to fewer **principal components**.
- **Preserve variance** while removing redundant features.
- **Transform** correlated variables into new uncorrelated components.

#### Steps Involved:
1. **Data Standardization** using `StandardScaler`
2. **Covariance Matrix** computation
3. **Eigenvalue & Eigenvector** extraction
4. **Principal Component Selection**
5. **Variance Visualization** through:
   - **Scree Plot**
   - **Cumulative Explained Variance Plot**

#### Evaluation Metrics:
- **Explained Variance Ratio**
- **Cumulative Explained Variance**
- **Reconstruction Error**

These metrics help assess how much information each principal component retains from the original dataset.

---

### 🧭 Linear Discriminant Analysis (LDA)
**LDA** was discussed as a **supervised** alternative to PCA.  
While PCA focuses on **maximizing variance**, LDA focuses on **maximizing class separability**.

#### Key Differences:
| Aspect | PCA | LDA |
|--------|-----|-----|
| Type | Unsupervised | Supervised |
| Goal | Maximize variance | Maximize class separation |
| Output | Principal Components | Discriminant Components |
| Use Case | Feature reduction | Classification improvement |

LDA can also be extended for **multiclass classification**, though it cannot create a perfectly linear separation if data overlap exists.

---

## 📈 Visualization
- **Pairplots** were used to visualize relationships between features and class distributions.  
- **Scree and Cumulative Variance Plots** helped identify the optimal number of components to retain.  
- **2D PCA Projection** visualized how well PCA separated malignant and benign classes.

---

## 🧮 Tools & Libraries Used
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

---

## 💡 Key Learnings
- PCA effectively reduces dimensionality while preserving essential variance.  
- Class imbalance can be handled using **class weight dictionaries** for better model prediction.  
- Visual interpretation through PCA helps understand feature contribution.  
- LDA, though supervised, complements PCA in enhancing classification clarity.

---

## 🧾 Conclusion
The notebook concluded that **PCA** is an effective method for dimensionality reduction in breast cancer data, retaining key information and improving model efficiency.  
**LDA**, though briefly discussed, provides a valuable perspective on supervised dimensionality reduction by enhancing **class separability**.

> “Dimensionality reduction is not about losing data — it’s about keeping the most meaningful information.”

---

## 🙌 Acknowledgment
Thanks for reviewing this project!  
This notebook aims to make complex concepts like PCA and LDA easier to understand through step-by-step implementation and clear visualizations.
