<div align="center">

# 🧠 Building Your First ML Model

### Health Classification for Insurance Premium Pricing

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optional-blue?style=for-the-badge)](https://xgboost.readthedocs.io)

*A complete end-to-end machine learning pipeline — from exploratory data analysis to model comparison and deployment-ready prediction — packaged in a single, well-documented Jupyter Notebook.*

</div>

---

## 📋 Table of Contents

- [🧠 Building Your First ML Model](#-building-your-first-ml-model)
    - [Health Classification for Insurance Premium Pricing](#health-classification-for-insurance-premium-pricing)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 Overview](#-overview)
  - [💼 Business Context](#-business-context)
  - [📊 Dataset](#-dataset)
    - [Feature Summary](#feature-summary)
  - [🏗️ Pipeline Architecture](#️-pipeline-architecture)
  - [🤖 Models Compared](#-models-compared)
  - [📓 Notebook Walkthrough](#-notebook-walkthrough)
  - [🔑 Key Techniques](#-key-techniques)
  - [🚀 Quick Start](#-quick-start)
    - [1. Navigate to the project](#1-navigate-to-the-project)
    - [2. Create and activate a virtual environment](#2-create-and-activate-a-virtual-environment)
    - [3. Install dependencies](#3-install-dependencies)
    - [4. Place the dataset](#4-place-the-dataset)
    - [5. Launch the notebook](#5-launch-the-notebook)
  - [📦 Requirements](#-requirements)
  - [🔮 Sample Prediction](#-sample-prediction)
  - [📁 Project Structure](#-project-structure)

---

## 🎯 Overview

This project builds a **binary health classification model** for **Anova Insurance**, predicting whether an individual is **Healthy (0)** or **Unhealthy (1)** based on lifestyle, biometric, and medical features. The prediction directly supports premium pricing decisions — unhealthy individuals are associated with higher risk and therefore higher premiums.

The notebook demonstrates the full ML lifecycle:

1. **Data loading & inspection**
2. **Exploratory Data Analysis (EDA)** with rich visualizations
3. **Data preprocessing** (cleaning, imputation, scaling)
4. **Multi-model training & cross-validated comparison**
5. **Best-model evaluation** (classification report, confusion matrix, ROC curves)
6. **Feature importance analysis**
7. **Reusable prediction function** for new individuals

---

## 💼 Business Context

| Aspect | Detail |
|--------|--------|
| **Company** | Anova Insurance |
| **Objective** | Classify individuals as Healthy/Unhealthy for premium pricing |
| **Target Variable** | `Target` — 0 (Healthy), 1 (Unhealthy) |
| **Impact** | Accurate classification reduces mis-pricing risk and improves portfolio health |

---

## 📊 Dataset

**File:** `mDugQt7wQOKNNIAFjVku_Healthcare_Data_Preprocessed_FIXED.csv`

### Feature Summary

| Feature | Type | Description |
|---------|------|-------------|
| `Age` | Numeric | Age of the individual |
| `BMI` | Numeric | Body Mass Index |
| `Blood_Pressure` | Numeric | Blood pressure reading |
| `Cholesterol` | Numeric | Cholesterol level |
| `Glucose_Level` | Numeric | Blood glucose level |
| `Heart_Rate` | Numeric | Resting heart rate |
| `Sleep_Hours` | Numeric | Average daily sleep hours |
| `Exercise_Hours` | Numeric | Average daily exercise hours |
| `Water_Intake` | Numeric | Daily water intake (liters) |
| `Stress_Level` | Numeric | Self-reported stress level |
| `Smoking` | Categorical | Smoking status (encoded) |
| `Alcohol` | Categorical | Alcohol consumption level |
| `Diet` | Categorical | Diet quality indicator |
| `MentalHealth` | Categorical | Mental health status |
| `PhysicalActivity` | Categorical | Physical activity level |
| `MedicalHistory` | Binary | Pre-existing medical conditions |
| `Allergies` | Binary | Allergy presence |
| `Diet_Type_Vegan` | Binary | Vegan diet flag |
| `Diet_Type_Vegetarian` | Binary | Vegetarian diet flag |
| `Blood_Group_AB` | Binary | Blood group AB |
| `Blood_Group_B` | Binary | Blood group B |
| `Blood_Group_O` | Binary | Blood group O |
| **`Target`** | **Binary** | **0 = Healthy, 1 = Unhealthy** |

---

## 🏗️ Pipeline Architecture

```
Raw CSV Data
    │
    ▼
Load & Inspect ─── Shape, dtypes, missing values, descriptive stats
    │
    ▼
EDA ─── Target distribution, feature histograms, box plots by class,
    │   categorical breakdowns, correlation heatmap
    │
    ▼
Preprocessing
    ├── Fix negative ages (absolute value)
    ├── Convert boolean columns to int
    ├── Train/test split (80/20, stratified)
    ├── Median imputation (SimpleImputer)
    └── Standard scaling (StandardScaler)
    │
    ▼
Model Training & Comparison
    ├── 5-fold Stratified CV (ROC-AUC scoring)
    ├── Fit on training set
    └── Evaluate on held-out test set
    │
    ▼
Best Model Evaluation
    ├── Classification report (precision, recall, F1)
    ├── Confusion matrix
    └── ROC curves (all models overlaid)
    │
    ▼
Feature Importance ─── Tree-based importances / |coefficients|
    │
    ▼
Prediction Function ─── predict_health_status(input_dict) → label + probability
```

---

## 🤖 Models Compared

| Model | Library | Key Hyperparameters |
|-------|---------|-------------------|
| **Logistic Regression** | scikit-learn | `max_iter=1000` |
| **Decision Tree** | scikit-learn | Default (gini) |
| **Random Forest** | scikit-learn | `n_estimators=200` |
| **Gradient Boosting** | scikit-learn | `n_estimators=200` |
| **K-Nearest Neighbors** | scikit-learn | `n_neighbors=7` |
| **Support Vector Machine** | scikit-learn | `probability=True` |
| **XGBoost** *(optional)* | xgboost | `n_estimators=200`, `eval_metric='logloss'` |

All models are evaluated using:
- **5-fold Stratified Cross-Validation** (ROC-AUC)
- **Test Set Accuracy**
- **Test Set ROC-AUC**

---

## 📓 Notebook Walkthrough

| Section | Cell(s) | Description |
|---------|---------|-------------|
| **1. Imports** | 1 | Load all libraries (sklearn, xgboost, seaborn, matplotlib) |
| **2. Load & Inspect** | 2–5 | Read CSV, display shape, info, describe, missing values |
| **3. EDA** | 6–10 | Target distribution, histograms, box plots, categorical analysis, correlation heatmap |
| **4. Preprocessing** | 11–13 | Fix data errors, feature/target split, train/test split, imputation + scaling |
| **5. Model Training** | 14–16 | Train 7 models, cross-validate, compare on bar charts |
| **6. Best Model Evaluation** | 17–19 | Classification report, confusion matrix, ROC curves |
| **7. Feature Importance** | 20 | Horizontal bar chart of top 20 features |
| **8. Prediction Function** | 21 | Reusable `predict_health_status()` with example |
| **9. Summary** | 22 | Final performance comparison table |

---

## 🔑 Key Techniques

- **Stratified splitting** — Preserves class distribution in both train and test sets
- **Median imputation** — Robust to outliers compared to mean imputation
- **Standard scaling** — Essential for distance-based models (KNN, SVM) and regularized models (Logistic Regression)
- **Cross-validation** — 5-fold stratified CV prevents overfitting to a single train/test split
- **ROC-AUC scoring** — Threshold-independent metric, ideal for imbalanced binary classification
- **Feature importance** — Supports model interpretability for insurance stakeholders

---

## 🚀 Quick Start

### 1. Navigate to the project

```powershell
cd "L4"
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
# Optional: XGBoost support
pip install xgboost
```

### 4. Place the dataset

Ensure `mDugQt7wQOKNNIAFjVku_Healthcare_Data_Preprocessed_FIXED.csv` is in the `L4/` directory (same level as the notebook).

### 5. Launch the notebook

```powershell
jupyter notebook "Building your First ML Model.ipynb"
```

Or open directly in **VS Code** with the Jupyter extension.

---

## 📦 Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | 1.24+ | Numerical operations |
| `pandas` | 2.0+ | DataFrame manipulation |
| `matplotlib` | 3.7+ | Static plotting |
| `seaborn` | 0.12+ | Statistical visualizations |
| `scikit-learn` | 1.3+ | ML models, preprocessing, evaluation |
| `xgboost` | 1.7+ | *(Optional)* XGBoost classifier |
| `jupyter` | — | Notebook runtime |

---

## 🔮 Sample Prediction

```python
example = {
    'Age': 45, 'BMI': 28.5, 'Blood_Pressure': 130, 'Cholesterol': 210,
    'Glucose_Level': 105, 'Heart_Rate': 80, 'Sleep_Hours': 6, 'Exercise_Hours': 0.5,
    'Water_Intake': 1.5, 'Stress_Level': 7, 'Smoking': 2, 'Alcohol': 1, 'Diet': 0,
    'MentalHealth': 1, 'PhysicalActivity': 0, 'MedicalHistory': 1, 'Allergies': 0,
    'Diet_Type_Vegan': 0, 'Diet_Type_Vegetarian': 0,
    'Blood_Group_AB': 0, 'Blood_Group_B': 1, 'Blood_Group_O': 0
}

result = predict_health_status(example)
# {'prediction': 'Unhealthy', 'unhealthy_probability': 0.8723}
```

---

## 📁 Project Structure

```
L4/
├── README.md                              # This file
└── Building your First ML Model.ipynb     # Complete ML pipeline notebook
    ├── Section 1: Imports
    ├── Section 2: Load & Inspect Data
    ├── Section 3: Exploratory Data Analysis
    ├── Section 4: Data Preprocessing
    ├── Section 5: Model Training & Comparison
    ├── Section 6: Best Model Evaluation
    ├── Section 7: Feature Importance
    ├── Section 8: Prediction Function
    └── Section 9: Summary
```

---

<div align="center">

*Part of the [Pinnacle Projects](../README.md) portfolio — production-grade AI & ML systems.*

</div>
