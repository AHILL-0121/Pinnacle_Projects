# 🏥 Health Status Classification — Anova Insurance (Project 9)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)]()

*Binary health-risk classifier to power insurance premium pricing decisions — with 6 algorithms compared and a Gradient Boosting champion achieving AUC-ROC ≈ 0.95.*

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Workflow](#-workflow)
- [Models Evaluated](#-models-evaluated)
- [Results](#-results)
- [Feature Importance](#-feature-importance)
- [Business Recommendations](#-business-recommendations)
- [Quick Start](#-quick-start)
- [Dependencies](#-dependencies)

---

## 🎯 Overview

**Anova Insurance** needs to optimize premium pricing based on applicant health risk. This notebook builds and compares six classification algorithms to predict whether an individual is **Healthy (0)** or **Unhealthy (1)**, enabling:

- Eligibility screening for health insurance coverage
- Risk-tiered premium rate determination
- Data-driven underwriting decisions

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Rows** | 10,000 |
| **Columns** | 23 (after preprocessing & encoding) |
| **Target** | `Target` — Binary: 0 = Healthy, 1 = Unhealthy |
| **Source** | Anova Insurance internal health survey data |

---

## 🔄 Workflow

```
Raw Data
   │
   ▼
1. Import Libraries
   │
   ▼
2. Load & Inspect Data
   │  Shape, dtypes, missing values, duplicates, class balance
   │
   ▼
3. Exploratory Data Analysis (EDA)
   │  Distributions, correlation heatmap, target imbalance
   │
   ▼
4. Data Preprocessing
   │  Null imputation, encoding, outlier treatment, scaling (StandardScaler)
   │
   ▼
5. Feature Engineering & Selection
   │  ANOVA F-test + correlation analysis → feature ranking
   │
   ▼
6. Model Building (6 algorithms trained)
   │
   ▼
7. Model Evaluation & Comparison
   │  Accuracy, F1, Precision, Recall, AUC-ROC, 5-Fold CV
   │
   ▼
8. Best Model Deep Dive (Gradient Boosting)
   │  Confusion matrix, ROC curve, threshold analysis
   │
   ▼
9. Feature Importance
   │  RF + GB averaged importance rankings
   │
   ▼
10. Conclusion & Business Recommendations
```

---

## 🤖 Models Evaluated

| # | Model | Type |
|---|-------|------|
| 1 | Logistic Regression | Linear |
| 2 | Decision Tree | Tree-based |
| 3 | Random Forest (200 estimators) | Ensemble — Bagging |
| 4 | **Gradient Boosting** (200 estimators, lr=0.1) | **Ensemble — Boosting** ✅ |
| 5 | K-Nearest Neighbors (k=7) | Instance-based |
| 6 | Support Vector Machine (RBF kernel) | Kernel-based |

All models trained on **StandardScaler**-normalized features; cross-validated with **5-Fold StratifiedKFold**.

---

## 📈 Results

**Best Model: Gradient Boosting Classifier**

| Metric | Score |
|--------|-------|
| **Accuracy** | ~0.8780 |
| **AUC-ROC** | ~0.9502 |
| **F1 Score** | ~0.88 |
| **Precision** | ~0.88 |
| **Recall** | ~0.88 |
| **5-Fold CV Accuracy** | ~0.875+ |

> **AUC-ROC ≈ 0.95** indicates excellent discriminative power between healthy and unhealthy applicants.

### Evaluation Artifacts

- Comparative bar chart across all 6 models and 5 metrics
- ROC curves for all models on a single axes
- Confusion matrix (counts + normalized) for best model
- 5-Fold cross-validation horizontal bar chart

---

## 🔍 Feature Importance

Top predictors (averaged RF + Gradient Boosting importances):

1. Key health biomarkers (BMI, blood pressure, cholesterol proxies)
2. Age-related risk factors
3. Lifestyle indicators

> Full ranked chart with top-10 features is generated in `Section 9` of the notebook.

---

## 💼 Business Recommendations

1. **Deploy Gradient Boosting** for automated applicant risk scoring
2. Use a **0.95 AUC** model to establish three premium tiers: Low / Medium / High risk
3. **Lower the decision threshold** below 0.5 to increase Recall for the Unhealthy class — false negatives (unhealthy misclassified as healthy) carry the highest financial risk for the insurer
4. Schedule **quarterly model retraining** as health-data distributions shift with population age

---

## 🚀 Quick Start

### Prerequisites

```powershell
cd "L4/Building your First ML Model"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt   # or install dependencies below
```

### Install Dependencies

```powershell
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook

```powershell
jupyter notebook "anova_insurance_health_classification(1).ipynb"
```

Or open it directly in VS Code with the Jupyter extension.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, results DataFrame |
| `numpy` | Numerical operations |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualizations, heatmaps |
| `scikit-learn` | All ML models, preprocessing, metrics, cross-validation |
| `jupyter` | Notebook execution environment |

---

## 📄 License & Author

**License:** MIT  
**Author:** AHILL S  
**Part of:** [Pinnacle Projects Portfolio](../../README.md)
