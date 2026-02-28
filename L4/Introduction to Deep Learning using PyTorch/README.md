# 💧 Water Quality Prediction — Deep Learning Neural Networks (Project 11)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)]()

*End-to-end deep learning notebook that trains two Multi-Layer Perceptron (MLP) neural networks on India's CPCB water quality dataset — one for WQI regression and one for multi-class quality classification.*

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Architecture](#-architecture)
- [Workflow](#-workflow)
- [Models](#-models)
- [Evaluation Metrics](#-evaluation-metrics)
- [Visualizations](#-visualizations)
- [Quick Start](#-quick-start)
- [Dependencies](#-dependencies)

---

## 🎯 Overview

This notebook solves **two complementary prediction tasks** on Indian river and groundwater data collected by the Central Pollution Control Board (CPCB):

| Task | Type | Output |
|------|------|--------|
| **WQI Prediction** | Regression | Continuous Water Quality Index score |
| **Quality Classification** | Multi-class | Categorical water quality label (5 classes) |

Together, these models can assist environmental agencies in monitoring water safety, prioritizing remediation efforts, and forecasting quality degradation across monitoring stations.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | Central Pollution Control Board (CPCB), India |
| **Records** | 19,029 samples |
| **Period** | 2019–2022 |
| **geography** | Monitoring stations across India |
| **Feature Count** | 15 physicochemical indicators + Year |
| **Targets** | `WQI` (continuous) and `Water Quality Classification` (5 classes) |
| **File** | `water_quality.csv` |

### Class Distribution

The classification target contains **5 water quality classes** ranging from safe to severely polluted (e.g., Excellent, Good, Poor, Very Poor, Unsuitable for Drinking). Class imbalance is visualized in the EDA section.

---

## ⚗️ Features

| Feature | Description |
|---------|-------------|
| `Year` | Year of measurement (2019–2022) |
| `pH` | Acidity/alkalinity of water |
| `EC` | Electrical Conductivity (µS/cm) |
| `CO3` | Carbonate concentration |
| `HCO3` | Bicarbonate concentration |
| `Cl` | Chloride concentration (mg/L) |
| `SO4` | Sulphate concentration (mg/L) |
| `NO3` | Nitrate concentration (mg/L) |
| `TH` | Total Hardness |
| `Ca` | Calcium concentration (mg/L) |
| `Mg` | Magnesium concentration (mg/L) |
| `Na` | Sodium concentration (mg/L) |
| `K` | Potassium concentration (mg/L) |
| `F` | Fluoride concentration (mg/L) |
| `TDS` | Total Dissolved Solids (mg/L) |

All features are standardized with `StandardScaler` before training (zero mean, unit variance).

---

## 🧠 Architecture

Both models use the **same deep MLP architecture**, differing only in output layer and loss function:

```
Input Layer (15 features)
        │
        ▼
  Dense(512, ReLU)
        │
        ▼
  Dense(256, ReLU)
        │
        ▼
  Dense(128, ReLU)
        │
        ▼
   Dense(64, ReLU)
        │
        ├──────────────────────────────┐
        ▼                              ▼
  Dense(1, Linear)            Dense(5, Softmax)
  [WQI Regression]            [Classification]
```

| Hyperparameter | Value |
|----------------|-------|
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Batch Size** | 256 |
| **Max Iterations** | 500 |
| **Early Stopping** | Yes (`n_iter_no_change=20`) |
| **Validation Split** | 10% |
| **Random State** | 42 |

---

## 🔄 Workflow

```
water_quality.csv
        │
        ▼
1. Import & Load Data (19,029 records × 16 columns)
        │
        ▼
2. Exploratory Data Analysis
   ├── Class distribution bar chart
   ├── WQI histogram
   ├── Feature correlation heatmap
   ├── Feature–WQI correlation ranking
   └── WQI box plots by quality class
        │
        ▼
3. Data Preprocessing
   ├── Drop rows with missing values → clean dataset
   ├── Label-encode classification target
   ├── Train/test split (80/20)
   │    ├── Regression: random split
   │    └── Classification: stratified split
   └── StandardScaler normalization
        │
        ▼
4. Model 1 — WQI Regression MLP
   ├── Train: MLPRegressor (15→512→256→128→64→1)
   ├── Evaluate: R², RMSE, MAE
   ├── Plot: loss curve + actual vs. predicted scatter
   └── Plot: residual analysis (residuals vs. predicted + histogram)
        │
        ▼
5. Model 2 — Classification MLP
   ├── Train: MLPClassifier (15→512→256→128→64→5)
   ├── Evaluate: Accuracy, F1-Macro, F1-Weighted
   ├── Print: full classification report per class
   ├── Plot: loss curve + validation accuracy
   ├── Plot: confusion matrix heatmap
   └── Plot: per-class precision / recall / F1 bar chart
        │
        ▼
6. Final Summary — side-by-side performance table
```

---

## 🤖 Models

### Model 1 — WQI Regression

| Setting | Value |
|---------|-------|
| **Class** | `sklearn.neural_network.MLPRegressor` |
| **Loss** | Mean Squared Error (MSE) |
| **Architecture** | `(512, 256, 128, 64)` hidden layers |
| **Train/Test Split** | 80% / 20% (random) |

**Metrics reported:**

| Metric | Description |
|--------|-------------|
| R² Train | Coefficient of determination on training set |
| R² Test | Coefficient of determination on held-out test set |
| RMSE Test | Root Mean Squared Error |
| MAE Test | Mean Absolute Error |

---

### Model 2 — Water Quality Classification

| Setting | Value |
|---------|-------|
| **Class** | `sklearn.neural_network.MLPClassifier` |
| **Loss** | Cross-Entropy |
| **Output** | 5-class Softmax |
| **Train/Test Split** | 80% / 20% (stratified) |

**Metrics reported:**

| Metric | Description |
|--------|-------------|
| Accuracy Train | Proportion correct on training set |
| Accuracy Test | Proportion correct on held-out test set |
| F1 Macro | Unweighted average F1 across all 5 classes |
| F1 Weighted | Class-size-weighted average F1 |
| Classification Report | Per-class Precision, Recall, F1, Support |

---

## 📈 Evaluation Metrics

### Regression

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

### Classification

$$\text{F1} = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 🖼️ Visualizations

| Plot | Section | Purpose |
|------|---------|---------|
| Class distribution bar chart | EDA | Identify class imbalance |
| WQI histogram | EDA | Understand target distribution |
| Feature correlation heatmap | EDA | Identify multicollinearity |
| Feature–WQI bar chart | EDA | Rank predictive features |
| WQI box plots by class | EDA | Validate class separability |
| Regression loss curve | Model 1 | Track training convergence |
| Actual vs. Predicted scatter | Model 1 | Visualize prediction accuracy |
| Residuals vs. Predicted | Model 1 | Check for systematic error |
| Residuals histogram | Model 1 | Verify normality of errors |
| Classification loss + val acc | Model 2 | Track training convergence |
| Confusion matrix heatmap | Model 2 | Per-class error analysis |
| Per-class P/R/F1 bar chart | Model 2 | Identify weak classes |

---

## 🚀 Quick Start

```powershell
cd "L4/Introduction to Deep Learning using PyTorch"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook water_quality_prediction.ipynb
```

Run all cells top-to-bottom (Kernel → Restart & Run All). No external API keys or additional data downloads required — `water_quality.csv` is bundled.

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install via:

```powershell
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

> **Python version:** 3.10+ recommended (3.9+ minimum).

---

## 📁 File Structure

```
Introduction to Deep Learning using PyTorch/
├── water_quality_prediction.ipynb   # Main notebook (7 sections, 30 cells)
├── water_quality.csv                # CPCB dataset (19,029 records)
└── README.md                        # This file
```
