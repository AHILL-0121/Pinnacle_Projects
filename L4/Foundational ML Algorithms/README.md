# 🚕 NYC Taxi Trip Duration — Predictive Modeling (Project 10)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)]()

*Regression model to predict NYC taxi trip duration from geospatial and temporal features — covering the full ML workflow from EDA through Gradient Boosting evaluation.*

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [Workflow](#-workflow)
- [Models Evaluated](#-models-evaluated)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Quick Start](#-quick-start)
- [Dependencies](#-dependencies)

---

## 🎯 Overview

This notebook predicts **how long a NYC taxi trip will take** (in seconds) using only the information available at the moment the trip begins: pickup/dropoff coordinates and pickup timestamp. Accurate trip-duration estimates help:

- Taxi dispatch systems predict driver availability
- Passengers get reliable ETA information
- Fleet operators optimize scheduling

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Rows** | 1,499 trips (after outlier removal) |
| **Raw Features** | Pickup/dropoff coordinates, pickup datetime, passenger count |
| **Target** | `trip_duration` (seconds) — log-transformed for modeling |
| **Source** | NYC Taxi & Limousine Commission (Kaggle competition data) |
| **Outlier Handling** | Top/bottom 1% of trip duration and distance removed |

---

## ⚙️ Feature Engineering

| Feature | Description |
|---------|-------------|
| `distance_km` | Haversine great-circle distance between pickup and dropoff |
| `pickup_hour` | Hour of day extracted from pickup timestamp (0–23) |
| `pickup_day` | Day of week (0=Monday … 6=Sunday) |
| `pickup_month` | Month of year |
| `is_weekend` | Binary flag — Saturday or Sunday |
| `rush_hour` | Binary flag — weekday 7–9 AM or 4–7 PM |

The **target variable** `trip_duration` is right-skewed: log-transformation is applied before model training and reversed for metric reporting.

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
   │  Shape, dtypes, descriptive statistics
   │
   ▼
3. Exploratory Data Analysis
   │  Trip duration distribution, geospatial scatter, temporal patterns
   │
   ▼
4. Outlier Removal
   │  Clip top/bottom 1% of duration and distance
   │
   ▼
5. Feature Engineering
   │  Haversine distance, temporal decomposition, binary flags
   │
   ▼
6. Preprocessing
   │  Log-transform target, StandardScaler on features, train/test split
   │
   ▼
7. Model Training
   │  Linear Regression | Random Forest | Gradient Boosting
   │
   ▼
8. Model Evaluation
   │  RMSE, MAE, R² comparison
   │
   ▼
9. Residual Analysis & Prediction Visualization
   │
   ▼
10. Conclusion
```

---

## 🤖 Models Evaluated

| # | Model | Type |
|---|-------|------|
| 1 | Linear Regression | Parametric — baseline |
| 2 | Random Forest Regressor | Ensemble — Bagging |
| 3 | **Gradient Boosting Regressor** | **Ensemble — Boosting** ✅ |

Evaluation metrics: **RMSE**, **MAE**, **R²** on hold-out test set.

---

## 📈 Results

**Best Model: Gradient Boosting Regressor**

| Metric | Gradient Boosting | Random Forest | Linear Regression |
|--------|:-----------------:|:-------------:|:-----------------:|
| **R²** | ✅ Highest | High | Baseline |
| **RMSE** | ✅ Lowest | Low | Highest |
| **MAE** | ✅ Lowest | Low | Highest |

### Conclusion (from notebook)

| Step | Summary |
|------|---------|
| **Data** | 1,499 NYC taxi trips with pickup/dropoff coordinates, timestamps, and trip duration |
| **EDA** | Highly right-skewed trip duration → log-transformed target |
| **Features Engineered** | Haversine distance, hour, day-of-week, month, weekend flag, rush-hour flag |
| **Outliers** | Removed top/bottom 1% of trip duration and distance |
| **Best Model** | Gradient Boosting Regressor |
| **Key Predictors** | `distance_km`, `pickup_hour`, `pickup_longitude/latitude`, `rush_hour` |

> The Gradient Boosting model achieved the highest R² and lowest RMSE, demonstrating that trip duration is well-predicted from distance and time-of-day features.

---

## 💡 Key Insights

1. **Distance is king** — `distance_km` (Haversine) is by far the strongest predictor
2. **Time of day matters** — `pickup_hour` and `rush_hour` flag significantly impact duration
3. **Coordinates add value** — raw lat/lon coordinates capture spatial clustering (e.g., airport trips, crosstown routes)
4. **Log transformation** is essential — without it, models underperform on long-tail trips
5. **Gradient Boosting > Random Forest > Linear Regression** for this geospatial regression task

---

## 🚀 Quick Start

### Prerequisites

```powershell
cd "L4/Foundational ML Algorithms"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook

```powershell
jupyter notebook "nyc_taxi_trip_duration(1).ipynb"
```

Or open it directly in VS Code with the Jupyter extension.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical ops, Haversine math |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | All ML models, preprocessing, metrics |
| `math` | `radians`, `cos`, `sin`, `asin`, `sqrt` for Haversine formula |
| `jupyter` | Notebook execution environment |

---

## 📄 License & Author

**License:** MIT  
**Author:** AHILL S  
**Part of:** [Pinnacle Projects Portfolio](../../README.md)
