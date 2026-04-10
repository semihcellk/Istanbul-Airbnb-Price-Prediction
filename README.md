# Istanbul Airbnb Price Prediction 🏠🇹🇷

> **🏆 Achieved 5th Place** in the Kaggle Class Competition for predicting Airbnb prices in Istanbul using Machine Learning.
>
> _This project was developed for the **YZV 311E - Data Mining** course at **ITU (Istanbul Technical University)** as part of the official class Kaggle competition in 2025._

This project focuses on predicting listing prices for Airbnb properties across various neighborhoods in Istanbul. By thoroughly cleaning out the raw Kaggle dataset, engineering complex geographical and text-based features, and using an optimized XGBoost model via Optuna, the project scored among the top submissions.

## 🚀 Features & Technical Stack

- **Data Processing & Feature Engineering:** Extracted Haversine distances to major landmarks (Taksim, Sultanahmet, Kadikoy, etc.), grouped availability patterns, merged Kaggle calendar and review files, generated text lengths and keyword counts (e.g. Bosphorus, Sea View), clustered geo-locations using KMeans. Target encoding with smoothing was applied for categorical variables.
- **Modeling:** XGBoost Regressor
- **Hyperparameter Tuning:** Optuna (10-fold CV)
- **Outlier Handling:** Log-transformation (np.log1p) of the target variable along with IQR and Z-score based extreme value cropping.
- **Technologies Used:** Python, Pandas, Numpy, Scikit-Learn, XGBoost, Optuna

## 📁 Directory Structure

```text
.
├── data/                  # Contains raw Kaggle CSVs, geojsons & generated processed datasets (Not tracked in git)
├── notebooks/             # Baseline Jupyter notebooks for EDA and straightforward target cleanup
├── src/                   # Python source code algorithms
│   ├── features/          # Complex feature engineering logic
│   └── models/            # XGBoost training and Optuna tuning
├── README.md
└── requirements.txt
```

## 🛠 Usage

1. Clone the repository
2. Install the necessary packages via `pip install -r requirements.txt`
3. Download the Kaggle datasets and place them under `data/` (e.g., `calendar.csv`, `reviews.csv`, `train.csv`, `test.csv`).
4. Generate the engineered features by executing:
   ```bash
   python src/features/feature_engineering.py
   ```
5. Train the model and generate the submission file:
   ```bash
   python src/models/train_xgboost.py
   ```
