# Istanbul Airbnb Price Prediction 🏠🇹🇷

> **🏆 Achieved 5th Place** in the [Kaggle Class Competition](https://www.kaggle.com/competitions/yzv-311-term-project-fall-25-26) for predicting Airbnb prices in Istanbul using Machine Learning.
>
> _This project was developed for the **YZV 311E — Data Mining** course at **ITU (Istanbul Technical University)** as part of the official class Kaggle competition in 2025._

This project focuses on predicting listing prices for Airbnb properties across various neighborhoods in Istanbul. By thoroughly cleaning the raw Kaggle dataset, engineering complex geographical and text-based features, and using an optimized XGBoost model via Optuna, the project scored among the top submissions.

---

## 🚀 Features & Technical Stack

- **Data Processing & Feature Engineering:** Extracted Haversine distances to major landmarks (Taksim, Sultanahmet, Kadikoy, etc.), grouped availability patterns, merged Kaggle calendar and review files, generated text lengths and keyword counts (e.g. Bosphorus, Sea View), clustered geo-locations using KMeans. Target encoding with smoothing was applied for categorical variables.
- **Modeling:** XGBoost Regressor
- **Hyperparameter Tuning:** Optuna (10-fold CV)
- **Outlier Handling:** Log-transformation (np.log1p) of the target variable along with IQR and Z-score based extreme value cropping.
- **Technologies Used:** Python, Pandas, Numpy, Scikit-Learn, XGBoost, Optuna

---

## 📊 Results

| Metric | Score |
|---|---|
| **Kaggle Ranking** | 5th Place |
| **Best CV RMSE** (log-space) | ~0.38 |
| **Cross-Validation** | 10-Fold |

---

## 📁 Directory Structure

```text
.
├── data/                  # Raw Kaggle CSVs, GeoJSON & generated datasets (not tracked in git)
│   ├── calendar.csv       # Listing availability calendar (download from Kaggle)
│   ├── reviews.csv        # Guest reviews (download from Kaggle)
│   ├── train.csv          # Training set (download from Kaggle)
│   ├── test.csv           # Test set (download from Kaggle)
│   ├── neighbourhoods.geojson
│   └── sample_submission.csv
├── notebooks/             # Jupyter notebooks for EDA and data cleaning
│   ├── 01_load_target_clean.ipynb       # Initial data loading, price cleaning & EDA
│   ├── 02_load_target_clean_filled.ipynb # Missing value imputation
│   └── 03_baseline_model.ipynb          # Random Forest baseline model
├── src/                   # Python source code
│   ├── features/
│   │   └── feature_engineering.py  # Complex feature engineering pipeline
│   └── models/
│       └── train_xgboost.py        # XGBoost training with Optuna tuning
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🔄 Pipeline Overview

The project follows a two-stage pipeline:

**Stage 1 — Data Preparation (Notebooks)**
1. `01_load_target_clean.ipynb` — Load raw data, clean the price column (remove `$` and `,` symbols), drop rows with missing/zero prices, and perform initial EDA.
2. `02_load_target_clean_filled.ipynb` — Impute missing values using median (numeric) and mode/Unknown (categorical), then save the cleaned dataset.

**Stage 2 — Feature Engineering & Modeling (Scripts)**
3. `src/features/feature_engineering.py` — Merge calendar & review data, compute Haversine distances, KMeans clusters, amenity parsing, text mining features, neighbourhood statistics, and interaction features.
4. `src/models/train_xgboost.py` — Outlier removal (IQR + Z-score), target encoding with K-Fold smoothing, Optuna hyperparameter search, 10-fold CV ensemble training, and submission file generation.

> **Note:** The notebooks use relative paths based on the original Kaggle/Colab environment (`../datasets/`). The production pipeline scripts in `src/` use the `data/` directory as defined in the project structure.

---

## 🛠 Usage

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/semihcellk/Istanbul-Airbnb-Price-Prediction.git
cd Istanbul-Airbnb-Price-Prediction

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

1. Download the Kaggle datasets from the [competition page](https://www.kaggle.com/competitions/yzv-311-term-project-fall-25-26) and place them under `data/` (e.g., `calendar.csv`, `reviews.csv`, `train.csv`, `test.csv`).

2. Run the data cleaning notebooks in order (`notebooks/01_*` → `02_*`) to produce `data/interim/filled_cleaned_train.csv`.

3. Generate the engineered features:
   ```bash
   python src/features/feature_engineering.py
   ```

4. Train the model and generate the submission file:
   ```bash
   python src/models/train_xgboost.py
   ```

The final submission will be saved to `data/submission/submission.csv`.

---

## 🤖 AI Usage Declaration

- **Syntax Support:** Used AI to generate the correct Regex and `ast.literal_eval` syntax for parsing the `amenities` column.
- **Implementation:** The `haversine_distance` mathematical formula implementation was assisted by AI tools.
- **Refactoring:** Modular structure of `load_calendar_features` was refined with AI suggestions.
- **Library Usage:** The boilerplate structure for Optuna's `objective_xgb` function was generated using AI assistance.
- **Debugging:** The `sanitize_column_names` function using Regex to fix XGBoost feature name errors was created with AI help.
- **Algorithm Logic:** The smoothing formula logic within `target_encode` was refined through idea exploration with AI.

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
