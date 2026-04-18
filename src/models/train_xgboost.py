import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
import re
import os

"""
AI USAGE DECLARATION:
- Library Usage: The boilerplate structure for Optuna's 'objective_xgb' function was generated using AI assistance.
- Debugging: The 'sanitize_column_names' function using Regex to fix XGBoost feature name errors was created with AI help.
- Algorithm Logic: The smoothing formula logic within 'target_encode' was refined through idea exploration with AI.
"""

DEMO_MODE = False

warnings.filterwarnings("ignore")

# PATHS
TRAIN_PATH = 'data/processed/train/engineered_train.csv'
TEST_PATH = 'data/processed/test/engineered_test.csv'
SUBMISSION_PATH = 'data/submission/submission.csv'

def clean_currency_percent(x):
    """Clean currency and percentage strings into numeric float values.

    Strips '$', '%', ',' characters and converts to float.
    Returns np.nan for unparseable or placeholder values like 'N/A'.
    """
    if isinstance(x, str):
        x = x.replace('%', '').replace('$', '').replace(',', '').strip()
        if x in ['N/A', '', 'Unknown', 'nan']: 
            return np.nan
        try: 
            return float(x)
        except (ValueError, TypeError): 
            return np.nan
    return x

def sanitize_column_names(df):
    """Remove special characters from column names for XGBoost compatibility.

    XGBoost requires feature names to contain only alphanumeric characters
    and underscores. This function strips all other characters.
    """
    new_cols = [re.sub(r'[^A-Za-z0-9_]', '', str(c)) for c in df.columns]
    df.columns = new_cols
    return df

def target_encode(train_df, test_df, col, target_col='price', n_splits=5, smoothing=10):
    """Apply target encoding with smoothing using K-Fold cross-validation.

    For training data, uses K-Fold CV to avoid target leakage.
    For test data, computes smoothed means using the full training set.
    Unseen categories are filled with the global target mean.

    Args:
        train_df: Training DataFrame (modified in-place).
        test_df: Test DataFrame (modified in-place).
        col: Categorical column name to encode.
        target_col: Target variable column name.
        n_splits: Number of CV folds.
        smoothing: Smoothing strength (higher = more regularization).

    Returns:
        Tuple of (train_df, test_df) with the new encoded column added.
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    new_col = f'{col}_target_enc'
    train_df[new_col] = np.nan
    
    global_mean = train_df[target_col].mean()
    
    for tr_ind, val_ind in kf.split(train_df):
        X_tr, X_val = train_df.iloc[tr_ind], train_df.iloc[val_ind]
        
        # Calculate mean and count per category
        agg_dict = X_tr.groupby(col).agg({target_col: ['mean', 'count']})
        agg_dict.columns = ['mean', 'count']
        
        # Apply smoothing
        smoothed_mean = (
            (agg_dict['mean'] * agg_dict['count'] + global_mean * smoothing) / 
            (agg_dict['count'] + smoothing)
        )
        
        train_df.loc[val_ind, new_col] = X_val[col].map(smoothed_mean)
    
    train_df[new_col].fillna(global_mean, inplace=True)
    
    # For test set
    agg_dict_full = train_df.groupby(col).agg({target_col: ['mean', 'count']})
    agg_dict_full.columns = ['mean', 'count']
    smoothed_mean_full = (
        (agg_dict_full['mean'] * agg_dict_full['count'] + global_mean * smoothing) / 
        (agg_dict_full['count'] + smoothing)
    )
    
    test_df[new_col] = test_df[col].map(smoothed_mean_full)
    test_df[new_col].fillna(global_mean, inplace=True)
    
    return train_df, test_df

def load_and_preprocess():
    """Load engineered data, remove outliers, encode categoricals, and prepare X/y.

    Pipeline: load CSVs → IQR + Z-score outlier removal → drop junk columns →
    clean currency/percent strings → target encode categoricals →
    label encode remaining objects → sanitize column names.

    Returns:
        Tuple of (X_train, y_train, X_test) ready for model training.
    """
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    
    print(f"Initial shapes - Train: {df_train.shape}, Test: {df_test.shape}")
    
    # Outlier Removal
    initial_count = len(df_train)
    
    # Use IQR method
    Q1 = df_train['price'].quantile(0.25)
    Q3 = df_train['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 2.5 * IQR
    
    df_train = df_train[(df_train['price'] >= lower_bound) & (df_train['price'] <= upper_bound)]
    
    # Also remove extreme Z-scores in log space
    z_scores = np.abs(stats.zscore(np.log1p(df_train['price'])))
    df_train = df_train[z_scores < 3.5]
    
    df_train.reset_index(drop=True, inplace=True)
    print(f"Removed {initial_count - len(df_train)} outliers ({(initial_count - len(df_train))/initial_count*100:.2f}%)")
    print(f"Remaining samples: {len(df_train)}")
    
    # DROP UNNECESSARY COLUMNS
    drop_cols = [
        'id', 'listing_url', 'scrape_id', 'last_scraped', 'source', 
        'name', 'description', 'neighborhood_overview', 'picture_url', 
        'host_id', 'host_url', 'host_name', 'host_location', 'host_about', 
        'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 
        'host_verifications', 'calendar_updated', 'license', 'amenities', 
        'bathrooms_text', 'host_since'
    ]
    
    # Drop Unnamed columns
    junk_cols = [c for c in df_train.columns if 'Unnamed' in str(c)]
    drop_cols.extend(junk_cols)
    
    cols_to_drop = [c for c in drop_cols if c in df_train.columns]
    df_train.drop(columns=cols_to_drop, inplace=True)
    df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns], inplace=True)
    
    print(f"Dropped {len(cols_to_drop)} columns")
    
    # CLEAN CURRENCY/PERCENTAGE COLUMNS
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in df_train.columns:
            df_train[col] = df_train[col].apply(clean_currency_percent)
            df_test[col] = df_test[col].apply(clean_currency_percent)
            
            # Fill with median
            median_val = df_train[col].median()
            df_train[col].fillna(median_val, inplace=True)
            df_test[col].fillna(median_val, inplace=True)
    
    # TARGET ENCODING
    target_cols = [
        'neighbourhood_cleansed', 'room_type', 'property_type', 
        'loc_room_interaction', 'prop_neigh_interaction', 'room_prop_interaction',
        'loc_cluster'
    ]
    
    for col in target_cols:
        if col in df_train.columns:
            df_train, df_test = target_encode(df_train, df_test, col, target_col='price', smoothing=15)
            print(f"Encoded: {col}")
    
    # PREPARE X, y
    X = df_train.drop(columns=['price']).copy()
    y = np.log1p(df_train['price'])  # Log transform target
    X_test = df_test.copy()
    
    # Align test columns with train (add missing columns as NaN, drop extras)
    missing_cols = [c for c in X.columns if c not in X_test.columns]
    for col in missing_cols:
        X_test[col] = np.nan
    X_test = X_test[X.columns]
    
    # LABEL ENCODING
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        all_vals = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(all_vals)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
    
    print(f"Label encoded {len(cat_cols)} categorical columns")
    
    # SANITIZE COLUMN NAMES
    X = sanitize_column_names(X)
    X_test = sanitize_column_names(X_test)
    
    print(f"Final shapes - X: {X.shape}, y: {y.shape}, X_test: {X_test.shape}")
    print(f"Target stats - Mean: {y.mean():.4f}, Std: {y.std():.4f}")
    
    return X, y, X_test

def objective_xgb(trial, X, y, n_splits=10):
    """Optuna objective function for XGBoost hyperparameter optimization.

    Evaluates hyperparameter combinations using K-Fold CV and returns
    the mean RMSE across folds (minimization target).
    """
    param = {
        'tree_method': 'hist',
        'random_state': 42,
        'n_estimators': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'early_stopping_rounds': 150,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(rmse)
    
    return np.mean(scores)

def train_xgboost_model(X, y, X_test, n_trials=30):
    """Train XGBoost with Optuna tuning and ensemble predictions via K-Fold CV.

    Steps: 1) Optuna search for best hyperparameters, 2) retrain with
    more estimators across K folds, 3) average fold predictions for test set.

    Returns:
        Tuple of (final_predictions in original scale, list of CV RMSE scores).
    """
    if DEMO_MODE:
        print("DEMO MODE ACTIVE")
        n_trials = 1       
        cv_splits = 2       
        n_estimators = 100  
    else:
        cv_splits = 10   
        n_estimators = 20000 
          
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    
    # Optimize hyperparameters
    print(f"Running {n_trials} trials with {cv_splits}-fold cross-validation...")
    
    study = optuna.create_study(direction='minimize', study_name='XGBoost_v5')
    study.optimize(lambda trial: objective_xgb(trial, X, y, n_splits=cv_splits), n_trials=n_trials)
    
    best_params = study.best_params
    print(f"Best CV RMSE: {study.best_value:.5f}")
    print(f"\nBest hyperparameters:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    # Update with final training parameters
    best_params.update({
        'n_estimators': n_estimators,
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50 if DEMO_MODE else 400,
        'verbosity': 0
    })
    
    # TRAIN FINAL MODEL WITH 10-FOLD CV
    print(f"TRAINING FINAL MODEL ({cv_splits}-Fold CV)")

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    test_preds = np.zeros(len(X_test))
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nFold {fold+1}/{cv_splits}")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Train model
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Validation prediction
        val_preds = model.predict(X_val)
        fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        cv_scores.append(fold_rmse)
        
        print(f"Validation RMSE: {fold_rmse:.5f}")
        print(f"Best iteration: {model.best_iteration}")
        
        # Test prediction (average across all folds)
        test_preds += model.predict(X_test) / cv_splits
    
    print("CROSS-VALIDATION RESULTS:")
    print(f"Mean CV RMSE: {np.mean(cv_scores):.5f} (+/- {np.std(cv_scores):.5f})")
    print(f"Min CV RMSE:  {np.min(cv_scores):.5f}")
    print(f"Max CV RMSE:  {np.max(cv_scores):.5f}")
    
    # Convert back from log space
    final_predictions = np.expm1(test_preds)
    
    return final_predictions, cv_scores

# MAIN EXECUTION

if __name__ == "__main__":
    
    # Load and preprocess data
    X, y, X_test = load_and_preprocess()
    
    # Train XGBoost model
    final_predictions, cv_scores = train_xgboost_model(X, y, X_test, n_trials=30)
    
    # Create submission file
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    test_ids = pd.read_csv('data/test.csv')['id']
    
    submission = pd.DataFrame({
        'ID': test_ids,
        'TARGET': final_predictions
    })
    
    submission.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"\nPrediction statistics:")
    print(f"   Mean:     {final_predictions.mean():.2f}")
    print(f"   Median:   {np.median(final_predictions):.2f}")
    print(f"   Std:      {final_predictions.std():.2f}")
    print(f"   Min:      {final_predictions.min():.2f}")
    print(f"   Max:      {final_predictions.max():.2f}")
    print(f"   Q25:      {np.percentile(final_predictions, 25):.2f}")
    print(f"   Q75:      {np.percentile(final_predictions, 75):.2f}")
    print(f"Submission saved: {SUBMISSION_PATH}")