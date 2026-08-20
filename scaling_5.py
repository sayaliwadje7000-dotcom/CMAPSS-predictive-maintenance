import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_data_for_modeling(train_df, test_df):
    """
    Prepares features and targets, and applies proper scaling for baseline models 
    while preventing data leakage (fitting scaler strictly on train data).
    """
    print("\n--- TASK 5: Scaling Decisions & Data Preparation ---")
    
    # 1. Define feature columns (drop metadata and label columns from features)
    drop_cols = ['unit_id', 'cycle', 'RUL', 'binary_label', 'root_cause']
    feature_cols = [col for col in train_df.columns if col not in drop_cols]
    
    print(f"Total features selected for modeling: {len(feature_cols)}")
    
    # Separate features (X) and binary labels (y) for Train and Test
    X_train = train_df[feature_cols]
    y_train = train_df['binary_label']
    
    X_test = test_df[feature_cols]
    y_test = test_df['binary_label']
    
    # 2. Scaling Decision for Tree Models (Random Forest / XGBoost / LightGBM)
    print("\n[Tree Models Note]: No scaling applied. Random Forest and XGBoost are scale-invariant.")
    X_train_trees = X_train.copy()
    X_test_trees = X_test.copy()
    
    # 3. Scaling Decision for Linear Models / SVM Baseline (Logistic Regression)
    print("[Linear Baseline Note]: Applying StandardScaler. Fitting ONLY on training fold to prevent leakage.")
    scaler = StandardScaler()
    
    # Fit strictly on X_train, transform both X_train and X_test separately
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames for clean handling downstream
    X_train_linear = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_linear = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    print("Scaling and data preparation complete!")
    
    return {
        'X_train_trees': X_train_trees,
        'X_test_trees': X_test_trees,
        'X_train_linear': X_train_linear,
        'X_test_linear': X_test_linear,
        'y_train': y_train,
        'y_test': y_test,
        'feature_cols': feature_cols,
        'scaler': scaler  # fitted on train only — reuse this for any new row, never re-fit
    }

if __name__ == "__main__":
    pass