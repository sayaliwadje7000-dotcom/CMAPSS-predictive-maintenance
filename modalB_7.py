import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix

def train_and_evaluate_model_b(train_df, test_df):
    """
    Trains a multi-class model (Model B) to predict engine root-cause/failure modes.
    Cascaded Design: Filtered strictly to imminent failure rows (binary_label == 1).
    """
    print("\n--- TASK 7: Model B - Root-Cause / Fault Mode Classification ---")
    
    # 1. Filter to rows where failure is imminent (binary_label == 1)
    train_failures = train_df[train_df['binary_label'] == 1].copy()
    test_failures = test_df[test_df['binary_label'] == 1].copy()
    
    # 2. Define features and target (do NOT drop root_cause here!)
    drop_cols = ['unit_id', 'cycle', 'RUL', 'binary_label', 'root_cause']
    feature_cols = [col for col in train_failures.columns if col not in drop_cols]
    
    X_train = train_failures[feature_cols]
    y_train = train_failures['root_cause']
    
    X_test = test_failures[feature_cols]
    y_test = test_failures['root_cause']
    
    print(f"Root-cause classes present in training data: {y_train.unique()}")
    print(f"Training feature shape (Failure rows only): {X_train.shape}")
    
    # 3. Train Multi-class Random Forest Classifier
    print("\nTraining Random Forest for Root-Cause Classification...")
    model_b = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    
    model_b.fit(X_train, y_train)
    
    # 4. Predict and Evaluate
    y_pred = model_b.predict(X_test)
    
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"Model B Macro F1-Score: {macro_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model_b

if __name__ == "__main__":
    pass