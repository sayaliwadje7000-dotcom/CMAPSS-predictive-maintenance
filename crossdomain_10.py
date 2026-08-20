import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, auc

def run_ai4i_validation(file_path='ai4i2020.csv'):
    """
    Validates that the methodology generalizes to the AI4I 2020 dataset.
    Kept strictly separate from CMAPSS results.
    """
    print("\n--- TASK 10: Cross-Domain Validation (AI4I 2020 Dataset) ---")
    
    # 1. Load dataset (Ensure 'ai4i2020.csv' is in your data/raw directory or working directory)
    try:
        df = pd.read_csv('ai4i2020-selected-columns.csv')
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please place the AI4I 2020 dataset in the directory.")
        return None
        
    print(f"AI4I Dataset loaded successfully. Shape: {df.shape}")
    
    # 2. Drop non-feature columns (like identifiers or product types if necessary)
    # AI4I typically includes: UDI, Product ID, Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear, Machine failure
    drop_cols = ['UDI', 'Product ID', 'Machine failure']
    if 'TWF' in df.columns: # Specific failure modes if present in full dataset
        drop_cols.extend(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    # One-hot encode categorical features like 'Type' (L, M, H)
    df_encoded = pd.get_dummies(df, columns=['Type'], drop_first=True)
    
    feature_cols = [col for col in df_encoded.columns if col not in ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
    
    X = df_encoded[feature_cols]
    y = df_encoded['Machine failure']
    
    print(f"Features selected for AI4I validation: {list(X.columns)}")
    print(f"Class distribution: Safe={sum(y==0)}, Failure={sum(y==1)}")
    
    # 3. Train/Test Split (Standard stratified split since AI4I is snapshot data, not sequential engine cycles)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Train Random Forest Classifier with class balancing
    print("\nTraining Random Forest on AI4I domain...")
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    
    print(f"\nAI4I Validation Results:")
    print(f"  - Macro F1-Score: {macro_f1:.4f}")
    print(f"  - PR-AUC: {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\n[Constraint Note]: These metrics are kept completely separate from CMAPSS results.")
    return model

if __name__ == "__main__":
    run_ai4i_validation()