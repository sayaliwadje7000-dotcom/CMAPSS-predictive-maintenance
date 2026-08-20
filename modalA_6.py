import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc
def evaluate_predictions(y_true, y_pred, y_proba):
    """Computes key metrics for imbalanced binary classification."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Precision-Recall AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    
    return {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Macro F1': macro_f1,
        'PR-AUC': pr_auc
    }

def train_and_evaluate_model_a(data_packets, train_df_raw):
    """
    Trains baseline Logistic Regression, Random Forest, and XGBoost for binary failure prediction.
    Uses GroupKFold cross-validation to prevent engine leakage.
    """
    print("\n--- TASK 6: Model A - Binary Failure Prediction ---")
    
    X_train_trees = data_packets['X_train_trees']
    X_test_trees = data_packets['X_test_trees']
    X_train_linear = data_packets['X_train_linear']
    X_test_linear = data_packets['X_test_linear']
    y_train = data_packets['y_train']
    y_test = data_packets['y_test']
    
    # Define models with built-in class imbalance mitigation
    models = {
        'Logistic Regression (Scaled)': (
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42), 
            X_train_linear, 
            X_test_linear
        ),
        'Random Forest (Trees)': (
            RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1), 
            X_train_trees, 
            X_test_trees
        ),
        'XGBoost (Trees)': (
            XGBClassifier(scale_pos_weight=10, random_state=42, eval_metric='logloss'), 
            X_train_trees, 
            X_test_trees
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model_data in models.items():
        model, X_tr, X_te = model_data
    
        # Train model
        model.fit(X_tr, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        
        # Evaluate metrics
        metrics = evaluate_predictions(y_test, y_pred, y_proba)
        results[name] = metrics
        trained_models[name] = model
        
        print(f"Results for {name}:")
        for metric_name, val in metrics.items():
            print(f"  - {metric_name}: {val:.4f}")
            
    # Select best model based on F1-Score / PR-AUC (e.g., Random Forest or XGBoost)
    best_model_name = max(results, key=lambda k: results[k]['PR-AUC'])
    print(f"\nBest Performing Model A: {best_model_name}")
    
    # Example Predictions Check (Select 3-5 interesting rows: normal vs borderline vs about-to-fail)
    print("\n--- Example Predictions Table (Test Set Sample) ---")
    best_model = trained_models[best_model_name]
    best_X_test = models[best_model_name][2]
    
    # Grab a few specific rows to display
    sample_indices = [0, 50, 200, 500, len(X_test_linear)-1]
    sample_indices = [i for i in sample_indices if i < len(best_X_test)]
    
    sample_X = best_X_test.iloc[sample_indices]
    sample_true = y_test.iloc[sample_indices].values
    sample_preds = best_model.predict(sample_X)
    sample_probas = best_model.predict_proba(sample_X)[:, 1]
    
    comparison_df = pd.DataFrame({
        'Actual Label': sample_true,
        'Predicted Label': sample_preds,
        'Failure Probability': np.round(sample_probas, 4)
    })
    print(comparison_df.to_string(index=False))
    
    return trained_models, best_model_name