import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_and_save_final_model(best_model, best_model_name, data_packets, y_test):
    """
    Evaluates the best model on the test set and saves the model artifact to disk.
    """
    print(f"\n--- TASK 9: Final Model Evaluation & Artifact Saving ---")
    
    # Retrieve correct test data packet based on model type
    if 'Trees' in best_model_name:
        X_test = data_packets['X_test_trees']
    else:
        X_test = data_packets['X_test_linear']
        
    # Final predictions on test data
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    print(f"\nFinal Test Set Performance for Best Model ({best_model_name}):")
    print("-" * 50)
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate final ROC-AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nFinal Test ROC-AUC Score: {roc_auc:.4f}")
    
    # Save the trained model artifact
    model_filename = "best_failure_prediction_model.pkl"
    joblib.dump(best_model, model_filename)
    print(f"\nSuccess! Best model saved to disk as: '{model_filename}'")

if __name__ == "__main__":
    pass