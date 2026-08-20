import pandas as pd
import numpy as np

def analyze_feature_importance(model, feature_cols, top_n=10):
    """
    Extracts and displays the top N most important features from a tree-based model.
    """
    print(f"\n--- TASK 8: Model Interpretation & Feature Importance ---")
    
    # Check if the model has feature_importances_ attribute (Tree models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create a dataframe pairing features with their importance scores
        feat_imp_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features Driving Engine Failures:")
        print(feat_imp_df.head(top_n).to_string(index=False))
        
        return feat_imp_df
    else:
        print("Model does not support direct feature importances.")
        return None

if __name__ == "__main__":
    pass