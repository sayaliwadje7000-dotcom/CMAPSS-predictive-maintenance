import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def split_train_test(df_featured, test_size=0.2, random_state=42):
    """
    Splits the dataset by engine unit_id to prevent data leakage.
    Ensures an entire engine goes completely into either train or test, never split.
    """
    print("\n--- TASK 4: Train/Test Split (Grouped by Engine) ---")
    
    # We want to split based on unique unit_id, not individual rows
    # GroupShuffleSplit splits groups rather than rows
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # Extract groups (unit_id)
    groups = df_featured['unit_id']
    
    # Perform the split
    train_idx, test_idx = next(splitter.split(df_featured, groups=groups))
    
    train_df = df_featured.iloc[train_idx].reset_index(drop=True)
    test_df = df_featured.iloc[test_idx].reset_index(drop=True)
    
    print(f"Total rows in Train: {train_df.shape[0]} ({train_df['unit_id'].nunique()} unique engines)")
    print(f"Total rows in Test: {test_df.shape[0]} ({test_df['unit_id'].nunique()} unique engines)")
    
    # Verify no engine overlaps between train and test
    train_units = set(train_df['unit_id'].unique())
    test_units = set(test_df['unit_id'].unique())
    overlap = train_units.intersection(test_units)
    print(f"Engine overlap between Train and Test: {overlap if overlap else 'None (Clean split!)'}")
    
    return train_df, test_df

if __name__ == "__main__":
    # Test script locally if run directly
    pass