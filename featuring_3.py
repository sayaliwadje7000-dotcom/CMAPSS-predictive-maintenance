import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Performs causal feature engineering on CMAPSS data grouped by unit_id.
    Automatically detects active sensor columns remaining after Step 1.
    """
    print("\n--- TASK 3: Feature Engineering ---")
    
    df_feat = df.copy()
    
    # Since constant sensors were already dropped in Step 1, 
    # we dynamically grab whatever sensor columns are present in the incoming dataframe.
    sensor_cols = [col for col in df_feat.columns if col.startswith('sensor_')]
    print(f"Active sensors found for rolling features: {len(sensor_cols)} sensors")
    
    # Compute rolling statistics (windows: 5, 10 cycles)
    windows = [5, 10]
    
    for w in windows:
        # Group by unit_id to prevent bleeding across different engines
        grouped = df_feat.groupby('unit_id')[sensor_cols]
        
        rolling_mean = grouped.rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        rolling_std = grouped.rolling(window=w, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
        rolling_min = grouped.rolling(window=w, min_periods=1).min().reset_index(level=0, drop=True)
        rolling_max = grouped.rolling(window=w, min_periods=1).max().reset_index(level=0, drop=True)
        
        # Rename columns
        rolling_mean.columns = [f'{col}_roll_mean_{w}' for col in sensor_cols]
        rolling_std.columns = [f'{col}_roll_std_{w}' for col in sensor_cols]
        rolling_min.columns = [f'{col}_roll_min_{w}' for col in sensor_cols]
        rolling_max.columns = [f'{col}_roll_max_{w}' for col in sensor_cols]
        
        df_feat = pd.concat([df_feat, rolling_mean, rolling_std, rolling_min, rolling_max], axis=1)
        
    # Cross-Sensor Ratios/Differences (if sensors exist)
    if 'sensor_2' in df_feat.columns and 'sensor_3' in df_feat.columns:
        print("Adding cross-sensor features...")
        df_feat['sensor_2_3_ratio'] = df_feat['sensor_2'] / (df_feat['sensor_3'] + 1e-5)
        df_feat['sensor_2_3_diff'] = df_feat['sensor_2'] - df_feat['sensor_3']

    print(f"Feature engineering complete. New shape: {df_feat.shape}")
    return df_feat