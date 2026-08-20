import pandas as pd
import numpy as np

def engineer_labels(df, n_threshold=20):
    """Computes RUL, binary failure labels, and heuristic root-cause labels."""
    
    # 1. Compute RUL (Remaining Useful Life)
    max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.rename(columns={'cycle': 'max_cycle'}, inplace=True)
    
    df = df.merge(max_cycles, on='unit_id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    
    # 2. Compute Binary Failure Label
    df['binary_label'] = (df['RUL'] <= n_threshold).astype(int)
    
    # 3. Compute Heuristic Root-Cause Multi-Class Label
    # Define physical sensor groups based on CMAPSS documentation
    thermal_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_11']
    pressure_sensors = ['sensor_7', 'sensor_8', 'sensor_12', 'sensor_13']
    mechanical_sensors = ['sensor_9', 'sensor_14', 'sensor_15', 'sensor_17']
    
    # Compute healthy baseline (mean and std) from the first 15 cycles of each unit
    healthy_baseline = df[df['cycle'] <= 15]
    mean_baseline = healthy_baseline.groupby('unit_id')[thermal_sensors + pressure_sensors + mechanical_sensors].mean()
    std_baseline = healthy_baseline.groupby('unit_id')[thermal_sensors + pressure_sensors + mechanical_sensors].std().replace(0, 1) # avoid div by zero
    
    # Function to calculate group deviations for the last 15 cycles or globally
    def assign_root_cause(row):
        uid = row['unit_id']
        if uid not in mean_baseline.index:
            return 'normal'
            
        # Get baseline values for this unit
        b_mean = mean_baseline.loc[uid]
        b_std = std_baseline.loc[uid]
        
        # Current row sensor values
        sensor_cols = thermal_sensors + pressure_sensors + mechanical_sensors
        current_vals = row[sensor_cols]
        
        # Calculate absolute standardized z-score deviation
        z_scores = np.abs((current_vals - b_mean) / b_std)
        
        # Group deviations
        thermal_dev = z_scores[thermal_sensors].mean()
        pressure_dev = z_scores[pressure_sensors].mean()
        mechanical_dev = z_scores[mechanical_sensors].mean()
        
        # Find subsystem with highest deviation
        deviations = {
            'thermal': thermal_dev,
            'pressure': pressure_dev,
            'mechanical': mechanical_dev
        }
        
        # If binary label is 0 (healthy), label as 'normal'
        if row['binary_label'] == 0:
            return 'normal'
            
        return max(deviations, key=deviations.get)

    print("Computing heuristic root-cause labels...")
    df['root_cause'] = df.apply(assign_root_cause, axis=1)

    print(f"Labels engineered successfully. Binary label counts:\n{df['binary_label'].value_counts()}")
    print(f"Root-cause label counts:\n{df['root_cause'].value_counts()}")
    
    return df

if __name__ == "__main__":
    pass