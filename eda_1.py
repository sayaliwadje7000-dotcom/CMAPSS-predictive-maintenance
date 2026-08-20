import pandas as pd
import numpy as np
import os 

def load_and_clean_data(file_path='train_FD003.txt'):


# 1. Define column names based on CMAPSS documentation
    index_names = ['unit_id', 'cycle']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f'sensor_{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # 2. Load training data properly inside read_csv
    train_df = pd.read_csv(
        file_path, 
        sep='\s+', 
        header=None, 
        names=col_names, 
        usecols=range(26)
  )
    print(f"Dataset shape: {train_df.shape}")

    # 3. Check for missing values
    missing_counts = train_df.isnull().sum().sum()
    print(f"Total missing values in dataset: {missing_counts}")

    # 4. Analyze cycles-to-failure (max cycle per engine unit)
    max_cycles = train_df.groupby('unit_id')['cycle'].max()
    print("\n--- Max Cycles per Unit Distribution ---")
    print(max_cycles.describe())

    # 5. Identify and drop constant or near-zero-variance sensors
    sensor_stds = train_df[sensor_names].std()
    print("\n--- Sensor Standard Deviations ---")
    print(sensor_stds)

    constant_sensors = sensor_stds[sensor_stds < 1e-4].index.tolist()
    print(f"\nConstant or near-zero-variance sensors to drop: {constant_sensors}")

    if constant_sensors:
        train_df = train_df.drop(columns=constant_sensors)
        print(f"Dataset shape after dropping constant sensors: {train_df.shape}")

        return train_df

if __name__ == "__main__":
 load_and_clean_data()