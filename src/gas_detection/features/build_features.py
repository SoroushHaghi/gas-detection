# src/gas_detection/features/build_features.py

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# --- Add Project Root to sys.path ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()
sys.path.append(str(PROJECT_ROOT))
# ------------------------------------

from src.gas_detection.config import load_config

def create_statistical_features(data, labels, window_size):
    """
    Creates statistical features (mean, std, min, max) from sliding windows.
    Returns:
        tuple: (X_features, y_windows)
        X_features is 2D: [samples, num_features * 4]
        y_windows is 1D: [samples]
    """
    X = []
    y = []
    
    # Convert data to numpy for faster processing
    data_np = data.to_numpy()
    labels_np = labels.to_numpy()
    
    num_features = data_np.shape[1]
    
    for i in range(len(data_np) - window_size):
        window = data_np[i : i + window_size]
        label = labels_np[i + window_size - 1]
        
        # Calculate features: mean, std, min, max for each sensor
        features = np.concatenate([
            np.mean(window, axis=0),
            np.std(window, axis=0),
            np.min(window, axis=0),
            np.max(window, axis=0)
        ])
        
        X.append(features)
        y.append(label)
        
    # Shape of X will be (num_samples, num_features * 4)
    return np.array(X), np.array(y)

def main():
    print("Starting feature engineering (Statistical Features)...")
    config = load_config()
    
    processed_data_path = PROJECT_ROOT / config['paths']['processed_data']
    output_dir = processed_data_path.parent
    X_output_path = output_dir / 'X_train.npy' # Overwriting old .npy
    y_output_path = output_dir / 'y_train.npy' # Overwriting old .npy

    print(f"Loading processed data from {processed_data_path}...")
    df = pd.read_csv(processed_data_path)
    
    y_data = df['target']
    X_data = df.drop(columns=['target'])
    
    window_size = config['feature_engineering']['window_size']
    print(f"Applying statistical windowing with size: {window_size}")

    X_features, y_windows = create_statistical_features(X_data, y_data, window_size)
    y_windows = y_windows.astype(int)

    print(f"Saving 2D feature array to {X_output_path}...")
    np.save(X_output_path, X_features)
    print(f"Saving label array to {y_output_path}...")
    np.save(y_output_path, y_windows)

    print("\nFeature engineering complete!")
    print(f"  -> Final X shape (Samples, Stats_Features): {X_features.shape}")
    print(f"  -> Final y shape (Samples,): {y_windows.shape}")

if __name__ == '__main__':
    main()