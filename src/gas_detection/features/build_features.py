
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.gas_detection.config import load_config

def create_windows(data, labels, window_size):
    """
    Creates time-series windows from the data.
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(labels[i + window_size])
    return np.array(X), np.array(y)

def main():
    """
    Main function to run the feature engineering pipeline.
    """
    # Load configuration
    config = load_config()

    # Load processed data
    processed_data_path = project_root / config['paths']['processed_data']
    df = pd.read_csv(processed_data_path)

    # Separate features and target
    features = df.drop('target', axis=1).values
    target = df['target'].values

    # Get window size from config
    window_size = config['feature_engineering']['window_size']

    # Create windows
    X, y = create_windows(features, target, window_size)
    
    # Convert y to integer type
    y = y.astype(int)

    # Save X and y as .npy files
    output_dir = processed_data_path.parent
    np.save(output_dir / 'X_train.npy', X)
    np.save(output_dir / 'y_train.npy', y)

    # Print final shapes
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

if __name__ == '__main__':
    main()
