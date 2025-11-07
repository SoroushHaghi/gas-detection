import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.gas_detection.config import load_config
from src.gas_detection.models.model_factory import build_cnn_model, build_gru_model

def main():
    """Main function to train the champion model."""
    # Load config from config.yml
    config = load_config()

    # Define paths for X and y data
    processed_data_path = project_root / config['paths']['processed_data']
    X_path = processed_data_path.with_name(processed_data_path.stem.replace('data_processed', 'X_train') + '.npy')
    y_path = processed_data_path.with_name(processed_data_path.stem.replace('data_processed', 'y_train') + '.npy')

    # Load data
    X = np.load(X_path)
    y = np.load(y_path)
    y = y - 1

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Determine shapes
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y))

    # Build model
    model = build_cnn_model(input_shape, num_classes)

    # Compile
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

    # Save model
    model_output_path = project_root / config['paths']['model_output']
    model.save(model_output_path)

    # Print success message
    print(f"Model trained and saved to {model_output_path}")

if __name__ == '__main__':
    main()