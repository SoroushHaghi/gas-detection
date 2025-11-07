import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.gas_detection.config import load_config

def representative_data_gen(X_data, num_samples=100):
    for i in range(num_samples):
        yield [X_data[i:i+1].astype(np.float32)]

def main():
    # Load config
    config = load_config()

    # Load X_train for representative dataset
    processed_data_path = project_root / config['paths']['processed_data']
    X_train_path = processed_data_path.with_name(processed_data_path.stem.replace('data_processed', 'X_train') + '.npy')
    X_train = np.load(X_train_path)

    # Load the Keras model
    model_path = project_root / config['paths']['model_output']
    model = tf.keras.models.load_model(model_path)

    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Set optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Set representative dataset
    converter.representative_dataset = lambda: representative_data_gen(X_train)

    # Enforce full INT8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert
    tflite_model = converter.convert()

    # Save the TFLite model
    tflite_output_path = project_root / config['paths']['tflite_output']
    with open(tflite_output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model saved to {tflite_output_path}")

if __name__ == '__main__':
    main()