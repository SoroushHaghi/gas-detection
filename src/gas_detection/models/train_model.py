# src/gas_detection/models/train_model.py

import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- Add Project Root to sys.path ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()
sys.path.append(str(PROJECT_ROOT))
# ------------------------------------

from src.gas_detection.config import load_config

def main():
    print("Starting Model Training (Random Forest)...")
    config = load_config()
    
    # 1. Load Data (from new build_features.py)
    processed_dir = PROJECT_ROOT / Path(config['paths']['processed_data']).parent
    X_path = processed_dir / 'X_train.npy'
    y_path = processed_dir / 'y_train.npy'
    
    print("Loading statistical features...")
    X = np.load(X_path)
    y = np.load(y_path)
    
    # 2. Map labels from [1-6] to [0-5]
    y = y - 1 
    
    # 3. Split Data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Training with {X_train.shape[0]} samples, validating with {X_val.shape[0]} samples.")

    # 4. Build & Train Model
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 5. Save Model (as .joblib, not .h5)
    model_output_path = (PROJECT_ROOT / config['paths']['model_output']).with_suffix('.joblib')
    
    print(f"Saving model to {model_output_path}...")
    joblib.dump(model, model_output_path)
    
    # 6. Evaluation
    y_pred = model.predict(X_val)
    print("\n--- Evaluation Report (Random Forest) ---")
    print(classification_report(y_val, y_pred))

if __name__ == '__main__':
    main()