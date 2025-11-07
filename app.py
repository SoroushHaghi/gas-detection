import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.gas_detection.config import load_config

@st.cache_resource
def load_model_and_scaler():
    try:
        config = load_config()
        scaler = joblib.load(config['paths']['scaler_output'])
        interpreter = tf.lite.Interpreter(model_path=config['paths']['tflite_output'])
        interpreter.allocate_tensors()
        return interpreter, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

@st.cache_data
def load_simulation_data():
    try:
        config = load_config()
        df = pd.read_csv(config['paths']['processed_data'])
        return df
    except Exception as e:
        st.error(f"Error loading simulation data: {e}")
        return None

st.title("Gas Detection Dashboard 🚨")

st.sidebar.header("Configuration")
interpreter, scaler = load_model_and_scaler()
sim_df = load_simulation_data()

if interpreter and scaler and sim_df is not None:
    st.sidebar.success("Model & Scaler loaded from cache! 🚀")
    st.sidebar.info(f"Simulation data loaded: {len(sim_df)} rows")
    st.dataframe(sim_df.head())