import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import sys
from pathlib import Path
import time

# --- Add Project Root to sys.path ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.gas_detection.config import load_config

# --- Session State Initialization ---
if 'sim_index' not in st.session_state:
    st.session_state.sim_index = 0
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = pd.DataFrame(columns=['S1'])
if 'auto_play' not in st.session_state:
    st.session_state.auto_play = False

# --- Caching Functions ---
@st.cache_resource
def load_model_and_scaler():
    try:
        config = load_config()
        scaler = joblib.load(config['paths']['scaler_output'])
        interpreter = tf.lite.Interpreter(model_path=config['paths']['tflite_output'])
        interpreter.allocate_tensors()
        return interpreter, scaler, config
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None

@st.cache_data
def load_simulation_data():
    try:
        config = load_config()
        return pd.read_csv(config['paths']['processed_data'])
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# --- Inference Helper ---
def run_inference(window_data, scaler, interpreter):
    scaled_window = scaler.transform(window_data)
    input_data = np.expand_dims(scaled_window, axis=0).astype(np.float32)
    input_details = interpreter.get_input_details()
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        input_data = (input_data / scale + zero_point).astype(np.int8)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - zero_point) * scale
    return np.argmax(output_data), np.max(output_data)

# --- Main App Layout ---
def main():
    st.title("Gas Detection Dashboard 🚨")
    interpreter, scaler, config = load_model_and_scaler()
    sim_df = load_simulation_data()

    if sim_df.empty or interpreter is None:
        st.stop()

    st.sidebar.success("System Online 🚀")
    speed = st.sidebar.slider("Simulation Speed (delay in seconds)", 0.01, 1.0, 0.1)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Live Prediction Status:")
    pred_ph = st.sidebar.empty()

    st.subheader("Live Sensor Data (S1):")
    chart_ph = st.line_chart(st.session_state.chart_data)

    # --- Simulation Logic Function ---
    def advance_simulation():
        window_size = config['feature_engineering']['window_size']
        current_idx = st.session_state.sim_index
        if current_idx < len(sim_df) - window_size:
            window_df = sim_df.iloc[current_idx : current_idx + window_size]
            feats = window_df.drop(columns=['target']) if 'target' in window_df.columns else window_df
            cls, conf = run_inference(feats, scaler, interpreter)
            
            lbls = {0:"Normal✅", 1:"Ammonia⚠️", 2:"Toluene☢️", 3:"Acetone", 4:"Ethylene", 5:"Ethanol"}
            res = lbls.get(cls, f"Unknown({cls})")
            if cls == 0: pred_ph.success(f"Status: {res}")
            else: pred_ph.error(f"Status: {res}\nConfidence: {conf:.2f}")
            
            new_row = pd.DataFrame({'S1': [sim_df.iloc[current_idx + window_size - 1]['S1']]})
            chart_ph.add_rows(new_row)
            st.session_state.chart_data = pd.concat([st.session_state.chart_data, new_row], ignore_index=True)
            st.session_state.sim_index += 1
            return True
        else:
            st.warning("End of simulation.")
            st.session_state.auto_play = False
            return False

    # --- Controls ---
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("▶ Next Step", use_container_width=True):
            advance_simulation()
    with c2:
        if st.button("⏯ Auto-Play", use_container_width=True):
            st.session_state.auto_play = not st.session_state.auto_play
            st.rerun()
    with c3:
        if st.button("🔄 Reset (t=0)", use_container_width=True, type="primary"):
            st.session_state.sim_index = 0
            st.session_state.chart_data = pd.DataFrame(columns=['S1'])
            st.session_state.auto_play = False
            st.rerun()
    with c4:
        if st.button("⏩ Jump to t=300", use_container_width=True):
            st.session_state.sim_index = 300
            st.session_state.chart_data = pd.DataFrame(columns=['S1'])
            st.rerun()

    if st.session_state.auto_play:
        time.sleep(speed)
        if advance_simulation():
            st.rerun()

    st.sidebar.write(f"**Time Step:** t={st.session_state.sim_index}")

if __name__ == '__main__':
    main()