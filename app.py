# app.py (V4: Final Polish with Toggles)

import streamlit as st
import pandas as pd
import numpy as np
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

# --- Class Labels Definition ---
# We define this globally so all parts of the app can use it
CLASS_LABELS = {0:"Gas 1", 1:"Gas 2", 2:"Gas 3", 3:"Gas 4", 4:"Gas 5", 5:"Gas 6"}

# --- Session State Initialization ---
if 'sim_index' not in st.session_state:
    st.session_state.sim_index = 0
if 'auto_play' not in st.session_state:
    st.session_state.auto_play = False
if 'prediction_log' not in st.session_state:
    st.session_state.prediction_log = []
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = pd.DataFrame(columns=CLASS_LABELS.values())
# Initialize toggles
for gas_name in CLASS_LABELS.values():
    if f"show_{gas_name}" not in st.session_state:
        st.session_state[f"show_{gas_name}"] = True # Default to show all

# --- Caching Functions ---
@st.cache_resource
def load_model_and_scaler():
    try:
        config = load_config()
        scaler = joblib.load(config['paths']['scaler_output'])
        model_path = (PROJECT_ROOT / config['paths']['model_output']).with_suffix('.joblib')
        model = joblib.load(model_path)
        return model, scaler, config
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

# --- Inference Helper (Returns ALL probabilities) ---
def run_inference(window_data, scaler, model):
    scaled_window = window_data # Data is already scaled
    features = np.concatenate([
        np.mean(scaled_window, axis=0),
        np.std(scaled_window, axis=0),
        np.min(scaled_window, axis=0),
        np.max(scaled_window, axis=0)
    ]).reshape(1, -1)
    
    pred_proba = model.predict_proba(features)[0] 
    predicted_class = np.argmax(pred_proba)
    confidence = np.max(pred_proba)
        
    return predicted_class, confidence, pred_proba 

# --- Main App Layout ---
def main():
    st.set_page_config(page_title="Gas Dashboard V4", layout="wide")
    st.title("Gas Detection Dashboard 🚨 (V4: Final)")
    
    model, scaler, config = load_model_and_scaler()
    sim_df = load_simulation_data()

    if sim_df.empty or model is None:
        st.error("Critical resources failed to load. App cannot start.")
        st.stop()

    # --- Sidebar ---
    st.sidebar.header("Configuration & Status")
    st.sidebar.success("System Online (Random Forest) 🚀")
    speed = st.sidebar.slider("Simulation Speed (delay)", 0.01, 0.5, 0.05, key="speed_slider")
    st.sidebar.markdown("---")
    
    # --- NEW: Toggle Buttons ---
    st.sidebar.subheader("Toggle Gas Plots:")
    visible_gases = []
    for gas_name in CLASS_LABELS.values():
        if st.sidebar.toggle(gas_name, key=f"show_{gas_name}"):
            visible_gases.append(gas_name)
    
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Current Time Step:** t={st.session_state.sim_index}")

    # --- Main Area ---
    col1, col2 = st.columns([2, 1]) 
    with col1:
        st.subheader("Live Model Confidence (All Gases):")
        st.write("") # Add spacing
        
        # Filter chart data based on toggles
        if not st.session_state.chart_data.empty and visible_gases:
            chart_ph = st.line_chart(st.session_state.chart_data[visible_gases], height=400)
        else:
            # Show empty chart if no toggles are on
            chart_ph = st.line_chart(pd.DataFrame(columns=CLASS_LABELS.values()), height=400)
            
    with col2:
        st.subheader("Prediction Log 📜")
        st.write("") # Add spacing
        log_container = st.container(height=400)
        for msg_type, msg_text in st.session_state.prediction_log:
            if msg_type == 'error':
                log_container.error(msg_text)
            else:
                log_container.info(msg_text)

    # --- Simulation Logic Function ---
    def advance_simulation():
        window_size = config['feature_engineering']['window_size']
        current_idx = st.session_state.sim_index
        
        if current_idx < len(sim_df) - window_size:
            window_df = sim_df.iloc[current_idx : current_idx + window_size]
            feats = window_df.drop(columns=['target']) if 'target' in window_df.columns else window_df
            true_label_val = int(sim_df.iloc[current_idx + window_size - 1]['target'])
            
            pred_cls, conf, all_probs = run_inference(feats, scaler, model)
            
            pred_res = CLASS_LABELS.get(pred_cls, "Unknown")
            true_res = CLASS_LABELS.get(true_label_val - 1, "Unknown")
            
            msg_text = f"[t={current_idx}] PREDICTED: {pred_res} | ACTUAL: {true_res} (Conf: {conf*100:.1f}%)"
            msg_type = "info" if pred_res == true_res else "error"
            st.session_state.prediction_log.insert(0, (msg_type, msg_text))
            
            # Update Chart (row must have all columns, filtering happens at display time)
            new_prob_row = pd.DataFrame([all_probs], columns=CLASS_LABELS.values())
            # Use the filtered list for add_rows
            if visible_gases:
                chart_ph.add_rows(new_prob_row[visible_gases])
            st.session_state.chart_data = pd.concat([st.session_state.chart_data, new_prob_row], ignore_index=True)
            
            st.session_state.sim_index += 1
            return True
        else:
            st.warning("End of simulation.")
            st.session_state.auto_play = False
            return False

    # --- Controls ---
    st.markdown("---") # Extra spacer for Req 5
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("▶ Next Step", use_container_width=True):
            advance_simulation()
    with c2:
        if st.button("⏯ Auto-Play", use_container_width=True):
            st.session_state.auto_play = not st.session_state.auto_play
            st.rerun()
    with c3:
        if st.button("🔄 Reset (t=0)", use_container_width=True):
            st.session_state.sim_index = 0
            st.session_state.chart_data = pd.DataFrame(columns=CLASS_LABELS.values())
            st.session_state.prediction_log = []
            st.session_state.auto_play = False
            st.rerun()
    with c4:
        if st.button("⏩ Jump to t=300", use_container_width=True, type="primary"):
             st.session_state.sim_index = 300
             st.session_state.chart_data = pd.DataFrame(columns=CLASS_LABELS.values())
             st.session_state.prediction_log = []
             st.rerun()

    if st.session_state.auto_play:
        time.sleep(speed)
        if advance_simulation():
            st.rerun()

if __name__ == '__main__':
    main()