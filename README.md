# Gas Detection MLOps Pipeline (V4 - RandomForest)

## 🚀 Live Dashboard (In Progress)

**[Your Live Streamlit Dashboard Link Will Go Here]**

---

## 1. Project Summary

This repository contains a complete, end-to-end MLOps pipeline designed to classify six different types of gases in real-time. The system uses data from a 16-sensor array (128 features) and simulates a live data stream in an interactive Streamlit dashboard.

The final model is a **RandomForestClassifier (V4)** that successfully predicts gas types with high accuracy, running efficiently on a live dashboard.

---

## 2. The Project Journey: A Tale of Two Models

This project was a practical exercise in MLOps, demonstrating not just model building, but critical debugging, pivoting, and optimization.

### V1 (The Failure): 1D-CNN on Raw Time-Series
The initial approach was to use a Deep Learning model (1D-CNN) directly on the raw, windowed sensor data (`(50, 128)` tensors).

* **Result:** The model completely failed. It "collapsed" and predicted **"Gas 1"** for every single time step, regardless of the actual gas present.
* **Diagnosis:** The 128 features contained too much noise and not enough clear signal for the CNN to learn effectively.

### V2/V3/V4 (The Pivot & Success): RandomForest on Statistical Features
Based on the V1 failure, we pivoted the entire strategy:
1.  **New Features:** We abandoned raw data. A new feature engineering script (`build_features.py`) was created to calculate statistical features (mean, std, min, max) across the 50-step window. This compressed the noisy `(50, 128)` tensor into a highly informative `(1, 512)` feature vector.
2.  **New Model:** We replaced the CNN with a `RandomForestClassifier`, which excels at tabular data.
3.  **New Result:** The new model trained to **~96% accuracy** and, as seen in the final dashboard, successfully predicts the correct gas type in real-time with high confidence.

This pivot demonstrates a key MLOps principle: **Feature Engineering often beats complex Model Architecture.**

---

## 3. Technology Stack

* **Dashboard:** Streamlit
* **Model:** Scikit-learn (`RandomForestClassifier`)
* **Data Processing:** Pandas, NumPy
* **Core Libraries:** Joblib, PyYAML

---

## 4. How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://gitlab.com/SoroushHaghi/gas_detection.git](https://gitlab.com/SoroushHaghi/gas_detection.git)
    cd gas_detection
    ```
2.  **(Recommended)** Create and activate a virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Build Features & Train Model (One-time setup):**
    *(This runs the V4 pipeline to create the statistical features and the `champion_model.joblib`)*
    ```bash
    python src/gas_detection/features/build_features.py
    python src/gas_detection/models/train_model.py
    ```
5.  **Run the Live Dashboard:**
    ```bash
    streamlit run app.py
    ```

---

## 5. Project Structure (V4)

gas_detection/ ├── .gitignore ├── app.py # The final Streamlit dashboard (V4) ├── config.yml # Project configuration (paths, parameters) ├── data/ │ ├── processed/ │ │ ├── data_processed.csv # Scaled sensor data (output of Day 3) │ │ ├── X_train.npy # Statistical features for training (output of V4 Feature Eng) │ │ └── y_train.npy # Labels for training │ └── raw/ │ └── gas_data.csv # The raw, combined LibSVM dataset ├── models/ │ ├── champion_model.joblib # The trained RandomForest model (V4) │ └── scaler.pkl # The StandardScaler (from Day 3) ├── notebooks/ │ └── eda_analysis.py # Exploratory Data Analysis script ├── README.md # This file ├── requirements.txt # Python dependencies └── src/ └── gas_detection/ ├── init.py ├── config.py # Config loader utility ├── features/ │ └── build_features.py # V4 Feature Engineering (Statistical) └── models/ └── train_model.py # V4 Model Training (RandomForest)