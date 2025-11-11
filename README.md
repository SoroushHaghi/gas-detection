# Gas Detection MLOps Pipeline (V5 - RandomForest)

## 🚀 Live Dashboard
[**Click Here to View the Live Project**](https://gas-detection-tubs.streamlit.app)

*(Note: The dashboard may take 1-2 minutes to "wake up" if it has been idle.)*

---

## 1. Project Summary

This repository contains a complete, end-to-end MLOps pipeline designed to classify six different types of gases in real-time. The system uses data from a 16-sensor array (128 features) and simulates a live data stream in an interactive Streamlit dashboard.

The final model is a **RandomForestClassifier (V5)** that successfully predicts gas types with high accuracy. The dashboard features a **live updating bar chart** that visualizes the model's confidence for all six gases at each time step.

---

## 2. The Project Journey: A Tale of Two Models

This project was a practical exercise in MLOps, demonstrating not just model building, but critical debugging, pivoting, and optimization.

### V1 (The Failure): 1D-CNN on Raw Time-Series
The initial approach was to use a Deep Learning model (1D-CNN) directly on the raw, windowed sensor data (`(50, 128)` tensors).

* **Result:** The model completely "collapsed" and predicted **"Gas 1"** for every single time step, regardless of the actual gas present.
* **Diagnosis:** The 128 features contained too much noise and not enough clear signal for the CNN to learn effectively.

### V5 (The Pivot & Success): RandomForest on Statistical Features
Based on the V1 failure, we pivoted the entire strategy:
1.  **New Features:** We abandoned raw data. A feature engineering script (`build_features.py`) was created to calculate statistical features (mean, std, min, max) across the 50-step window. This compressed the noisy `(50, 128)` tensor into a highly informative `(1, 512)` feature vector.
2.  **New Model:** We replaced the CNN with a `RandomForestClassifier`, which excels at tabular (statistical) data.
3.  **New Result:** The new model trained to **~96% accuracy** and successfully predicts the correct gas type in real-time.

This pivot demonstrates a key MLOps principle: **Intelligent Feature Engineering often beats a complex Model Architecture.**

---

## 3. Technology Stack

* **Deployment:** Streamlit Community Cloud
* **Dashboard:** Streamlit
* **Model:** Scikit-learn (`RandomForestClassifier`)
* **Data Processing:** Pandas, NumPy
* **Versioning:** Git (GitLab with GitHub Mirroring)
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
    *(This runs the V5 pipeline to create the statistical features and the model/scaler files in the `artifacts/` directory)*
    ```bash
    python src/gas_detection/data/process_data.py
    python src/gas_detection/features/build_features.py
    python src/gas_detection/models/train_model.py
    ```
5.  **Run the Live Dashboard:**
    ```bash
    streamlit run app.py
    ```