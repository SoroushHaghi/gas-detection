# Gas Detection MLOps Pipeline (V-Final: 10-Second Model)

## 🚀 Live Dashboard
[**Click Here to View the Live Project**](https://gas-detection-tubs.streamlit.app)

*(Note: The dashboard may take 1-2 minutes to "wake up" if it has been idle.)*

---

## 1. Project Summary

This repository contains a complete, end-to-end MLOps pipeline designed to classify six different types of gases in real-time. The system uses data from a 16-sensor array (128 features) and simulates a live data stream in an interactive Streamlit dashboard.

The final model is a **RandomForestClassifier** that uses a **10-second data window** and achieves **95% accuracy**. This provides a rapid-response "digital nose" that balances high precision with low latency.

---

## 2. The Project Journey: A Tale of Two Models

This project was a practical exercise in MLOps, demonstrating not just model building, but critical debugging, pivoting, and optimization.

### V1 (The Failure): 1D-CNN on Raw Time-Series
The initial approach was to use a Deep Learning model (1D-CNN) directly on the raw, windowed sensor data (`(50, 128)` tensors).

*   **Result:** The model "collapsed" and predicted **"Ethanol"** (the most common class) for every single time step, regardless of the actual gas present.
*   **Diagnosis:** The 128 features contained too much noise for the CNN to learn effectively.

### V-Final (The Pivot & Success): RandomForest on Statistical Features
Based on the V1 failure, we pivoted the entire strategy:
1.  **New Features:** We abandoned raw data. A feature engineering script (`build_features.py`) was created to calculate statistical features (mean, std, min, max) across a data window.
2.  **New Model:** We replaced the CNN with a `RandomForestClassifier`, which excels at tabular (statistical) data.
3.  **Optimization (The 10-Second Test):** We discovered that reducing the data window from 50 seconds (`window_size: 50`) to **10 seconds** (`window_size: 10`) resulted in only a **1% drop in accuracy** (96% -> 95%). This was a massive win, creating a model that is 5x faster while remaining highly accurate.

---

## 3. Technology Stack

*   **Deployment:** Streamlit Community Cloud
*   **Dashboard:** Streamlit
*   **Model:** Scikit-learn (`RandomForestClassifier`)
*   **Data Processing:** Pandas, NumPy
*   **Versioning:** Git (GitLab with GitHub Mirroring)
*   **Core Libraries:** Joblib, PyYAML

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
    *(This runs the final pipeline to create the 10-second features and the model/scaler files in the `models/` directory)*
    ```bash
    python src/gas_detection/data/process_data.py
    python src/gas_detection/features/build_features.py
    python src/gas_detection/models/train_model.py
    ```
5.  **Run the Live Dashboard:**
    ```bash
    streamlit run app.py
    ```

---

## 5. Project Structure (V-Final)

gas_detection/ ├── .gitignore ├── app.py # The final Streamlit dashboard (V-Final) ├── config.yml # Project configuration (paths, window_size: 10) ├── data/ │ ├── processed/ │ │ ├── data_processed.csv # Scaled sensor data │ │ ├── X_train.npy # Statistical features (10s window) │ │ └── y_train.npy # Labels for training │ └── raw/ │ └── gas_data.csv # The raw, combined dataset ├── models/ │ ├── champion_model.joblib # The trained 10-second RandomForest model │ └── scaler.pkl # The StandardScaler ├── notebooks/ │ └── eda_analysis.py # Exploratory Data Analysis script ├── README.md # This file ├── requirements.txt # Python dependencies └── src/ └── gas_detection/ ├── init.py ├── config.py # Config loader utility ├── data/ │ └── process_data.py # Script to clean and scale data ├── features/ │ └── build_features.py # Feature Engineering (Statistical) └── models/ └── train_model.py # Model Training (RandomForest)