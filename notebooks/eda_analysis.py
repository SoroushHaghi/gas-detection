
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import sys
import yaml

# Add the project root to the Python path
# This allows us to import modules from the 'src' directory
try:
    root_path = pathlib.Path(__file__).resolve().parents[1]
    sys.path.append(str(root_path))
    from src.gas_detection.config import load_config
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from the project's root directory or that the 'src' directory is in the PYTHONPATH.")
    sys.exit(1)

def main():
    """
    Main function to run the EDA analysis.
    """
    # 1. Load config
    try:
        config_path = root_path / 'config.yml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: The configuration file was not found at {config_path}")
        sys.exit(1)

    # 2. Load the processed data
    processed_data_path = root_path / config['paths']['processed_data']
    try:
        df = pd.read_csv(processed_data_path)
    except FileNotFoundError:
        print(f"Error: The processed data file was not found at {processed_data_path}")
        sys.exit(1)

    # 3. Class Balance Analysis
    print("--- Class Balance Analysis ---")
    class_balance = df['target'].value_counts(normalize=True)
    print(class_balance * 100)
    print("-" * 30)

    # 4. Define class labels and map them
    # NOTE: This mapping is based on the expected structure of the dataset.
    # It might need adjustment if the target encoding is different.
    class_labels = {
        1: "Gas 1",
        2: "Gas 2",
        3: "Gas 3",
        4: "Gas 4",
        5: "Gas 5",
        6: "Gas 6"
    }
    df['target_label'] = df['target'].map(class_labels)

    # 5. Feature Correlation Analysis (Box Plot)
    print("--- Feature Correlation Analysis ---")
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='target_label', y='S1', data=df, order=class_labels.values())
    plt.title('Sensor S1 Readings vs. Gas Type')
    plt.xlabel('Gas Type')
    plt.ylabel('Sensor S1 Reading')
    plt.xticks(rotation=45)
    plt.grid(True)

    # 6. Save the plot
    output_dir = root_path / 'reports' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'eda_class_distribution.png'
    
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot saved successfully to: {plot_path}")


if __name__ == '__main__':
    main()
