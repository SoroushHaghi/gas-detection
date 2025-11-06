# src/gas_detection/config.py

import yaml
from pathlib import Path

def load_config(config_path="config.yml"):
    """
    Securely loads the configuration from a YAML file.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    try:
        with open(config_path, 'r') as file:
            # Critical: Use safe_load for security
            config = yaml.safe_load(file)
        print("Configuration loaded successfully.")
        return config
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None
    except Exception as e:
        print(f"Unexpected error loading configuration: {e}")
        return None

# Example usage (for testing the module itself)
if __name__ == "__main__":
    # For testing, assumes config.yml is in the parent directory
    # (when run from inside src/gas_detection)
    # A more robust test would find the project root, but this is for a quick check.
    
    # Let's try to find the root directory relative to this file
    # This file is in: .../src/gas_detection/config.py
    # We want to go up 3 levels to the project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    CONFIG_FILE_PATH = PROJECT_ROOT / "config.yml"

    print(f"Attempting to load config from: {CONFIG_FILE_PATH}")
    config = load_config(CONFIG_FILE_PATH) 
    
    if config:
        print("\nConfiguration Content:")
        print(config)
        print(f"\nRaw data path: {config['paths']['raw_data']}")
        print(f"Window size: {config['data']['window_size']}")
