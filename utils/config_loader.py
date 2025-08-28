import yaml
import os
from dotenv import load_dotenv, find_dotenv

def load_config(config_path: str = "config/config.yaml") -> dict:
    # Load environment variables from .env file first
    load_dotenv(find_dotenv(), override=True)

    # Load YAML config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        print(config)
    return config



