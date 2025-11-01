import yaml
from pathlib import Path
from typing import Dict, Any
import os

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    config_path = Path(config_path)
    
    if not config_path.exists():
        default_config = {
            "models": {
                "default_model": "codegen-2b",
                "auto_download": True,
                "cache_models": True
            },
            "generation": {
                "default_temperature": 0.7,
                "max_length": 300,
                "num_suggestions": 3
            },
            "analysis": {
                "enable_linting": True,
                "enable_type_checking": True,
                "enable_security_scan": True
            },
            "ui": {
                "theme": "dark",
                "show_line_numbers": True,
                "auto_format": True
            }
        }
        save_config(default_config, config_path)
        return default_config
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], config_path: str = "config.yaml"):
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def get_model_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("models", {})

def get_generation_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("generation", {})

def update_config(section: str, key: str, value: Any):
    config = load_config()
    
    if section not in config:
        config[section] = {}
    
    config[section][key] = value
    save_config(config)

def get_default_generation_params() -> Dict[str, Any]:
    config = load_config()
    generation = config.get("generation", {})
    
    return {
        "temperature": generation.get("default_temperature", 0.7),
        "max_length": generation.get("max_length", 300),
        "num_suggestions": generation.get("num_suggestions", 3)
    }