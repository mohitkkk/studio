"""
Backend configuration settings
"""
import os
import logging
import torch
from typing import Dict, Any

# Server configuration
SERVER_CONFIG = {
    "host": "0.0.0.0",  # Listen on all interfaces
    "port": 3000,
    "public_ip": "13.202.208.115",  # Public-facing IP
    "cors_origins": ["*"],  # For development; restrict in production
    "reload": False,     # Set to True during development
    "log_level": "info"

}

# Default configuration values
DEFAULT_CONFIG = {
    "vault_directory": "vault_files/",
    "vault_metadata": "vault_metadata.json",
    "chat_history_directory": "chat_histories/",
    "log_file": "chatbot.log",
    "log_level": "INFO",
    "ollama_model": "llama3",
    "ollama_embedding_model": "mxbai-embed-large",
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k_per_file": 7,
    "similarity_threshold": 0.5,
    "system_prompt": "",
    "clean_response": True,
    "remove_hedging": True,
    "remove_references": True,
    "remove_disclaimers": True,
    "ensure_html_structure": True,
    "no_info_title": "Information Not Found",
    "no_info_message": "",
    "include_suggestions": True,
    "no_info_html_format": True
}

# Ollama configuration
OLLAMA_CONFIG = {
    "host": os.environ.get("OLLAMA_HOST", "http://13.202.208.115:11434"),
    "model": os.environ.get("OLLAMA_MODEL", "llama3"),
    "embedding_model": os.environ.get("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large"),
}

def get_full_config() -> Dict[str, Any]:
    """Returns the complete configuration dictionary"""
    config = DEFAULT_CONFIG.copy()
    
    # Override with environment variables if present
    for key in config:
        env_key = f"CHATBOT_{key.upper()}"
        if env_key in os.environ:
            # Handle different types of config values
            original_value = config[key]
            env_value = os.environ[env_key]
            
            # Convert to appropriate type based on the original value
            if isinstance(original_value, bool):
                config[key] = env_value.lower() in ("true", "1", "yes")
            elif isinstance(original_value, int):
                config[key] = int(env_value)
            elif isinstance(original_value, float):
                config[key] = float(env_value)
            else:
                config[key] = env_value
                
            logging.info(f"Config override from environment: {key}={config[key]}")
    
    # Set device for PyTorch operations
    if torch.cuda.is_available():
        config["device"] = torch.device("cuda")
        config["use_gpu"] = True
        logging.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        config["device"] = torch.device("cpu")
        config["use_gpu"] = False
        logging.info("No GPU available, using CPU")
    
    # Ensure important directories from config have trailing slashes
    for key in ["vault_directory", "chat_history_directory"]:
        if key in config and not config[key].endswith('/'):
            config[key] += '/'
    
    return config
