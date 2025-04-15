#!/usr/bin/env python3
import os
import yaml

def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_ddpg.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")