"""
Logging Configuration and Utilities
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
import yaml


def setup_logger(
    name: str = "traffic_system",
    config_path: Optional[str] = None,
    log_level: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure logger for the application.
    
    Args:
        name: Logger name
        config_path: Path to config.yaml file
        log_level: Override log level from config (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Default configuration
    log_config = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "logs/traffic_system.log",
        "max_bytes": 10485760,  # 10MB
        "backup_count": 5
    }
    
    # Load configuration from file if provided
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'logging' in config:
                    log_config.update(config['logging'])
        except Exception as e:
            print(f"Warning: Could not load logging config: {e}")
    
    # Override log level if provided
    if log_level:
        log_config['level'] = log_level
    
    # Convert string level to logging constant
    level = getattr(logging, log_config['level'].upper(), logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(log_config['format'])
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file = Path(log_config['file'])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=log_config['max_bytes'],
        backupCount=log_config['backup_count']
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(log_config['format'])
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "traffic_system") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # If no handlers exist, set up with defaults
        logger = setup_logger(name)
    return logger

