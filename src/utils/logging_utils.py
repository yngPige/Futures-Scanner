"""
Logging utilities for 3lacks Scanner.

This module provides custom logging utilities for the application.
"""

import os
import logging
from datetime import datetime

class ErrorLogFilter(logging.Filter):
    """Filter that only allows ERROR and CRITICAL level logs."""
    
    def filter(self, record):
        """Filter logs based on level."""
        # Only allow ERROR and CRITICAL levels
        return record.levelno >= logging.ERROR

def configure_logging():
    """Configure logging to only show errors in console and save to error log file."""
    # Create error logs directory if it doesn't exist
    error_logs_dir = 'error_logs'
    os.makedirs(error_logs_dir, exist_ok=True)
    
    # Generate error log filename with timestamp
    error_log_filename = os.path.join(error_logs_dir, f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Create handlers
    # Console handler (only show errors)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.addFilter(ErrorLogFilter())
    
    # Error file handler (only errors)
    error_file_handler = logging.FileHandler(error_log_filename)
    error_file_handler.setLevel(logging.ERROR)
    
    # Full log file handler (all levels for debugging)
    full_file_handler = logging.FileHandler("crypto_scanner.log")
    full_file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    error_file_handler.setFormatter(formatter)
    full_file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all logs
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(error_file_handler)
    root_logger.addHandler(full_file_handler)
    
    return root_logger
