"""
Test script to verify error logging configuration.

This script tests the new logging configuration that only shows errors.
"""

import sys
import os
import logging

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the logging utility
from src.utils.logging_utils import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

def test_logging_levels():
    """Test different logging levels to verify only errors are shown."""
    print("\nTesting logging levels - you should only see ERROR and CRITICAL in the console:")
    
    logger.debug("This is a DEBUG message - should not be visible in console")
    logger.info("This is an INFO message - should not be visible in console")
    logger.warning("This is a WARNING message - should not be visible in console")
    logger.error("This is an ERROR message - should be visible in console")
    logger.critical("This is a CRITICAL message - should be visible in console")
    
    print("\nCheck the error_logs directory for the error log file")
    print("Check crypto_scanner.log for all log messages")

def simulate_error():
    """Simulate an error condition."""
    try:
        # Intentionally cause an error
        result = 1 / 0
    except Exception as e:
        logger.error(f"Caught an error: {e}")

def main():
    """Main function."""
    print("Testing Error Logging Configuration")
    print("==================================")
    
    # Test logging levels
    test_logging_levels()
    
    # Simulate an error
    print("\nSimulating an error:")
    simulate_error()
    
    print("\nTest completed. Check the log files.")

if __name__ == "__main__":
    main()
