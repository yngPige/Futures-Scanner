#!/usr/bin/env python
"""
Test script for LLM model loading.

This script tests the LLM model loading functionality with improved error handling.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Import the monkey patch first
import monkey_patch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_loading(model_key, use_gpu=False):
    """Test loading an LLM model."""
    try:
        from src.analysis.local_llm import LocalLLMAnalyzer, AVAILABLE_MODELS, DEFAULT_MODEL_PATH
        
        # Get model info
        model_info = AVAILABLE_MODELS.get(model_key, {})
        model_name = model_info.get('name', model_key)
        
        # Print model info
        print(f"Testing model loading for {model_key}:")
        print(f"- Model name: {model_name}")
        print(f"- Model size: {model_info.get('size_gb', 'unknown')} GB")
        print(f"- Model path: {os.path.join(DEFAULT_MODEL_PATH, model_name)}")
        print(f"- GPU acceleration: {'enabled' if use_gpu else 'disabled'}")
        
        # Configure GPU usage
        n_gpu_layers = -1 if use_gpu else 0
        
        # Initialize the analyzer
        print("\nInitializing LLM analyzer...")
        llm_analyzer = LocalLLMAnalyzer(
            model_name=model_name,
            n_gpu_layers=n_gpu_layers
        )
        
        # Check if LLM was initialized successfully
        if llm_analyzer.llm is None:
            print("\nERROR: Failed to initialize LLM model.")
            return False
        
        print("\nModel loaded successfully!")
        return True
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test LLM model loading")
    parser.add_argument("--model", type=str, default="llama3-8b", help="Model key to test")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    
    args = parser.parse_args()
    
    # Test model loading
    success = test_model_loading(args.model, args.gpu)
    
    if success:
        print("\nTest passed!")
        sys.exit(0)
    else:
        print("\nTest failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
