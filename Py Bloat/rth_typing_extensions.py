"""
Runtime hook for typing_extensions.

This hook ensures that typing_extensions is properly initialized at runtime.
"""

import os
import sys
import importlib.metadata

# Add a fallback for typing_extensions metadata
if 'typing_extensions' not in importlib.metadata.packages_distributions():
    # Create a fake distribution for typing_extensions
    class FakeDistribution:
        def read_text(self, path):
            if path == 'METADATA':
                return 'Name: typing-extensions\nVersion: 4.0.0\n'
            return ''
        
        def locate_file(self, path):
            return ''
    
    # Patch importlib.metadata to return our fake distribution
    _orig_distribution = importlib.metadata.distribution
    def patched_distribution(name):
        if name == 'typing_extensions':
            return FakeDistribution()
        return _orig_distribution(name)
    
    importlib.metadata.distribution = patched_distribution
