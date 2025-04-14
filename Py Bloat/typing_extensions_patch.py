"""
Monkey patch for typing_extensions.

This patch ensures that typing_extensions works correctly in the executable.
"""

import sys
import importlib.metadata

# Define a function to patch typing_extensions
def patch_typing_extensions():
    """Patch typing_extensions to work in the executable."""
    try:
        # Try to import typing_extensions
        import typing_extensions
        print("typing_extensions imported successfully")
    except ImportError:
        print("Failed to import typing_extensions")
        return
    
    # Check if metadata is available
    try:
        importlib.metadata.metadata('typing_extensions')
        print("typing_extensions metadata found")
    except importlib.metadata.PackageNotFoundError:
        print("typing_extensions metadata not found, applying patch")
        
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
