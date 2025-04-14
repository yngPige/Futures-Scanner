"""
PyInstaller hook for various dependencies.

This hook ensures that the metadata for various packages is properly included in the executable.
"""

from PyInstaller.utils.hooks import copy_metadata

# List of packages that need their metadata included
packages = [
    'typing_extensions',
    'pandas',
    'numpy',
    'scikit-learn',
    'matplotlib',
    'plotly',
    'joblib',
    'ccxt',
    'yfinance',
    'colorama',
    'requests',
    'huggingface_hub',
    'transformers',
    'keyboard',
    'importlib_metadata',
    'setuptools',
    'packaging',
    'pydantic',
]

# Copy metadata for all packages
datas = []
for package in packages:
    try:
        datas.extend(copy_metadata(package))
    except Exception:
        pass  # Skip if package metadata can't be found
