"""
PyInstaller hook for typing_extensions package.

This hook ensures that the metadata for typing_extensions is properly included in the executable.
"""

from PyInstaller.utils.hooks import copy_metadata

# Copy the metadata for typing_extensions
datas = copy_metadata('typing_extensions')
