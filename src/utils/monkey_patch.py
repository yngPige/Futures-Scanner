"""
Monkey patch for numpy to add NaN as an alias for nan.
This is needed for compatibility with pandas-ta.
"""

import numpy as np
import sys

def apply_patches():
    """Apply all monkey patches."""
    # Add NaN as an alias for nan in numpy
    np.NaN = np.nan

    # Make sure the patch is applied to the imported numpy module
    sys.modules['numpy'].NaN = np.nan

    print("Monkey patch applied: numpy.NaN is now available as an alias for numpy.nan")

# Apply patches when the module is imported
apply_patches()
