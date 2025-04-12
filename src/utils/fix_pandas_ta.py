"""
Fix for pandas-ta compatibility issues with newer numpy versions.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_squeeze_pro():
    """
    Fix the squeeze_pro.py file in pandas-ta to work with newer numpy versions.
    """
    try:
        # Find the pandas_ta package location
        import pandas_ta
        package_path = os.path.dirname(pandas_ta.__file__)
        squeeze_pro_path = os.path.join(package_path, 'momentum', 'squeeze_pro.py')
        
        if not os.path.exists(squeeze_pro_path):
            logger.error(f"Could not find squeeze_pro.py at {squeeze_pro_path}")
            return False
        
        # Read the file
        with open(squeeze_pro_path, 'r') as f:
            content = f.read()
        
        # Check if the file already contains the fix
        if 'from numpy import nan as npNaN' in content:
            logger.info("squeeze_pro.py already fixed")
            return True
        
        # Replace the problematic import
        content = content.replace(
            'from numpy import NaN as npNaN',
            'from numpy import nan as npNaN'
        )
        
        # Write the fixed file
        with open(squeeze_pro_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully fixed {squeeze_pro_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing squeeze_pro.py: {e}")
        return False

if __name__ == "__main__":
    fix_squeeze_pro()
