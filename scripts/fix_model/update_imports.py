"""
Helper script to update imports in fix_model scripts.
"""

import os
import re

def update_imports():
    """Update imports in all fix_model scripts."""
    script_files = [f for f in os.listdir('.') if f.endswith('.py') and f != '__init__.py' and f != 'update_imports.py']
    
    for filename in script_files:
        with open(filename, 'r') as file:
            content = file.read()
        
        # Add sys.path.append if not already present
        if 'sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))' not in content:
            updated_content = re.sub(
                r'import sys\nimport os',
                r'import sys\nimport os\n\n# Add the project root directory to sys.path\nsys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))',
                content
            )
            
            # If the pattern wasn't found, try another common pattern
            if updated_content == content:
                updated_content = re.sub(
                    r'import os\nimport sys',
                    r'import os\nimport sys\n\n# Add the project root directory to sys.path\nsys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))',
                    content
                )
            
            # If still not found, add it at the top after the docstring
            if updated_content == content:
                docstring_end = content.find('"""', content.find('"""') + 3) + 3
                if docstring_end > 3:
                    updated_content = content[:docstring_end] + '\nimport os\nimport sys\n\n# Add the project root directory to sys.path\nsys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))\n' + content[docstring_end:]
                else:
                    updated_content = 'import os\nimport sys\n\n# Add the project root directory to sys.path\nsys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))\n' + content
        else:
            updated_content = content
        
        with open(filename, 'w') as file:
            file.write(updated_content)
        
        print(f"Updated imports in {filename}")

if __name__ == "__main__":
    update_imports()
