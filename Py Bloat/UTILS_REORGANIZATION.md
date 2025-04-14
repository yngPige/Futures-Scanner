# Utility Functions Reorganization

The 3lacks Scanner project has undergone a reorganization of utility functions to improve maintainability and organization. Here's what you need to know:

## Changes Made

1. **Utility Functions**: All utility functions have been moved to the `utils/` directory and organized by category:
   - Data utilities: `utils/data/`
   - Model utilities: `utils/models/`
   - Build utilities: `utils/build/`
   - Setup utilities: `utils/setup/`
   - Visualization utilities: `utils/visualization/`

2. **Documentation**: Documentation files have been moved to `utils/docs/`.

## Running Utilities

To run utility functions, use the new `run_utils.py` utility:

```bash
# List all available utilities
python run_utils.py --list

# Run a specific utility
python run_utils.py --category data --module download_llm_model.py
```

## Utility Documentation

For a detailed explanation of the utility functions, see the [utils/README.md](utils/README.md) file.

## Transition Period

The original utility files in the root directory and scripts directory will be removed in a future update. Please update any references to these files to use the new locations.

If you have any questions or issues with the new organization, please open an issue on GitHub.
