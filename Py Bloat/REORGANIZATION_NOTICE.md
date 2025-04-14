# File Structure Reorganization Notice

The 3lacks Scanner project has undergone a file structure reorganization to improve maintainability and organization. Here's what you need to know:

## Changes Made

1. **Utility Scripts**: All utility scripts have been moved to the `scripts/` directory and organized by category:
   - Download scripts: `scripts/download/`
   - Fix model scripts: `scripts/fix_model/`
   - Create scripts: `scripts/create/`
   - Build scripts: `scripts/build/`

2. **Documentation**: Documentation files have been moved to the `docs/` directory.

3. **Setup Scripts**: Installation and setup scripts have been moved to the `setup/` directory.

4. **Assets**: Asset files (icons, images) have been moved to the `assets/` directory.

5. **Monkey Patch**: The `monkey_patch.py` file has been moved to `src/utils/`.

## Running Scripts

To run utility scripts, use the new `run_script.py` utility:

```bash
# List all available scripts
python run_script.py --list

# Run a specific script
python run_script.py --category download --script download_llm_model.py
```

## File Structure Documentation

For a detailed explanation of the new file structure, see the [File Structure Document](docs/FILE_STRUCTURE.md).

## Transition Period

The original script files in the root directory will be removed in a future update. Please update any references to these files to use the new locations.

If you have any questions or issues with the new organization, please open an issue on GitHub.
