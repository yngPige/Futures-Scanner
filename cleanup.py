#!/usr/bin/env python
"""
Cleanup script for Crypto Futures Scanner

This script cleans up result files, temporary files, and other generated content
to free up disk space and prepare the project for version control.
"""

import os
import shutil
import glob
import tempfile
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Clean up result files and temporary files.')
    parser.add_argument('--all', action='store_true', help='Clean up all files (data, models, results, reports, logs)')
    parser.add_argument('--data', action='store_true', help='Clean up data files')
    parser.add_argument('--models', action='store_true', help='Clean up model files')
    parser.add_argument('--results', action='store_true', help='Clean up result files')
    parser.add_argument('--reports', action='store_true', help='Clean up report files')
    parser.add_argument('--logs', action='store_true', help='Clean up log files')
    parser.add_argument('--temp', action='store_true', help='Clean up temporary files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without actually deleting')
    
    return parser.parse_args()

def clean_directory(directory, dry_run=False):
    """Clean up a directory by removing all files and subdirectories."""
    if not os.path.exists(directory):
        logger.info(f"Directory {directory} does not exist, skipping.")
        return
    
    logger.info(f"Cleaning directory: {directory}")
    
    if dry_run:
        for root, dirs, files in os.walk(directory):
            for file in files:
                logger.info(f"Would delete: {os.path.join(root, file)}")
        return
    
    try:
        # Remove all files in the directory
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
                logger.info(f"Deleted file: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                logger.info(f"Deleted directory: {item_path}")
    except Exception as e:
        logger.error(f"Error cleaning directory {directory}: {e}")

def clean_files(pattern, dry_run=False):
    """Clean up files matching a pattern."""
    files = glob.glob(pattern)
    
    if not files:
        logger.info(f"No files matching pattern {pattern}, skipping.")
        return
    
    logger.info(f"Cleaning files matching pattern: {pattern}")
    
    if dry_run:
        for file in files:
            logger.info(f"Would delete: {file}")
        return
    
    try:
        for file in files:
            os.remove(file)
            logger.info(f"Deleted file: {file}")
    except Exception as e:
        logger.error(f"Error cleaning files matching {pattern}: {e}")

def clean_temp_files(dry_run=False):
    """Clean up temporary files in the system temp directory."""
    temp_dir = tempfile.gettempdir()
    patterns = [
        os.path.join(temp_dir, "*_analysis_*.bat"),
        os.path.join(temp_dir, "*_analysis_*.sh"),
        os.path.join(temp_dir, "*_prediction_*.bat"),
        os.path.join(temp_dir, "*_prediction_*.sh"),
        os.path.join(temp_dir, "*_backtest_*.bat"),
        os.path.join(temp_dir, "*_backtest_*.sh"),
        os.path.join(temp_dir, "*_all_*.bat"),
        os.path.join(temp_dir, "*_all_*.sh")
    ]
    
    logger.info("Cleaning temporary files in system temp directory")
    
    for pattern in patterns:
        clean_files(pattern, dry_run)

def main():
    """Main function."""
    args = parse_args()
    
    # If no specific option is provided, show help
    if not any([args.all, args.data, args.models, args.results, args.reports, args.logs, args.temp]):
        logger.info("No cleanup option specified. Use --help to see available options.")
        return
    
    # Clean up data files
    if args.all or args.data:
        clean_directory("data", args.dry_run)
    
    # Clean up model files
    if args.all or args.models:
        clean_directory("models", args.dry_run)
    
    # Clean up result files
    if args.all or args.results:
        clean_directory("results", args.dry_run)
    
    # Clean up report files
    if args.all or args.reports:
        clean_directory("reports", args.dry_run)
    
    # Clean up log files
    if args.all or args.logs:
        clean_files("*.log", args.dry_run)
    
    # Clean up temporary files
    if args.all or args.temp:
        clean_temp_files(args.dry_run)
    
    if args.dry_run:
        logger.info("Dry run completed. No files were actually deleted.")
    else:
        logger.info("Cleanup completed successfully.")

if __name__ == "__main__":
    main()
