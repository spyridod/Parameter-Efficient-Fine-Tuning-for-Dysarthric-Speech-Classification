"""
Helper functions for file operations and directory management
"""

import os
import shutil
from pathlib import Path


def create_output_dir(output_base_dir, iteration, test_speakers):
    """
    Create output directory for current experiment

    Args:
        output_base_dir: Base directory for all outputs
        iteration: Current iteration number
        test_speakers: List of test speaker IDs

    Returns:
        Path to created directory
    """
    test_speakers_str = "_".join(sorted(test_speakers))
    dir_name = f"{output_base_dir}/iteration_{iteration}_speakers_{test_speakers_str}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def clear_cache(cache_dir):
    """
    Clear cache directory

    Args:
        cache_dir: Directory to clear
    """
    for filename in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")