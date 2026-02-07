"""
Shared utility functions for dupfinder.
"""

import logging
from pathlib import Path

from dupfinder.constants import EXCLUDED_FOLDERS

logger = logging.getLogger(__name__)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def is_excluded_path(file_path: str, excluded_folders: set[str] = EXCLUDED_FOLDERS) -> bool:
    """Check if a file path contains any excluded folder."""
    path_parts = Path(file_path).parts
    return any(excluded in part.lower() for part in path_parts for excluded in excluded_folders)


def find_files_by_extension(
    folder: str,
    extensions: set[str],
    excluded_folders: set[str] = EXCLUDED_FOLDERS,
) -> list[str]:
    """
    Find all files with given extensions in a folder recursively.

    Args:
        folder: Root folder to search
        extensions: Set of file extensions to match (including dot, e.g., ".jpg")
        excluded_folders: Set of folder names to exclude

    Returns:
        Sorted list of file paths
    """
    files = []
    folder_path = Path(folder)

    logger.info(f"Scanning for files in: {folder}")

    for ext in extensions:
        files.extend(folder_path.rglob(f"*{ext}"))
        files.extend(folder_path.rglob(f"*{ext.upper()}"))

    # Remove duplicates and convert to strings
    files = list({str(p) for p in files})

    # Filter out files in excluded folders
    files = [f for f in files if not is_excluded_path(f, excluded_folders)]

    files.sort()

    logger.info(f"Found {len(files)} files")
    return files


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
