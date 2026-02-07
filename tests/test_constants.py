"""
Tests for dupfinder.constants module.
"""

import pytest

from dupfinder.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_HASH_SIZE,
    DEFAULT_IMAGE_THRESHOLD,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_WORKERS,
    DEFAULT_VIDEO_THRESHOLD,
    EXCLUDED_FOLDERS,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)


class TestImageExtensions:
    """Tests for IMAGE_EXTENSIONS constant."""

    def test_is_frozenset(self) -> None:
        """IMAGE_EXTENSIONS should be immutable."""
        assert isinstance(IMAGE_EXTENSIONS, frozenset)

    def test_common_formats_included(self) -> None:
        """Common image formats should be included."""
        common_formats = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        assert common_formats.issubset(IMAGE_EXTENSIONS)

    def test_raw_formats_included(self) -> None:
        """RAW camera formats should be included."""
        raw_formats = {".raw", ".cr2", ".nef", ".arw", ".dng"}
        assert raw_formats.issubset(IMAGE_EXTENSIONS)

    def test_extensions_are_lowercase(self) -> None:
        """All extensions should be lowercase."""
        for ext in IMAGE_EXTENSIONS:
            assert ext == ext.lower(), f"Extension {ext} is not lowercase"

    def test_extensions_start_with_dot(self) -> None:
        """All extensions should start with a dot."""
        for ext in IMAGE_EXTENSIONS:
            assert ext.startswith("."), f"Extension {ext} doesn't start with dot"


class TestVideoExtensions:
    """Tests for VIDEO_EXTENSIONS constant."""

    def test_is_frozenset(self) -> None:
        """VIDEO_EXTENSIONS should be immutable."""
        assert isinstance(VIDEO_EXTENSIONS, frozenset)

    def test_common_formats_included(self) -> None:
        """Common video formats should be included."""
        common_formats = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm"}
        assert common_formats.issubset(VIDEO_EXTENSIONS)

    def test_extensions_are_lowercase(self) -> None:
        """All extensions should be lowercase."""
        for ext in VIDEO_EXTENSIONS:
            assert ext == ext.lower(), f"Extension {ext} is not lowercase"

    def test_extensions_start_with_dot(self) -> None:
        """All extensions should start with a dot."""
        for ext in VIDEO_EXTENSIONS:
            assert ext.startswith("."), f"Extension {ext} doesn't start with dot"


class TestExcludedFolders:
    """Tests for EXCLUDED_FOLDERS constant."""

    def test_is_frozenset(self) -> None:
        """EXCLUDED_FOLDERS should be immutable."""
        assert isinstance(EXCLUDED_FOLDERS, frozenset)

    def test_windows_folders_included(self) -> None:
        """Windows system folders should be excluded."""
        windows_folders = {"$recycle.bin", "system volume information"}
        assert windows_folders.issubset(EXCLUDED_FOLDERS)

    def test_macos_folders_included(self) -> None:
        """macOS system folders should be excluded."""
        macos_folders = {".trash", ".trashes", ".ds_store"}
        assert macos_folders.issubset(EXCLUDED_FOLDERS)

    def test_linux_folders_included(self) -> None:
        """Linux system folders should be excluded."""
        linux_folders = {"lost+found", "trash"}
        assert linux_folders.issubset(EXCLUDED_FOLDERS)

    def test_dev_folders_included(self) -> None:
        """Development folders should be excluded."""
        dev_folders = {".git", "__pycache__", "node_modules", ".venv"}
        assert dev_folders.issubset(EXCLUDED_FOLDERS)

    def test_folders_are_lowercase(self) -> None:
        """All folder names should be lowercase for case-insensitive matching."""
        for folder in EXCLUDED_FOLDERS:
            assert folder == folder.lower(), f"Folder {folder} is not lowercase"


class TestDefaultValues:
    """Tests for default configuration values."""

    def test_image_threshold_valid_range(self) -> None:
        """DEFAULT_IMAGE_THRESHOLD should be between 0 and 1."""
        assert 0 < DEFAULT_IMAGE_THRESHOLD <= 1

    def test_video_threshold_valid_range(self) -> None:
        """DEFAULT_VIDEO_THRESHOLD should be between 0 and 1."""
        assert 0 < DEFAULT_VIDEO_THRESHOLD <= 1

    def test_hash_size_is_power_of_two(self) -> None:
        """DEFAULT_HASH_SIZE should be a power of 2 for efficiency."""
        assert DEFAULT_HASH_SIZE > 0
        assert (DEFAULT_HASH_SIZE & (DEFAULT_HASH_SIZE - 1)) == 0

    def test_num_frames_positive(self) -> None:
        """DEFAULT_NUM_FRAMES should be positive."""
        assert DEFAULT_NUM_FRAMES > 0

    def test_num_workers_positive(self) -> None:
        """DEFAULT_NUM_WORKERS should be positive."""
        assert DEFAULT_NUM_WORKERS > 0

    def test_batch_size_positive(self) -> None:
        """DEFAULT_BATCH_SIZE should be positive."""
        assert DEFAULT_BATCH_SIZE > 0

    def test_image_threshold_value(self) -> None:
        """DEFAULT_IMAGE_THRESHOLD should be 0.90."""
        assert DEFAULT_IMAGE_THRESHOLD == 0.90

    def test_video_threshold_value(self) -> None:
        """DEFAULT_VIDEO_THRESHOLD should be 0.85."""
        assert DEFAULT_VIDEO_THRESHOLD == 0.85
