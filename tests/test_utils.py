"""
Tests for dupfinder.utils module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dupfinder.utils import (
    find_files_by_extension,
    format_duration,
    format_file_size,
    is_excluded_path,
)


class TestFormatFileSize:
    """Tests for format_file_size function."""

    def test_bytes(self) -> None:
        """Should format bytes correctly."""
        assert format_file_size(0) == "0.00 B"
        assert format_file_size(512) == "512.00 B"
        assert format_file_size(1023) == "1023.00 B"

    def test_kilobytes(self) -> None:
        """Should format kilobytes correctly."""
        assert format_file_size(1024) == "1.00 KB"
        assert format_file_size(1536) == "1.50 KB"
        assert format_file_size(10240) == "10.00 KB"

    def test_megabytes(self) -> None:
        """Should format megabytes correctly."""
        assert format_file_size(1024 * 1024) == "1.00 MB"
        assert format_file_size(5 * 1024 * 1024) == "5.00 MB"
        assert format_file_size(int(1.5 * 1024 * 1024)) == "1.50 MB"

    def test_gigabytes(self) -> None:
        """Should format gigabytes correctly."""
        assert format_file_size(1024 * 1024 * 1024) == "1.00 GB"
        assert format_file_size(int(2.5 * 1024 * 1024 * 1024)) == "2.50 GB"

    def test_terabytes(self) -> None:
        """Should format terabytes correctly."""
        assert format_file_size(1024 * 1024 * 1024 * 1024) == "1.00 TB"
        assert format_file_size(int(1.5 * 1024**4)) == "1.50 TB"


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_seconds_only(self) -> None:
        """Should format seconds only."""
        assert format_duration(0) == "0s"
        assert format_duration(30) == "30s"
        assert format_duration(59) == "59s"

    def test_minutes_and_seconds(self) -> None:
        """Should format minutes and seconds."""
        assert format_duration(60) == "1m 0s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(3599) == "59m 59s"

    def test_hours_minutes_seconds(self) -> None:
        """Should format hours, minutes, and seconds."""
        assert format_duration(3600) == "1h 0m 0s"
        assert format_duration(3661) == "1h 1m 1s"
        assert format_duration(7325) == "2h 2m 5s"

    def test_float_seconds(self) -> None:
        """Should truncate fractional seconds."""
        assert format_duration(59.9) == "59s"
        assert format_duration(90.5) == "1m 30s"


class TestIsExcludedPath:
    """Tests for is_excluded_path function."""

    def test_not_excluded(self) -> None:
        """Normal paths should not be excluded."""
        assert not is_excluded_path("/home/user/photos/image.jpg")
        assert not is_excluded_path("/var/data/videos/movie.mp4")
        assert not is_excluded_path("C:\\Users\\Name\\Pictures\\photo.png")

    def test_recycle_bin_excluded(self) -> None:
        """$RECYCLE.BIN should be excluded."""
        assert is_excluded_path("/mnt/c/$RECYCLE.BIN/image.jpg")
        assert is_excluded_path("D:\\$RECYCLE.BIN\\file.png")

    def test_trash_excluded(self) -> None:
        """Trash folders should be excluded."""
        assert is_excluded_path("/home/user/.Trash/photo.jpg")
        assert is_excluded_path("/volume/.Trashes/file.png")

    def test_git_excluded(self) -> None:
        """Git folders should be excluded."""
        assert is_excluded_path("/project/.git/objects/image.png")

    def test_pycache_excluded(self) -> None:
        """__pycache__ should be excluded."""
        assert is_excluded_path("/project/__pycache__/module.pyc")

    def test_node_modules_excluded(self) -> None:
        """node_modules should be excluded."""
        assert is_excluded_path("/project/node_modules/package/image.png")

    def test_venv_excluded(self) -> None:
        """Virtual environments should be excluded."""
        assert is_excluded_path("/project/.venv/lib/image.png")
        assert is_excluded_path("/project/venv/lib/image.png")

    def test_case_insensitive(self) -> None:
        """Matching should be case-insensitive."""
        assert is_excluded_path("/mnt/$Recycle.Bin/file.jpg")
        assert is_excluded_path("/home/.TRASH/file.jpg")

    def test_custom_excluded_folders(self) -> None:
        """Should work with custom excluded folders."""
        custom = {"custom_folder", "another_folder"}
        assert is_excluded_path("/path/custom_folder/file.jpg", custom)
        assert is_excluded_path("/path/another_folder/file.jpg", custom)
        assert not is_excluded_path("/path/$RECYCLE.BIN/file.jpg", custom)


class TestFindFilesByExtension:
    """Tests for find_files_by_extension function."""

    def test_find_images(self, nested_folder_structure: Path) -> None:
        """Should find image files recursively."""
        files = find_files_by_extension(
            str(nested_folder_structure),
            {".jpg", ".jpeg", ".png"},
        )
        # Should find images in root, photos, and photos/vacation
        # But NOT in $RECYCLE.BIN or .cache
        assert len(files) == 3

    def test_find_with_case_variations(self, temp_dir: Path) -> None:
        """Should find files with both lower and upper case extensions."""
        from PIL import Image
        import numpy as np

        # Create files with different case extensions
        img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        img.save(temp_dir / "lower.jpg")
        img.save(temp_dir / "upper.JPG")

        files = find_files_by_extension(str(temp_dir), {".jpg"})
        # Should find both lower and upper case
        assert len(files) == 2

    def test_returns_sorted_list(self, temp_dir: Path) -> None:
        """Should return sorted list of paths."""
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        img.save(temp_dir / "c_image.jpg")
        img.save(temp_dir / "a_image.jpg")
        img.save(temp_dir / "b_image.jpg")

        files = find_files_by_extension(str(temp_dir), {".jpg"})
        assert files == sorted(files)

    def test_excludes_system_folders(self, nested_folder_structure: Path) -> None:
        """Should exclude images in system folders."""
        files = find_files_by_extension(
            str(nested_folder_structure),
            {".jpg"},
        )
        # Verify no files from excluded folders
        for f in files:
            assert "$recycle" not in f.lower()
            assert ".cache" not in f.lower()

    def test_empty_folder(self, temp_dir: Path) -> None:
        """Should return empty list for folder with no matching files."""
        files = find_files_by_extension(str(temp_dir), {".jpg"})
        assert files == []

    def test_no_matching_extensions(self, sample_image: Path) -> None:
        """Should return empty list when no files match extensions."""
        folder = sample_image.parent
        files = find_files_by_extension(str(folder), {".xyz", ".abc"})
        assert files == []
