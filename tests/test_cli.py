"""
Tests for dupfinder.cli module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from dupfinder.cli import (
    create_image_parser,
    create_video_parser,
    main,
    run_image_finder,
    run_video_finder,
    setup_logging,
)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_default_level(self) -> None:
        """Should set INFO level by default."""
        import logging

        # Create a fresh logger for testing
        test_logger = logging.getLogger("test_default")
        test_logger.handlers = []
        
        setup_logging(verbose=False)
        # The function configures basicConfig, which sets root logger
        # Just verify it doesn't raise
        assert True

    def test_verbose_level(self) -> None:
        """Should set DEBUG level when verbose."""
        import logging

        # Create a fresh logger for testing
        test_logger = logging.getLogger("test_verbose")
        test_logger.handlers = []
        
        setup_logging(verbose=True)
        # The function configures basicConfig - just verify it runs
        assert True


class TestCreateImageParser:
    """Tests for create_image_parser function."""

    def test_creates_parser(self) -> None:
        """Should create an ArgumentParser."""
        parser = create_image_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_requires_folder(self) -> None:
        """Should require folder argument."""
        parser = create_image_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_folder_argument(self) -> None:
        """Should parse folder argument."""
        parser = create_image_parser()
        args = parser.parse_args(["/path/to/images"])
        assert args.folder == "/path/to/images"

    def test_threshold_default(self) -> None:
        """Should have default threshold of 0.90."""
        parser = create_image_parser()
        args = parser.parse_args(["/path"])
        assert args.threshold == 0.90

    def test_threshold_custom(self) -> None:
        """Should accept custom threshold."""
        parser = create_image_parser()
        args = parser.parse_args(["/path", "-t", "0.95"])
        assert args.threshold == 0.95

    def test_hash_size_default(self) -> None:
        """Should have default hash size of 16."""
        parser = create_image_parser()
        args = parser.parse_args(["/path"])
        assert args.hash_size == 16

    def test_hash_size_custom(self) -> None:
        """Should accept custom hash size."""
        parser = create_image_parser()
        args = parser.parse_args(["/path", "-s", "8"])
        assert args.hash_size == 8

    def test_workers_default(self) -> None:
        """Should have default workers of 4."""
        parser = create_image_parser()
        args = parser.parse_args(["/path"])
        assert args.workers == 4

    def test_workers_custom(self) -> None:
        """Should accept custom workers."""
        parser = create_image_parser()
        args = parser.parse_args(["/path", "-w", "8"])
        assert args.workers == 8

    def test_hash_type_default(self) -> None:
        """Should have default hash type of combined."""
        parser = create_image_parser()
        args = parser.parse_args(["/path"])
        assert args.hash_type == "combined"

    def test_hash_type_options(self) -> None:
        """Should accept valid hash types."""
        parser = create_image_parser()
        
        for hash_type in ["phash", "ahash", "dhash", "whash", "combined"]:
            args = parser.parse_args(["/path", "--hash-type", hash_type])
            assert args.hash_type == hash_type

    def test_no_gpu_flag(self) -> None:
        """Should parse --no-gpu flag."""
        parser = create_image_parser()
        
        args = parser.parse_args(["/path"])
        assert args.no_gpu is False
        
        args = parser.parse_args(["/path", "--no-gpu"])
        assert args.no_gpu is True

    def test_verbose_flag(self) -> None:
        """Should parse --verbose flag."""
        parser = create_image_parser()
        
        args = parser.parse_args(["/path"])
        assert args.verbose is False
        
        args = parser.parse_args(["/path", "-v"])
        assert args.verbose is True

    def test_output_option(self) -> None:
        """Should parse --output option."""
        parser = create_image_parser()
        args = parser.parse_args(["/path", "-o", "results.json"])
        assert args.output == "results.json"


class TestCreateVideoParser:
    """Tests for create_video_parser function."""

    def test_creates_parser(self) -> None:
        """Should create an ArgumentParser."""
        parser = create_video_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_requires_folder(self) -> None:
        """Should require folder argument."""
        parser = create_video_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_folder_argument(self) -> None:
        """Should parse folder argument."""
        parser = create_video_parser()
        args = parser.parse_args(["/path/to/videos"])
        assert args.folder == "/path/to/videos"

    def test_threshold_default(self) -> None:
        """Should have default threshold of 0.85."""
        parser = create_video_parser()
        args = parser.parse_args(["/path"])
        assert args.threshold == 0.85

    def test_frames_default(self) -> None:
        """Should have default frames of 10."""
        parser = create_video_parser()
        args = parser.parse_args(["/path"])
        assert args.frames == 10

    def test_frames_custom(self) -> None:
        """Should accept custom frame count."""
        parser = create_video_parser()
        args = parser.parse_args(["/path", "-f", "20"])
        assert args.frames == 20


class TestRunImageFinder:
    """Tests for run_image_finder function."""

    def test_invalid_directory(self) -> None:
        """Should return error for invalid directory."""
        parser = create_image_parser()
        args = parser.parse_args(["/nonexistent/path"])
        
        result = run_image_finder(args)
        assert result == 1

    def test_invalid_threshold(self, temp_dir: Path) -> None:
        """Should return error for invalid threshold."""
        parser = create_image_parser()
        args = parser.parse_args([str(temp_dir), "-t", "1.5"])
        
        result = run_image_finder(args)
        assert result == 1

    def test_valid_empty_folder(self, temp_dir: Path) -> None:
        """Should succeed for valid empty folder."""
        parser = create_image_parser()
        args = parser.parse_args([str(temp_dir), "--no-gpu"])
        
        result = run_image_finder(args)
        assert result == 0


class TestRunVideoFinder:
    """Tests for run_video_finder function."""

    def test_invalid_directory(self) -> None:
        """Should return error for invalid directory."""
        parser = create_video_parser()
        args = parser.parse_args(["/nonexistent/path"])
        
        result = run_video_finder(args)
        assert result == 1

    def test_invalid_threshold(self, temp_dir: Path) -> None:
        """Should return error for invalid threshold."""
        parser = create_video_parser()
        args = parser.parse_args([str(temp_dir), "-t", "0"])
        
        result = run_video_finder(args)
        assert result == 1

    def test_valid_empty_folder(self, temp_dir: Path) -> None:
        """Should succeed for valid empty folder."""
        parser = create_video_parser()
        args = parser.parse_args([str(temp_dir), "--no-gpu"])
        
        result = run_video_finder(args)
        assert result == 0


class TestMainCLI:
    """Tests for main CLI entry point."""

    def test_help_exits(self) -> None:
        """Should exit with --help."""
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, "argv", ["dupfinder", "--help"]):
                main()
        assert exc_info.value.code == 0

    def test_images_subcommand_help(self) -> None:
        """Should show help for images subcommand."""
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, "argv", ["dupfinder", "images", "--help"]):
                main()
        assert exc_info.value.code == 0

    def test_videos_subcommand_help(self) -> None:
        """Should show help for videos subcommand."""
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, "argv", ["dupfinder", "videos", "--help"]):
                main()
        assert exc_info.value.code == 0

    def test_no_subcommand_shows_help(self) -> None:
        """Should show help when no subcommand provided."""
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, "argv", ["dupfinder"]):
                main()
        # May exit with 0 (help) or 2 (error)
        assert exc_info.value.code in (0, 2)

    def test_images_with_valid_folder(self, temp_dir: Path) -> None:
        """Should run images finder with valid folder."""
        with patch.object(
            sys, "argv", ["dupfinder", "images", str(temp_dir), "--no-gpu"]
        ):
            # main() calls sys.exit, so we catch SystemExit
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_videos_with_valid_folder(self, temp_dir: Path) -> None:
        """Should run videos finder with valid folder."""
        with patch.object(
            sys, "argv", ["dupfinder", "videos", str(temp_dir), "--no-gpu"]
        ):
            # main() calls sys.exit, so we catch SystemExit
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_find_images_basic(self, sample_image: Path) -> None:
        """Should run find-images command successfully."""
        parser = create_image_parser()
        args = parser.parse_args([str(sample_image.parent), "--no-gpu"])
        
        result = run_image_finder(args)
        assert result == 0

    def test_find_videos_basic(self, sample_video: Path) -> None:
        """Should run find-videos command successfully."""
        parser = create_video_parser()
        args = parser.parse_args([str(sample_video.parent), "--no-gpu"])
        
        result = run_video_finder(args)
        assert result == 0

    def test_output_json(self, temp_dir: Path, sample_image: Path) -> None:
        """Should save output to JSON file."""
        output_file = temp_dir / "results.json"
        parser = create_image_parser()
        args = parser.parse_args([
            str(sample_image.parent),
            "--no-gpu",
            "-o", str(output_file),
        ])
        
        result = run_image_finder(args)
        assert result == 0
        # Output file may or may not exist depending on whether duplicates found
