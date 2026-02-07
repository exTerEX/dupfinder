"""
Command-line interface for dupfinder.

Provides CLI commands for finding duplicate images and videos.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dupfinder.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_HASH_SIZE,
    DEFAULT_IMAGE_THRESHOLD,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_WORKERS,
    DEFAULT_VIDEO_THRESHOLD,
)
from dupfinder.images import ImageDuplicateFinder
from dupfinder.images import print_results as print_image_results
from dupfinder.videos import VideoDuplicateFinder
from dupfinder.videos import print_results as print_video_results


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def create_image_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Create argument parser for image duplicate finder."""
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Find duplicate images based on content similarity",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s /path/to/images
  %(prog)s /path/to/images --threshold 0.95
  %(prog)s /path/to/images --hash-type phash
  %(prog)s /path/to/images --output duplicates.json
            """,
        )

    parser.add_argument(
        "folder",
        help="Folder to scan for images (searches recursively)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=DEFAULT_IMAGE_THRESHOLD,
        help=f"Similarity threshold (0-1) to consider as duplicates (default: {DEFAULT_IMAGE_THRESHOLD})",
    )
    parser.add_argument(
        "-s",
        "--hash-size",
        type=int,
        default=DEFAULT_HASH_SIZE,
        help=f"Hash size for perceptual hashing (default: {DEFAULT_HASH_SIZE})",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--hash-type",
        choices=["phash", "ahash", "dhash", "whash", "combined"],
        default="combined",
        help="Hash algorithm to use (default: combined)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (use CPU only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for GPU processing (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def create_video_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Create argument parser for video duplicate finder."""
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Find duplicate videos based on content similarity",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s /path/to/videos
  %(prog)s /path/to/videos --threshold 0.9
  %(prog)s /path/to/videos --frames 15 --workers 8
  %(prog)s /path/to/videos --output duplicates.json
            """,
        )

    parser.add_argument(
        "folder",
        help="Folder to scan for videos (searches recursively)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=DEFAULT_VIDEO_THRESHOLD,
        help=f"Similarity threshold (0-1) to consider as duplicates (default: {DEFAULT_VIDEO_THRESHOLD})",
    )
    parser.add_argument(
        "-f",
        "--frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=f"Number of frames to extract per video (default: {DEFAULT_NUM_FRAMES})",
    )
    parser.add_argument(
        "-s",
        "--hash-size",
        type=int,
        default=DEFAULT_HASH_SIZE,
        help=f"Hash size for perceptual hashing (default: {DEFAULT_HASH_SIZE})",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (use CPU only)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def run_image_finder(args: argparse.Namespace) -> int:
    """Run the image duplicate finder with parsed arguments."""
    setup_logging(args.verbose)

    if not os.path.isdir(args.folder):
        print(f"Error: '{args.folder}' is not a valid directory")
        return 1

    if not 0 < args.threshold <= 1:
        print("Error: Threshold must be between 0 and 1")
        return 1

    # Initialize finder
    finder = ImageDuplicateFinder(
        similarity_threshold=args.threshold,
        hash_size=args.hash_size,
        num_workers=args.workers,
        hash_type=args.hash_type,
        use_gpu=not args.no_gpu,
        batch_size=args.batch_size,
    )

    # Find and process images
    image_files = finder.find_images(args.folder)

    if not image_files:
        print("No image files found in the specified folder.")
        return 0

    finder.process_images(image_files)

    if not finder.image_infos:
        print("Could not process any image files.")
        return 1

    # Find duplicates
    duplicate_groups = finder.find_duplicates()

    # Print results
    print_image_results(duplicate_groups, finder.image_infos, args.output)

    return 0


def run_video_finder(args: argparse.Namespace) -> int:
    """Run the video duplicate finder with parsed arguments."""
    setup_logging(args.verbose)

    if not os.path.isdir(args.folder):
        print(f"Error: '{args.folder}' is not a valid directory")
        return 1

    if not 0 < args.threshold <= 1:
        print("Error: Threshold must be between 0 and 1")
        return 1

    # Initialize finder
    finder = VideoDuplicateFinder(
        similarity_threshold=args.threshold,
        num_frames=args.frames,
        hash_size=args.hash_size,
        num_workers=args.workers,
        use_gpu=not args.no_gpu,
    )

    # Find and process videos
    video_files = finder.find_videos(args.folder)

    if not video_files:
        print("No video files found in the specified folder.")
        return 0

    finder.process_videos(video_files)

    if not finder.video_infos:
        print("Could not process any video files.")
        return 1

    # Find duplicates
    duplicate_groups = finder.find_duplicates()

    # Print results
    print_video_results(duplicate_groups, finder.video_infos, args.output)

    return 0


def find_images_main() -> None:
    """Entry point for find-duplicate-images command."""
    parser = create_image_parser()
    args = parser.parse_args()
    sys.exit(run_image_finder(args))


def find_videos_main() -> None:
    """Entry point for find-duplicate-videos command."""
    parser = create_video_parser()
    args = parser.parse_args()
    sys.exit(run_video_finder(args))


def main() -> None:
    """
    Main entry point for the unified dupfinder command.

    Usage:
        dupfinder images /path/to/images [options]
        dupfinder videos /path/to/videos [options]
    """
    parser = argparse.ArgumentParser(
        prog="dupfinder",
        description="Find duplicate images and videos based on content similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  images    Find duplicate images
  videos    Find duplicate videos

Examples:
  dupfinder images /path/to/images
  dupfinder videos /path/to/videos --threshold 0.9
  dupfinder images /path/to/images --output duplicates.json

For command-specific help:
  dupfinder images --help
  dupfinder videos --help
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Images subcommand
    images_parser = subparsers.add_parser(
        "images",
        help="Find duplicate images",
        description="Find duplicate images based on content similarity",
    )
    create_image_parser(images_parser)

    # Videos subcommand
    videos_parser = subparsers.add_parser(
        "videos",
        help="Find duplicate videos",
        description="Find duplicate videos based on content similarity",
    )
    create_video_parser(videos_parser)

    # Parse args
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)
    elif args.command == "images":
        sys.exit(run_image_finder(args))
    elif args.command == "videos":
        sys.exit(run_video_finder(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
