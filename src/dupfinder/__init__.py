"""
dupfinder - Find duplicate images and videos based on content similarity.

This package provides tools for detecting duplicate media files using
perceptual hashing, with optional GPU acceleration.

Example usage as a library:

    from dupfinder import ImageDuplicateFinder, VideoDuplicateFinder

    # Find duplicate images
    finder = ImageDuplicateFinder(similarity_threshold=0.90)
    image_files = finder.find_images("/path/to/images")
    finder.process_images(image_files)
    duplicates = finder.find_duplicates()

    # Find duplicate videos
    finder = VideoDuplicateFinder(similarity_threshold=0.85)
    video_files = finder.find_videos("/path/to/videos")
    finder.process_videos(video_files)
    duplicates = finder.find_duplicates()

Command-line usage:

    # Find duplicate images
    find-duplicate-images /path/to/images

    # Find duplicate videos
    find-duplicate-videos /path/to/videos

    # Unified CLI
    dupfinder images /path/to/images
    dupfinder videos /path/to/videos
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from dupfinder.accelerator import (
    AcceleratorType,
    GPUAccelerator,
    get_accelerator,
)
from dupfinder.constants import (
    EXCLUDED_FOLDERS,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from dupfinder.images import (
    ImageDuplicateFinder,
    ImageHasher,
    ImageInfo,
)
from dupfinder.videos import (
    VideoDuplicateFinder,
    VideoHasher,
    VideoInfo,
)

__all__ = [
    # Version
    "__version__",
    # Image module
    "ImageDuplicateFinder",
    "ImageHasher",
    "ImageInfo",
    # Video module
    "VideoDuplicateFinder",
    "VideoHasher",
    "VideoInfo",
    # Accelerator
    "GPUAccelerator",
    "AcceleratorType",
    "get_accelerator",
    # Constants
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "EXCLUDED_FOLDERS",
]
