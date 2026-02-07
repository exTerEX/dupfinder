"""
Pytest fixtures and configuration for dupfinder tests.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def test_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data that persists for the session."""
    temp_dir = Path(tempfile.mkdtemp(prefix="dupfinder_test_"))
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for individual test."""
    temp_dir = Path(tempfile.mkdtemp(prefix="dupfinder_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """Create a sample RGB image."""
    img_path = temp_dir / "sample.jpg"
    # Create a simple gradient image
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            img_array[i, j] = [i * 2, j * 2, (i + j)]
    img = Image.fromarray(img_array)
    img.save(img_path, "JPEG")
    return img_path


@pytest.fixture
def sample_grayscale_image(temp_dir: Path) -> Path:
    """Create a sample grayscale image."""
    img_path = temp_dir / "sample_gray.png"
    img_array = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            img_array[i, j] = (i + j) % 256
    img = Image.fromarray(img_array, mode="L")
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def duplicate_images(temp_dir: Path) -> tuple[Path, Path, Path]:
    """Create a set of duplicate images with different resolutions."""
    # Original image
    original_path = temp_dir / "original.jpg"
    img_array = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        for j in range(200):
            img_array[i, j] = [i % 256, j % 256, (i * j) % 256]
    original = Image.fromarray(img_array)
    original.save(original_path, "JPEG", quality=95)

    # Resized duplicate
    resized_path = temp_dir / "resized.jpg"
    resized = original.resize((100, 100), Image.Resampling.LANCZOS)
    resized.save(resized_path, "JPEG", quality=90)

    # Slightly modified duplicate
    modified_path = temp_dir / "modified.jpg"
    modified_array = img_array.copy()
    modified_array[10:20, 10:20] = [255, 255, 255]  # Small white patch
    modified = Image.fromarray(modified_array)
    modified.save(modified_path, "JPEG", quality=95)

    return original_path, resized_path, modified_path


@pytest.fixture
def different_images(temp_dir: Path) -> tuple[Path, Path]:
    """Create two distinctly different images."""
    # Red image
    red_path = temp_dir / "red.jpg"
    red_array = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)
    Image.fromarray(red_array).save(red_path, "JPEG")

    # Blue image
    blue_path = temp_dir / "blue.jpg"
    blue_array = np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8)
    Image.fromarray(blue_array).save(blue_path, "JPEG")

    return red_path, blue_path


@pytest.fixture
def sample_video(temp_dir: Path) -> Path:
    """Create a sample video file."""
    video_path = temp_dir / "sample.mp4"
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (100, 100))
    
    # Write 60 frames (2 seconds at 30fps)
    for i in range(60):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create changing gradient
        frame[:, :, 0] = (i * 4) % 256  # Blue channel
        frame[:, :, 1] = (i * 3) % 256  # Green channel
        frame[:, :, 2] = (i * 2) % 256  # Red channel
        writer.write(frame)
    
    writer.release()
    return video_path


@pytest.fixture
def duplicate_videos(temp_dir: Path) -> tuple[Path, Path]:
    """Create duplicate video files with different resolutions."""
    # Original video
    original_path = temp_dir / "original.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(original_path), fourcc, 30.0, (200, 200))
    
    frames = []
    for i in range(60):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(frame, (20 + i, 20 + i), (180 - i, 180 - i), (255, 128, 64), -1)
        frames.append(frame)
        writer.write(frame)
    writer.release()

    # Resized duplicate
    resized_path = temp_dir / "resized.mp4"
    writer = cv2.VideoWriter(str(resized_path), fourcc, 30.0, (100, 100))
    for frame in frames:
        resized_frame = cv2.resize(frame, (100, 100))
        writer.write(resized_frame)
    writer.release()

    return original_path, resized_path


@pytest.fixture
def nested_folder_structure(temp_dir: Path) -> Path:
    """Create a nested folder structure with images and excluded folders."""
    # Create subdirectories
    subdir1 = temp_dir / "photos"
    subdir2 = temp_dir / "photos" / "vacation"
    recycle = temp_dir / "$RECYCLE.BIN"
    cache = temp_dir / ".cache"
    
    subdir1.mkdir(parents=True)
    subdir2.mkdir(parents=True)
    recycle.mkdir(parents=True)
    cache.mkdir(parents=True)
    
    # Create images in various locations
    locations = [temp_dir, subdir1, subdir2, recycle, cache]
    for idx, loc in enumerate(locations):
        img_array = np.full((50, 50, 3), [idx * 50, idx * 40, idx * 30], dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(loc / f"image_{idx}.jpg", "JPEG")
    
    return temp_dir


@pytest.fixture
def image_folder_with_duplicates(temp_dir: Path) -> tuple[Path, int]:
    """Create a folder with duplicate and unique images."""
    # Create 3 duplicate pairs and 2 unique images
    
    # Duplicate pair 1
    img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(img1).save(temp_dir / "dup1_a.jpg", "JPEG")
    Image.fromarray(img1).resize((80, 80)).save(temp_dir / "dup1_b.jpg", "JPEG")
    
    # Duplicate pair 2
    img2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(img2).save(temp_dir / "dup2_a.jpg", "JPEG")
    Image.fromarray(img2).save(temp_dir / "dup2_b.png", "PNG")
    
    # Duplicate triple
    img3 = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
    Image.fromarray(img3).save(temp_dir / "dup3_a.jpg", "JPEG")
    Image.fromarray(img3).resize((100, 100)).save(temp_dir / "dup3_b.jpg", "JPEG")
    Image.fromarray(img3).resize((75, 75)).save(temp_dir / "dup3_c.jpg", "JPEG")
    
    # Unique images
    for i in range(2):
        unique = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(unique).save(temp_dir / f"unique_{i}.jpg", "JPEG")
    
    return temp_dir, 3  # 3 duplicate groups expected
