"""
Video Duplicate Finder module.

Detects duplicate videos based on content similarity using perceptual hashing
of extracted frames.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import cv2
import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

from dupfinder.constants import (
    DEFAULT_HASH_SIZE,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_WORKERS,
    DEFAULT_VIDEO_THRESHOLD,
    EXCLUDED_FOLDERS,
    VIDEO_EXTENSIONS,
)
from dupfinder.utils import find_files_by_extension, format_duration, format_file_size

logger = logging.getLogger(__name__)

# Lazy import GPU accelerator
_gpu_available = None
_get_accelerator = None
_compute_similarity_matrix_gpu = None


def _init_gpu() -> bool:
    """Initialize GPU module lazily."""
    global _gpu_available, _get_accelerator, _compute_similarity_matrix_gpu
    if _gpu_available is None:
        try:
            from dupfinder.accelerator import (
                compute_similarity_matrix_gpu,
                get_accelerator,
            )

            _get_accelerator = get_accelerator
            _compute_similarity_matrix_gpu = compute_similarity_matrix_gpu
            _gpu_available = True
        except ImportError:
            _gpu_available = False
    return _gpu_available


@dataclass
class VideoInfo:
    """Stores information about a video file."""

    path: str
    duration: float = 0.0
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    file_size: int = 0
    frame_hashes: list[str] = field(default_factory=list)
    average_hash: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": self.path,
            "duration": self.duration,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "average_hash": self.average_hash,
        }


class VideoHasher:
    """Handles video frame extraction and perceptual hashing."""

    def __init__(
        self,
        num_frames: int = DEFAULT_NUM_FRAMES,
        hash_size: int = DEFAULT_HASH_SIZE,
    ) -> None:
        """
        Initialize the video hasher.

        Args:
            num_frames: Number of frames to extract for comparison
            hash_size: Size of the perceptual hash (larger = more precise)
        """
        self.num_frames = num_frames
        self.hash_size = hash_size

    def extract_frames(self, video_path: str) -> tuple[list[np.ndarray], VideoInfo]:
        """
        Extract frames from a video at regular intervals.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple of (list of frames, video info)
        """
        video_info = VideoInfo(path=video_path)
        frames = []

        try:
            video_info.file_size = os.path.getsize(video_path)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                return frames, video_info

            # Get video properties
            video_info.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            video_info.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if video_info.frame_count > 0 and video_info.fps > 0:
                video_info.duration = video_info.frame_count / video_info.fps

            if video_info.frame_count <= 0:
                logger.warning(f"Could not determine frame count for: {video_path}")
                cap.release()
                return frames, video_info

            # Calculate frame indices to extract (evenly distributed)
            # Skip first and last 5% to avoid intro/outro
            start_frame = int(video_info.frame_count * 0.05)
            end_frame = int(video_info.frame_count * 0.95)

            if end_frame <= start_frame:
                start_frame = 0
                end_frame = video_info.frame_count - 1

            frame_interval = max(1, (end_frame - start_frame) // (self.num_frames + 1))
            frame_indices = [start_frame + (i + 1) * frame_interval for i in range(self.num_frames)]

            for frame_idx in frame_indices:
                if frame_idx >= video_info.frame_count:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret and frame is not None:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

            cap.release()

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")

        return frames, video_info

    def compute_frame_hash(self, frame: np.ndarray) -> str:
        """
        Compute perceptual hash for a single frame.

        Args:
            frame: RGB frame as numpy array

        Returns:
            Hexadecimal hash string
        """
        try:
            # Convert to PIL Image
            img = Image.fromarray(frame)

            # Compute perceptual hash
            phash = imagehash.phash(img, hash_size=self.hash_size)

            return str(phash)
        except Exception as e:
            logger.error(f"Error computing hash: {e}")
            return ""

    def compute_video_hashes(self, video_path: str) -> VideoInfo:
        """
        Compute perceptual hashes for a video.

        Args:
            video_path: Path to the video file

        Returns:
            VideoInfo with computed hashes
        """
        frames, video_info = self.extract_frames(video_path)

        if not frames:
            return video_info

        # Compute hash for each frame
        for frame in frames:
            frame_hash = self.compute_frame_hash(frame)
            if frame_hash:
                video_info.frame_hashes.append(frame_hash)

        # Compute average hash from all frames
        if video_info.frame_hashes:
            # Combine frame hashes into a single representative hash
            combined = "".join(video_info.frame_hashes)
            video_info.average_hash = hashlib.md5(combined.encode()).hexdigest()

        return video_info


class VideoDuplicateFinder:
    """Finds duplicate videos based on perceptual hash similarity."""

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_VIDEO_THRESHOLD,
        num_frames: int = DEFAULT_NUM_FRAMES,
        hash_size: int = DEFAULT_HASH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        use_gpu: bool = True,
    ) -> None:
        """
        Initialize the duplicate finder.

        Args:
            similarity_threshold: Minimum similarity (0-1) to consider as duplicate
            num_frames: Number of frames to extract per video
            hash_size: Size of perceptual hash
            num_workers: Number of parallel workers for processing
            use_gpu: Whether to use GPU acceleration if available
        """
        self.similarity_threshold = similarity_threshold
        self.num_workers = num_workers
        self.hash_size = hash_size
        self.hasher = VideoHasher(num_frames=num_frames, hash_size=hash_size)
        self.video_infos: dict[str, VideoInfo] = {}

        # Initialize GPU if requested and available
        self.use_gpu = use_gpu and _init_gpu()
        self.accelerator = None
        if self.use_gpu and _get_accelerator is not None:
            try:
                self.accelerator = _get_accelerator()
                logger.info(f"GPU acceleration enabled: {self.accelerator.get_backend_name()}")
            except Exception as e:
                logger.warning(f"GPU acceleration not available: {e}")
                self.use_gpu = False

    def find_videos(self, folder: str) -> list[str]:
        """
        Find all video files in a folder recursively.

        Args:
            folder: Root folder to search

        Returns:
            List of video file paths
        """
        return find_files_by_extension(folder, VIDEO_EXTENSIONS, EXCLUDED_FOLDERS)

    def process_videos(self, video_files: list[str]) -> dict[str, VideoInfo]:
        """
        Process all videos and compute their hashes.

        Args:
            video_files: List of video file paths

        Returns:
            Dictionary mapping file path to VideoInfo
        """
        logger.info("Processing videos and computing hashes...")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self.hasher.compute_video_hashes, vf): vf for vf in video_files
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                video_path = futures[future]
                try:
                    video_info = future.result()
                    if video_info.frame_hashes:
                        self.video_infos[video_path] = video_info
                except Exception as e:
                    logger.error(f"Error processing {video_path}: {e}")

        logger.info(f"Successfully processed {len(self.video_infos)} videos")
        return self.video_infos

    def compute_similarity(self, info1: VideoInfo, info2: VideoInfo) -> float:
        """
        Compute similarity between two videos based on their frame hashes.

        Args:
            info1: First video info
            info2: Second video info

        Returns:
            Similarity score (0-1)
        """
        if not info1.frame_hashes or not info2.frame_hashes:
            return 0.0

        # Compare frame hashes using Hamming distance
        similarities = []

        # Compare each frame from video1 to find best match in video2
        for hash1 in info1.frame_hashes:
            best_sim = 0.0
            h1 = imagehash.hex_to_hash(hash1)

            for hash2 in info2.frame_hashes:
                h2 = imagehash.hex_to_hash(hash2)

                # Hamming distance
                distance = h1 - h2
                max_distance = len(h1.hash.flatten()) * len(h1.hash.flatten())
                similarity = 1 - (distance / max_distance)
                best_sim = max(best_sim, similarity)

            similarities.append(best_sim)

        # Return average of best matches
        return sum(similarities) / len(similarities) if similarities else 0.0

    def find_duplicates(self) -> list[list[tuple[str, float]]]:
        """
        Find groups of duplicate videos.

        Returns:
            List of duplicate groups, where each group is a list of
            (file_path, similarity_score) tuples
        """
        logger.info("Finding duplicate videos...")

        video_paths = list(self.video_infos.keys())
        n = len(video_paths)

        # Use GPU-accelerated comparison if available and worthwhile
        if self.use_gpu and n > 20:
            return self._find_duplicates_gpu(video_paths)

        return self._find_duplicates_cpu(video_paths)

    def _find_duplicates_gpu(self, video_paths: list[str]) -> list[list[tuple[str, float]]]:
        """Find duplicates using GPU-accelerated frame hash comparison."""
        n = len(video_paths)
        logger.info(f"Computing video similarities using GPU for {n} videos...")

        # For videos, we need to compare frame hashes
        # First, create a combined hash per video from all frame hashes
        combined_hashes = []
        valid_paths = []

        for path in video_paths:
            info = self.video_infos[path]
            if info.frame_hashes:
                # Use first frame hash as representative
                combined_hashes.append(info.frame_hashes[0])
                valid_paths.append(path)

        if len(combined_hashes) < 2:
            return []

        # Compute similarity matrix using GPU
        try:
            if _compute_similarity_matrix_gpu is None:
                raise RuntimeError("GPU not initialized")
            sim_matrix = _compute_similarity_matrix_gpu(combined_hashes, self.hash_size)
        except Exception as e:
            logger.warning(f"GPU computation failed, falling back to CPU: {e}")
            return self._find_duplicates_cpu(video_paths)

        # For videos with multiple frame hashes, refine similarity with full comparison
        # This is a two-pass approach: quick GPU filter, then detailed CPU verification
        duplicate_groups: list[list[tuple[str, float]]] = []
        assigned: set[int] = set()

        for i in range(len(valid_paths)):
            if i in assigned:
                continue

            # Find candidates above a lower threshold for GPU pre-filter
            pre_threshold = max(0.5, self.similarity_threshold - 0.2)
            candidate_indices = np.where(sim_matrix[i] >= pre_threshold)[0]

            if len(candidate_indices) <= 1:
                continue

            # Refine with full frame comparison
            current_group: list[tuple[str, float]] = [(valid_paths[i], 1.0)]

            for j in candidate_indices:
                if j <= i or j in assigned:
                    continue

                # Full frame-by-frame comparison
                info1 = self.video_infos[valid_paths[i]]
                info2 = self.video_infos[valid_paths[j]]
                similarity = self.compute_similarity(info1, info2)

                if similarity >= self.similarity_threshold:
                    current_group.append((valid_paths[j], similarity))
                    assigned.add(j)

            if len(current_group) > 1:
                assigned.add(i)
                duplicate_groups.append(current_group)

        logger.info(f"Found {len(duplicate_groups)} groups of duplicates")
        return duplicate_groups

    def _find_duplicates_cpu(self, video_paths: list[str]) -> list[list[tuple[str, float]]]:
        """Find duplicates using CPU-based comparison."""
        n = len(video_paths)

        # Track which videos have been assigned to a group
        assigned: set[str] = set()
        duplicate_groups: list[list[tuple[str, float]]] = []

        # Compare all pairs
        total_comparisons = n * (n - 1) // 2

        with tqdm(total=total_comparisons, desc="Comparing") as pbar:
            for i in range(n):
                if video_paths[i] in assigned:
                    pbar.update(n - i - 1)
                    continue

                current_group: list[tuple[str, float]] = [(video_paths[i], 1.0)]

                for j in range(i + 1, n):
                    pbar.update(1)

                    if video_paths[j] in assigned:
                        continue

                    info1 = self.video_infos[video_paths[i]]
                    info2 = self.video_infos[video_paths[j]]

                    similarity = self.compute_similarity(info1, info2)

                    if similarity >= self.similarity_threshold:
                        current_group.append((video_paths[j], similarity))
                        assigned.add(video_paths[j])

                if len(current_group) > 1:
                    assigned.add(video_paths[i])
                    duplicate_groups.append(current_group)

        logger.info(f"Found {len(duplicate_groups)} groups of duplicates")
        return duplicate_groups


def print_results(
    duplicate_groups: list[list[tuple[str, float]]],
    video_infos: dict[str, VideoInfo],
    output_file: str | None = None,
) -> None:
    """Print duplicate detection results."""
    if not duplicate_groups:
        print("\n✓ No duplicate videos found!")
        return

    print(f"\n{'=' * 80}")
    print(f"DUPLICATE VIDEO GROUPS FOUND: {len(duplicate_groups)}")
    print(f"{'=' * 80}\n")

    results = []

    for group_idx, group in enumerate(duplicate_groups, 1):
        print(f"Group {group_idx}:")
        print("-" * 40)

        group_data = []

        for video_path, similarity in group:
            info = video_infos.get(video_path)

            if info:
                print(f"  • {video_path}")
                print(f"    Resolution: {info.width}x{info.height}")
                print(f"    Duration: {format_duration(info.duration)}")
                print(f"    Size: {format_file_size(info.file_size)}")
                print(f"    Similarity: {similarity * 100:.1f}%")
                print()

                group_data.append(
                    {
                        "path": video_path,
                        "resolution": f"{info.width}x{info.height}",
                        "duration": info.duration,
                        "size": info.file_size,
                        "similarity": similarity,
                    }
                )

        results.append(group_data)
        print()

    # Calculate potential space savings
    total_savings = 0
    for group in duplicate_groups:
        sizes = [video_infos[path].file_size for path, _ in group if path in video_infos]
        if len(sizes) > 1:
            total_savings += sum(sorted(sizes)[:-1])  # All but largest

    print(f"{'=' * 80}")
    print(f"Potential space savings by removing duplicates: {format_file_size(total_savings)}")
    print(f"{'=' * 80}")

    # Save to JSON if output file specified
    if output_file:
        output_data = {
            "duplicate_groups": results,
            "total_groups": len(duplicate_groups),
            "potential_savings_bytes": total_savings,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_file}")
