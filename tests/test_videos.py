"""
Tests for dupfinder.videos module.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dupfinder.videos import VideoDuplicateFinder, VideoHasher, VideoInfo


class TestVideoInfo:
    """Tests for VideoInfo dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        info = VideoInfo(path="/path/to/video.mp4")
        assert info.path == "/path/to/video.mp4"
        assert info.duration == 0.0
        assert info.fps == 0.0
        assert info.frame_count == 0
        assert info.width == 0
        assert info.height == 0
        assert info.file_size == 0
        assert info.frame_hashes == []
        assert info.average_hash is None

    def test_to_dict(self) -> None:
        """Should convert to dictionary correctly."""
        info = VideoInfo(
            path="/path/video.mp4",
            duration=120.5,
            fps=30.0,
            frame_count=3615,
            width=1920,
            height=1080,
            file_size=50000000,
            average_hash="abc123",
        )
        d = info.to_dict()
        assert d["path"] == "/path/video.mp4"
        assert d["duration"] == 120.5
        assert d["fps"] == 30.0
        assert d["width"] == 1920
        assert d["height"] == 1080
        assert d["average_hash"] == "abc123"


class TestVideoHasher:
    """Tests for VideoHasher class."""

    def test_init_defaults(self) -> None:
        """Should use default values."""
        hasher = VideoHasher()
        assert hasher.num_frames == 10
        assert hasher.hash_size == 16

    def test_init_custom_values(self) -> None:
        """Should accept custom values."""
        hasher = VideoHasher(num_frames=15, hash_size=8)
        assert hasher.num_frames == 15
        assert hasher.hash_size == 8

    def test_extract_frames(self, sample_video: Path) -> None:
        """Should extract frames from video."""
        hasher = VideoHasher(num_frames=5)
        frames, info = hasher.extract_frames(str(sample_video))

        assert len(frames) <= 5
        assert info.path == str(sample_video)
        assert info.fps > 0
        assert info.frame_count > 0
        assert info.width == 100
        assert info.height == 100
        assert info.file_size > 0

    def test_extract_frames_invalid_video(self, temp_dir: Path) -> None:
        """Should handle invalid video files gracefully."""
        invalid_path = temp_dir / "invalid.mp4"
        invalid_path.write_text("not a video")

        hasher = VideoHasher()
        frames, info = hasher.extract_frames(str(invalid_path))

        assert frames == []
        assert info.path == str(invalid_path)

    def test_extract_frames_nonexistent(self) -> None:
        """Should handle nonexistent files gracefully."""
        hasher = VideoHasher()
        frames, info = hasher.extract_frames("/nonexistent/video.mp4")

        assert frames == []

    def test_compute_frame_hash(self, sample_video: Path) -> None:
        """Should compute hash for a single frame."""
        hasher = VideoHasher()
        frames, _ = hasher.extract_frames(str(sample_video))

        if frames:
            hash_str = hasher.compute_frame_hash(frames[0])
            assert isinstance(hash_str, str)
            assert len(hash_str) > 0

    def test_compute_video_hashes(self, sample_video: Path) -> None:
        """Should compute hashes for entire video."""
        hasher = VideoHasher(num_frames=5)
        info = hasher.compute_video_hashes(str(sample_video))

        assert info.path == str(sample_video)
        assert len(info.frame_hashes) > 0
        assert info.average_hash is not None

    def test_hash_consistency(self, sample_video: Path) -> None:
        """Same video should produce same hashes."""
        hasher = VideoHasher(num_frames=5)
        
        info1 = hasher.compute_video_hashes(str(sample_video))
        info2 = hasher.compute_video_hashes(str(sample_video))

        assert info1.frame_hashes == info2.frame_hashes
        assert info1.average_hash == info2.average_hash


class TestVideoDuplicateFinder:
    """Tests for VideoDuplicateFinder class."""

    def test_init_defaults(self) -> None:
        """Should use default values."""
        finder = VideoDuplicateFinder(use_gpu=False)
        assert finder.similarity_threshold == 0.85
        assert finder.hash_size == 16

    def test_init_custom_values(self) -> None:
        """Should accept custom values."""
        finder = VideoDuplicateFinder(
            similarity_threshold=0.90,
            num_frames=15,
            hash_size=8,
            num_workers=2,
            use_gpu=False,
        )
        assert finder.similarity_threshold == 0.90
        assert finder.num_workers == 2
        assert finder.hash_size == 8

    def test_find_videos(self, temp_dir: Path) -> None:
        """Should find video files in folder."""
        # Create sample video files
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
        for name in ["video1.mp4", "video2.avi"]:
            video_path = temp_dir / name
            writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (100, 100))
            for _ in range(30):
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()

        finder = VideoDuplicateFinder(use_gpu=False)
        videos = finder.find_videos(str(temp_dir))

        assert len(videos) == 2

    def test_find_videos_excludes_system_folders(self, temp_dir: Path) -> None:
        """Should exclude videos in system folders."""
        # Create video in normal folder
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
        normal_path = temp_dir / "video.mp4"
        writer = cv2.VideoWriter(str(normal_path), fourcc, 30.0, (100, 100))
        for _ in range(30):
            writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.release()

        # Create video in excluded folder
        recycle = temp_dir / "$RECYCLE.BIN"
        recycle.mkdir()
        excluded_path = recycle / "video.mp4"
        writer = cv2.VideoWriter(str(excluded_path), fourcc, 30.0, (100, 100))
        for _ in range(30):
            writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.release()

        finder = VideoDuplicateFinder(use_gpu=False)
        videos = finder.find_videos(str(temp_dir))

        assert len(videos) == 1
        assert str(normal_path) in videos

    def test_process_videos(self, sample_video: Path) -> None:
        """Should process videos and compute hashes."""
        finder = VideoDuplicateFinder(use_gpu=False, num_frames=5)
        infos = finder.process_videos([str(sample_video)])

        assert len(infos) == 1
        assert str(sample_video) in infos
        info = infos[str(sample_video)]
        assert len(info.frame_hashes) > 0

    def test_compute_similarity_identical(self, sample_video: Path) -> None:
        """Identical videos should have high similarity."""
        finder = VideoDuplicateFinder(use_gpu=False, num_frames=5)
        info = finder.hasher.compute_video_hashes(str(sample_video))

        similarity = finder.compute_similarity(info, info)
        assert similarity == 1.0

    @pytest.mark.skipif(
        not cv2.VideoWriter_fourcc(*"mp4v"),
        reason="OpenCV video codecs not available"
    )
    def test_compute_similarity_duplicates(
        self, duplicate_videos: tuple[Path, Path]
    ) -> None:
        """Duplicate videos should have high similarity."""
        original, resized = duplicate_videos
        finder = VideoDuplicateFinder(use_gpu=False, num_frames=5)

        info_original = finder.hasher.compute_video_hashes(str(original))
        info_resized = finder.hasher.compute_video_hashes(str(resized))

        # Skip if video extraction failed
        if not info_original.frame_hashes or not info_resized.frame_hashes:
            pytest.skip("Video extraction failed")

        similarity = finder.compute_similarity(info_original, info_resized)
        # Resized videos should be somewhat similar
        assert similarity > 0.5


class TestVideoDuplicateFinderIntegration:
    """Integration tests for VideoDuplicateFinder."""

    def test_full_workflow(self, sample_video: Path) -> None:
        """Test complete duplicate finding workflow."""
        finder = VideoDuplicateFinder(
            similarity_threshold=0.85,
            num_frames=5,
            use_gpu=False,
        )

        # Find videos
        videos = finder.find_videos(str(sample_video.parent))
        assert isinstance(videos, list)

        # Process videos
        infos = finder.process_videos(videos)
        assert isinstance(infos, dict)

        # Find duplicates
        groups = finder.find_duplicates()
        assert isinstance(groups, list)

    def test_empty_folder(self, temp_dir: Path) -> None:
        """Should handle empty folders gracefully."""
        finder = VideoDuplicateFinder(use_gpu=False)
        
        videos = finder.find_videos(str(temp_dir))
        assert videos == []

        finder.process_videos(videos)
        groups = finder.find_duplicates()
        assert groups == []

    def test_invalid_videos_skipped(self, temp_dir: Path) -> None:
        """Should skip invalid video files during processing."""
        # Create invalid "video" file
        invalid_path = temp_dir / "invalid.mp4"
        invalid_path.write_text("not a video file")

        # Create valid video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        valid_path = temp_dir / "valid.mp4"
        writer = cv2.VideoWriter(str(valid_path), fourcc, 30.0, (100, 100))
        for _ in range(30):
            writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.release()

        finder = VideoDuplicateFinder(use_gpu=False, num_frames=5)
        videos = finder.find_videos(str(temp_dir))
        infos = finder.process_videos(videos)

        # Should only have processed the valid video
        assert str(valid_path) in infos
        # Invalid video should be skipped (no frame_hashes)
        assert str(invalid_path) not in infos or not infos.get(str(invalid_path), VideoInfo(path="")).frame_hashes
