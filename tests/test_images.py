"""
Tests for dupfinder.images module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dupfinder.images import ImageDuplicateFinder, ImageHasher, ImageInfo


class TestImageInfo:
    """Tests for ImageInfo dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        info = ImageInfo(path="/path/to/image.jpg")
        assert info.path == "/path/to/image.jpg"
        assert info.width == 0
        assert info.height == 0
        assert info.file_size == 0
        assert info.format == ""
        assert info.phash is None
        assert info.ahash is None
        assert info.dhash is None
        assert info.whash is None

    def test_to_dict(self) -> None:
        """Should convert to dictionary correctly."""
        info = ImageInfo(
            path="/path/image.jpg",
            width=1920,
            height=1080,
            file_size=1024000,
            format="JPEG",
            mode="RGB",
            phash="abc123",
            ahash="def456",
            dhash="ghi789",
        )
        d = info.to_dict()
        assert d["path"] == "/path/image.jpg"
        assert d["width"] == 1920
        assert d["height"] == 1080
        assert d["file_size"] == 1024000
        assert d["format"] == "JPEG"
        assert d["phash"] == "abc123"


class TestImageHasher:
    """Tests for ImageHasher class."""

    def test_init_default_hash_size(self) -> None:
        """Should use default hash size."""
        hasher = ImageHasher()
        assert hasher.hash_size == 16

    def test_init_custom_hash_size(self) -> None:
        """Should accept custom hash size."""
        hasher = ImageHasher(hash_size=8)
        assert hasher.hash_size == 8

    def test_compute_hashes_rgb_image(self, sample_image: Path) -> None:
        """Should compute hashes for RGB image."""
        hasher = ImageHasher()
        info = hasher.compute_hashes(str(sample_image))

        assert info.path == str(sample_image)
        assert info.width == 100
        assert info.height == 100
        assert info.file_size > 0
        assert info.format == "JPEG"
        assert info.phash is not None
        assert info.ahash is not None
        assert info.dhash is not None
        assert info.whash is not None

    def test_compute_hashes_grayscale_image(self, sample_grayscale_image: Path) -> None:
        """Should compute hashes for grayscale image."""
        hasher = ImageHasher()
        info = hasher.compute_hashes(str(sample_grayscale_image))

        assert info.width == 100
        assert info.height == 100
        assert info.phash is not None

    def test_compute_hashes_invalid_file(self, temp_dir: Path) -> None:
        """Should handle invalid image files gracefully."""
        invalid_path = temp_dir / "invalid.jpg"
        invalid_path.write_text("not an image")

        hasher = ImageHasher()
        info = hasher.compute_hashes(str(invalid_path))

        assert info.path == str(invalid_path)
        assert info.phash is None

    def test_compute_hashes_nonexistent_file(self) -> None:
        """Should handle nonexistent files gracefully."""
        hasher = ImageHasher()
        info = hasher.compute_hashes("/nonexistent/path/image.jpg")

        assert info.phash is None

    def test_hash_consistency(self, sample_image: Path) -> None:
        """Same image should produce same hash."""
        hasher = ImageHasher()
        info1 = hasher.compute_hashes(str(sample_image))
        info2 = hasher.compute_hashes(str(sample_image))

        assert info1.phash == info2.phash
        assert info1.ahash == info2.ahash
        assert info1.dhash == info2.dhash
        assert info1.whash == info2.whash

    def test_hash_length(self, sample_image: Path) -> None:
        """Hash length should be related to hash_size."""
        hasher = ImageHasher(hash_size=8)
        info = hasher.compute_hashes(str(sample_image))

        # Hash is hex string, length = (hash_size^2) / 4
        expected_hex_length = (8 * 8) // 4  # 16 hex chars
        assert len(info.phash) == expected_hex_length


class TestImageDuplicateFinder:
    """Tests for ImageDuplicateFinder class."""

    def test_init_defaults(self) -> None:
        """Should use default values."""
        finder = ImageDuplicateFinder(use_gpu=False)
        assert finder.similarity_threshold == 0.90
        assert finder.hash_size == 16
        assert finder.hash_type == "combined"

    def test_init_custom_values(self) -> None:
        """Should accept custom values."""
        finder = ImageDuplicateFinder(
            similarity_threshold=0.95,
            hash_size=8,
            num_workers=2,
            hash_type="phash",
            use_gpu=False,
        )
        assert finder.similarity_threshold == 0.95
        assert finder.hash_size == 8
        assert finder.num_workers == 2
        assert finder.hash_type == "phash"

    def test_find_images(self, nested_folder_structure: Path) -> None:
        """Should find images in folder recursively."""
        finder = ImageDuplicateFinder(use_gpu=False)
        images = finder.find_images(str(nested_folder_structure))

        # Should find images but exclude $RECYCLE.BIN and .cache
        assert len(images) == 3

    def test_process_images(self, duplicate_images: tuple[Path, Path, Path]) -> None:
        """Should process images and compute hashes."""
        original, resized, modified = duplicate_images
        finder = ImageDuplicateFinder(use_gpu=False)

        image_files = [str(original), str(resized), str(modified)]
        infos = finder.process_images(image_files)

        assert len(infos) == 3
        for path in image_files:
            assert path in infos
            assert infos[path].phash is not None

    def test_compute_similarity_identical(self, sample_image: Path) -> None:
        """Identical images should have similarity of 1.0."""
        finder = ImageDuplicateFinder(use_gpu=False)
        info = finder.hasher.compute_hashes(str(sample_image))

        similarity = finder.compute_similarity(info, info)
        assert similarity == 1.0

    def test_compute_similarity_duplicates(
        self, duplicate_images: tuple[Path, Path, Path]
    ) -> None:
        """Duplicate images should have high similarity."""
        original, resized, modified = duplicate_images
        finder = ImageDuplicateFinder(use_gpu=False)

        info_original = finder.hasher.compute_hashes(str(original))
        info_resized = finder.hasher.compute_hashes(str(resized))
        info_modified = finder.hasher.compute_hashes(str(modified))

        # Original and resized should be very similar
        sim_resized = finder.compute_similarity(info_original, info_resized)
        assert sim_resized > 0.8

        # Original and modified should also be similar
        sim_modified = finder.compute_similarity(info_original, info_modified)
        assert sim_modified > 0.8

    def test_compute_similarity_different(
        self, temp_dir: Path
    ) -> None:
        """Different images should have low similarity."""
        # Create images with different patterns (not solid colors)
        import numpy as np
        from PIL import Image
        
        # Horizontal gradient
        horizontal_path = temp_dir / "horizontal.jpg"
        h_array = np.zeros((100, 100, 3), dtype=np.uint8)
        for j in range(100):
            h_array[:, j] = [j * 2, j * 2, j * 2]
        Image.fromarray(h_array).save(horizontal_path, "JPEG")
        
        # Vertical gradient
        vertical_path = temp_dir / "vertical.jpg"
        v_array = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            v_array[i, :] = [i * 2, i * 2, i * 2]
        Image.fromarray(v_array).save(vertical_path, "JPEG")

        finder = ImageDuplicateFinder(use_gpu=False)

        info_h = finder.hasher.compute_hashes(str(horizontal_path))
        info_v = finder.hasher.compute_hashes(str(vertical_path))

        similarity = finder.compute_similarity(info_h, info_v)
        # These patterns should be different enough
        assert similarity < 0.9

    def test_compute_similarity_hash_types(
        self, duplicate_images: tuple[Path, Path, Path]
    ) -> None:
        """Should respect hash_type setting."""
        original, resized, _ = duplicate_images

        for hash_type in ["phash", "ahash", "dhash", "whash"]:
            finder = ImageDuplicateFinder(use_gpu=False, hash_type=hash_type)

            info1 = finder.hasher.compute_hashes(str(original))
            info2 = finder.hasher.compute_hashes(str(resized))

            similarity = finder.compute_similarity(info1, info2)
            assert 0 <= similarity <= 1

    def test_find_duplicates_fast(
        self, image_folder_with_duplicates: tuple[Path, int]
    ) -> None:
        """Should find duplicate groups."""
        folder, expected_groups = image_folder_with_duplicates
        finder = ImageDuplicateFinder(
            similarity_threshold=0.85,
            use_gpu=False,
        )

        images = finder.find_images(str(folder))
        finder.process_images(images)
        groups = finder.find_duplicates_fast()

        # Should find some duplicate groups
        assert len(groups) > 0

    def test_find_duplicates_no_duplicates(self, temp_dir: Path) -> None:
        """Should return empty list when no duplicates exist."""
        import numpy as np
        from PIL import Image
        
        # Create clearly different images with patterns
        img1_path = temp_dir / "pattern1.jpg"
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img1[::2, :] = [255, 0, 0]  # Red stripes
        Image.fromarray(img1).save(img1_path, "JPEG")
        
        img2_path = temp_dir / "pattern2.jpg"
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2[:, ::2] = [0, 0, 255]  # Blue stripes
        Image.fromarray(img2).save(img2_path, "JPEG")

        finder = ImageDuplicateFinder(
            similarity_threshold=0.95,
            use_gpu=False,
        )

        finder.process_images([str(img1_path), str(img2_path)])
        groups = finder.find_duplicates_fast()

        # Pattern images should be different enough
        assert len(groups) == 0

    def test_find_duplicates_exact_match(self, temp_dir: Path) -> None:
        """Exact copies should be grouped together."""
        # Create identical images
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        path1 = temp_dir / "copy1.jpg"
        path2 = temp_dir / "copy2.jpg"
        img.save(path1, "JPEG", quality=95)
        img.save(path2, "JPEG", quality=95)

        finder = ImageDuplicateFinder(use_gpu=False)
        finder.process_images([str(path1), str(path2)])
        groups = finder.find_duplicates_fast()

        # Should find exactly one group with both images
        assert len(groups) == 1
        paths_in_group = {item[0] for item in groups[0]}
        assert str(path1) in paths_in_group
        assert str(path2) in paths_in_group


class TestImageDuplicateFinderIntegration:
    """Integration tests for ImageDuplicateFinder."""

    def test_full_workflow(self, nested_folder_structure: Path) -> None:
        """Test complete duplicate finding workflow."""
        finder = ImageDuplicateFinder(
            similarity_threshold=0.90,
            use_gpu=False,
        )

        # Find images
        images = finder.find_images(str(nested_folder_structure))
        assert isinstance(images, list)

        # Process images
        infos = finder.process_images(images)
        assert isinstance(infos, dict)

        # Find duplicates
        groups = finder.find_duplicates_fast()
        assert isinstance(groups, list)

    def test_empty_folder(self, temp_dir: Path) -> None:
        """Should handle empty folders gracefully."""
        finder = ImageDuplicateFinder(use_gpu=False)
        
        images = finder.find_images(str(temp_dir))
        assert images == []

        finder.process_images(images)
        groups = finder.find_duplicates_fast()
        assert groups == []
