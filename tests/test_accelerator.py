"""
Tests for dupfinder.accelerator module.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from dupfinder.accelerator import (
    AcceleratorType,
    GPUAccelerator,
    compute_phash_gpu,
    compute_similarity_matrix_gpu,
    get_accelerator,
)


class TestAcceleratorType:
    """Tests for AcceleratorType enum."""

    def test_cuda_type(self) -> None:
        """Should have CUDA type."""
        assert hasattr(AcceleratorType, "CUDA")

    def test_rocm_type(self) -> None:
        """Should have ROCm type."""
        assert hasattr(AcceleratorType, "ROCM")

    def test_opencl_type(self) -> None:
        """Should have OpenCL type."""
        assert hasattr(AcceleratorType, "OPENCL")

    def test_cpu_type(self) -> None:
        """Should have CPU type."""
        assert hasattr(AcceleratorType, "CPU")


class TestGPUAccelerator:
    """Tests for GPUAccelerator class."""

    def test_singleton_pattern(self) -> None:
        """Should return same instance each time."""
        acc1 = GPUAccelerator()
        acc2 = GPUAccelerator()
        assert acc1 is acc2

    def test_backend_is_set(self) -> None:
        """Should have a backend set after initialization."""
        acc = GPUAccelerator()
        assert isinstance(acc.backend, AcceleratorType)

    def test_num_cpus_positive(self) -> None:
        """Should detect positive number of CPUs."""
        acc = GPUAccelerator()
        assert acc.num_cpus > 0
        assert acc.num_cpus == mp.cpu_count()

    def test_is_gpu_available_property(self) -> None:
        """Should properly report GPU availability."""
        acc = GPUAccelerator()
        is_gpu = acc.is_gpu_available
        assert isinstance(is_gpu, bool)

        if is_gpu:
            assert acc.backend in (
                AcceleratorType.CUDA,
                AcceleratorType.ROCM,
                AcceleratorType.OPENCL,
            )
        else:
            assert acc.backend == AcceleratorType.CPU

    def test_get_backend_name(self) -> None:
        """Should return human-readable backend name."""
        acc = GPUAccelerator()
        name = acc.get_backend_name()
        assert isinstance(name, str)
        assert len(name) > 0

        # Should contain descriptive text
        expected_names = {
            AcceleratorType.CUDA: "CUDA",
            AcceleratorType.ROCM: "ROCm",
            AcceleratorType.OPENCL: "OpenCL",
            AcceleratorType.CPU: "CPU",
        }
        assert expected_names[acc.backend] in name


class TestGetAccelerator:
    """Tests for get_accelerator function."""

    def test_returns_accelerator(self) -> None:
        """Should return a GPUAccelerator instance."""
        acc = get_accelerator()
        assert isinstance(acc, GPUAccelerator)

    def test_returns_singleton(self) -> None:
        """Should return the same singleton instance."""
        acc1 = get_accelerator()
        acc2 = get_accelerator()
        assert acc1 is acc2


class TestResizeImageBatch:
    """Tests for image batch resizing."""

    def test_resize_single_image(self) -> None:
        """Should resize a single image."""
        acc = get_accelerator()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        resized = acc.resize_image_batch([img], (50, 50))
        
        assert len(resized) == 1
        assert resized[0].shape[:2] == (50, 50)

    def test_resize_multiple_images(self) -> None:
        """Should resize multiple images."""
        acc = get_accelerator()
        images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8),
            np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8),
        ]
        
        resized = acc.resize_image_batch(images, (64, 64))
        
        assert len(resized) == 3
        for img in resized:
            assert img.shape[:2] == (64, 64)

    def test_resize_grayscale(self) -> None:
        """Should resize grayscale images."""
        acc = get_accelerator()
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        resized = acc.resize_image_batch([img], (50, 50))
        
        assert len(resized) == 1
        # Result might be 2D or 3D with single channel
        assert resized[0].shape[0] == 50
        assert resized[0].shape[1] == 50

    def test_resize_empty_list(self) -> None:
        """Should handle empty list."""
        acc = get_accelerator()
        resized = acc.resize_image_batch([], (50, 50))
        assert resized == []


class TestComputeDCTBatch:
    """Tests for DCT batch computation."""

    def test_dct_single_image(self) -> None:
        """Should compute DCT for single image."""
        acc = get_accelerator()
        img = np.random.rand(32, 32).astype(np.float32)
        
        dct_results = acc.compute_dct_batch([img])
        
        assert len(dct_results) == 1
        assert dct_results[0].shape == (32, 32)

    def test_dct_multiple_images(self) -> None:
        """Should compute DCT for multiple images."""
        acc = get_accelerator()
        images = [
            np.random.rand(32, 32).astype(np.float32),
            np.random.rand(32, 32).astype(np.float32),
        ]
        
        dct_results = acc.compute_dct_batch(images)
        
        assert len(dct_results) == 2


class TestComputeSimilarityMatrix:
    """Tests for similarity matrix computation."""

    def test_similarity_identical_hashes(self) -> None:
        """Identical hashes should have similarity 1.0."""
        acc = get_accelerator()
        hash_array = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
        
        matrix = acc.compute_similarity_matrix([hash_array, hash_array])
        
        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == 1.0
        assert matrix[1, 1] == 1.0
        assert matrix[0, 1] == 1.0
        assert matrix[1, 0] == 1.0

    def test_similarity_opposite_hashes(self) -> None:
        """Opposite hashes should have similarity 0.0."""
        acc = get_accelerator()
        hash1 = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
        hash2 = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
        
        matrix = acc.compute_similarity_matrix([hash1, hash2])
        
        assert matrix[0, 1] == 0.0
        assert matrix[1, 0] == 0.0

    def test_similarity_matrix_symmetric(self) -> None:
        """Similarity matrix should be symmetric."""
        acc = get_accelerator()
        hashes = [
            np.random.randint(0, 2, 64, dtype=np.uint8),
            np.random.randint(0, 2, 64, dtype=np.uint8),
            np.random.randint(0, 2, 64, dtype=np.uint8),
        ]
        
        matrix = acc.compute_similarity_matrix(hashes)
        
        assert matrix.shape == (3, 3)
        np.testing.assert_array_almost_equal(matrix, matrix.T)

    def test_similarity_diagonal_is_one(self) -> None:
        """Diagonal elements should be 1.0."""
        acc = get_accelerator()
        hashes = [
            np.random.randint(0, 2, 64, dtype=np.uint8),
            np.random.randint(0, 2, 64, dtype=np.uint8),
        ]
        
        matrix = acc.compute_similarity_matrix(hashes)
        
        assert matrix[0, 0] == 1.0
        assert matrix[1, 1] == 1.0

    def test_similarity_empty_list(self) -> None:
        """Should handle empty list."""
        acc = get_accelerator()
        matrix = acc.compute_similarity_matrix([])
        assert len(matrix) == 0


class TestComputeSimilarityMatrixGPU:
    """Tests for compute_similarity_matrix_gpu function."""

    def test_hex_string_hashes(self) -> None:
        """Should handle hex string hashes."""
        # Create hex hashes (16x16 hash = 256 bits = 64 hex chars)
        hash1 = "f" * 64  # All 1s
        hash2 = "0" * 64  # All 0s
        
        matrix = compute_similarity_matrix_gpu([hash1, hash2], hash_size=16)
        
        assert matrix.shape == (2, 2)
        assert matrix[0, 1] == 0.0  # Opposite hashes

    def test_empty_list(self) -> None:
        """Should handle empty list."""
        matrix = compute_similarity_matrix_gpu([])
        assert len(matrix) == 0


class TestComputePhashGPU:
    """Tests for compute_phash_gpu function."""

    def test_phash_returns_list(self) -> None:
        """Should return list of hashes."""
        images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
        ]
        
        hashes = compute_phash_gpu(images, hash_size=8)
        
        assert len(hashes) == 2
        for h in hashes:
            assert len(h) == 64  # 8x8 = 64 bits

    def test_phash_grayscale(self) -> None:
        """Should handle grayscale images."""
        images = [
            np.random.randint(0, 255, (100, 100), dtype=np.uint8),
        ]
        
        hashes = compute_phash_gpu(images, hash_size=8)
        
        assert len(hashes) == 1

    def test_phash_consistency(self) -> None:
        """Same image should produce same hash."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        hashes1 = compute_phash_gpu([img], hash_size=8)
        hashes2 = compute_phash_gpu([img], hash_size=8)
        
        np.testing.assert_array_equal(hashes1[0], hashes2[0])


class TestCPUFallback:
    """Tests for CPU fallback functionality."""

    def test_cpu_resize_works(self) -> None:
        """CPU resize should work correctly."""
        acc = get_accelerator()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Force CPU path
        result = acc._resize_batch_cpu([img], (50, 50))
        
        assert len(result) == 1
        assert result[0].shape[:2] == (50, 50)

    def test_cpu_dct_works(self) -> None:
        """CPU DCT should work correctly."""
        acc = get_accelerator()
        img = np.random.rand(32, 32).astype(np.float32)
        
        # Force CPU path
        result = acc._dct_batch_cpu([img])
        
        assert len(result) == 1
        assert result[0].shape == (32, 32)

    def test_cpu_similarity_works(self) -> None:
        """CPU similarity matrix should work correctly."""
        acc = get_accelerator()
        hash_arrays = [
            np.random.randint(0, 2, 64, dtype=np.uint8),
            np.random.randint(0, 2, 64, dtype=np.uint8),
        ]
        hash_matrix = np.vstack(hash_arrays)
        
        # Force CPU path
        result = acc._similarity_matrix_cpu(hash_matrix, len(hash_arrays))
        
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result, result.T)
