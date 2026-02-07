"""
GPU Accelerator Module

Provides GPU-accelerated operations for image/video processing with automatic
fallback through: CUDA -> AMD/ROCm -> OpenCL -> CPU multiprocessing

Accelerates:
1. Image resizing/preprocessing
2. DCT computation for perceptual hashing
3. Hamming distance computation for similarity matrix
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import warnings
from enum import Enum, auto

import numpy as np

# Suppress PyTorch CUDA capability warnings during detection
warnings.filterwarnings("ignore", message=".*CUDA capability.*")
warnings.filterwarnings("ignore", message=".*cuda capability.*")
warnings.filterwarnings("ignore", message=".*Please install PyTorch.*")

logger = logging.getLogger(__name__)


class AcceleratorType(Enum):
    """Available acceleration backends."""

    CUDA = auto()
    ROCM = auto()  # AMD via PyTorch ROCm
    OPENCL = auto()  # AMD/Intel via PyOpenCL
    CPU = auto()


class GPUAccelerator:
    """
    GPU-accelerated operations with automatic backend detection and fallback.

    Priority: CUDA -> ROCm -> OpenCL -> CPU multiprocessing
    """

    _instance: GPUAccelerator | None = None
    _initialized: bool = False

    def __new__(cls) -> GPUAccelerator:
        """Singleton pattern to avoid multiple GPU initializations."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self.backend: AcceleratorType = AcceleratorType.CPU
        self.device = None
        self.torch_device = None
        self._torch = None
        self._cp = None  # CuPy
        self._cl = None  # PyOpenCL
        self._cl_ctx = None
        self._cl_queue = None
        self.num_cpus = mp.cpu_count()

        self._detect_backend()
        self._initialized = True

    def _detect_backend(self) -> None:
        """Detect available GPU backend with priority fallback."""
        # Try CUDA first (via PyTorch or CuPy)
        if self._try_cuda():
            return

        # Try AMD ROCm (via PyTorch)
        if self._try_rocm():
            return

        # Try OpenCL (AMD/Intel/NVIDIA fallback)
        if self._try_opencl():
            return

        # Fall back to CPU
        self._setup_cpu()

    def _try_cuda(self) -> bool:
        """Try to initialize CUDA backend."""
        # Try PyTorch CUDA first
        try:
            import torch

            if torch.cuda.is_available():
                # Verify CUDA actually works by running a simple operation
                try:
                    test_device = torch.device("cuda")
                    # Run a small test to ensure the GPU is compatible
                    test_tensor = torch.zeros(1, device=test_device)
                    _ = test_tensor + 1  # Simple operation to verify kernel execution
                    del test_tensor
                    torch.cuda.empty_cache()

                    self._torch = torch
                    self.torch_device = test_device
                    self.backend = AcceleratorType.CUDA
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"Using CUDA acceleration via PyTorch: {gpu_name}")
                    return True
                except RuntimeError as e:
                    # CUDA capability mismatch or other runtime error
                    logger.warning(f"PyTorch CUDA available but not functional: {e}")
        except ImportError:
            pass

        # Try CuPy as alternative
        try:
            import cupy as cp

            # Test if CUDA is actually available and functional
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                # Run a test operation
                test_arr = cp.zeros(1)
                _ = test_arr + 1
                del test_arr
                cp.get_default_memory_pool().free_all_blocks()

                self._cp = cp
                self.backend = AcceleratorType.CUDA
                device_props = cp.cuda.runtime.getDeviceProperties(0)
                gpu_name = (
                    device_props["name"].decode()
                    if isinstance(device_props["name"], bytes)
                    else device_props["name"]
                )
                logger.info(f"Using CUDA acceleration via CuPy: {gpu_name}")
                return True
        except (ImportError, Exception) as e:
            logger.debug(f"CuPy CUDA not available: {e}")

        return False

    def _try_rocm(self) -> bool:
        """Try to initialize AMD ROCm backend via PyTorch."""
        try:
            import torch

            # Check for ROCm (shows as cuda in PyTorch but on AMD hardware)
            if torch.cuda.is_available():
                # Already handled in CUDA check
                return False

            # Check for AMD HIP/ROCm
            if hasattr(torch, "hip") and torch.hip.is_available():
                self._torch = torch
                self.torch_device = torch.device("cuda")  # ROCm uses cuda device
                self.backend = AcceleratorType.ROCM
                logger.info("Using AMD ROCm acceleration via PyTorch")
                return True
        except (ImportError, AttributeError):
            pass

        return False

    def _try_opencl(self) -> bool:
        """Try to initialize OpenCL backend."""
        try:
            import pyopencl as cl

            platforms = cl.get_platforms()
            if not platforms:
                return False

            # Prefer GPU devices
            for platform in platforms:
                try:
                    devices = platform.get_devices(device_type=cl.device_type.GPU)
                    if devices:
                        self._cl = cl
                        self._cl_ctx = cl.Context(devices=[devices[0]])
                        self._cl_queue = cl.CommandQueue(self._cl_ctx)
                        self.backend = AcceleratorType.OPENCL
                        logger.info(f"Using OpenCL acceleration: {devices[0].name}")
                        return True
                except cl.RuntimeError:
                    continue

        except ImportError:
            pass

        return False

    def _setup_cpu(self) -> None:
        """Set up CPU multiprocessing backend."""
        self.backend = AcceleratorType.CPU
        logger.info(f"Using CPU multiprocessing with {self.num_cpus} cores")

    @property
    def is_gpu_available(self) -> bool:
        """Check if any GPU acceleration is available."""
        return self.backend in (
            AcceleratorType.CUDA,
            AcceleratorType.ROCM,
            AcceleratorType.OPENCL,
        )

    def get_backend_name(self) -> str:
        """Get human-readable backend name."""
        names = {
            AcceleratorType.CUDA: "CUDA (NVIDIA GPU)",
            AcceleratorType.ROCM: "ROCm (AMD GPU)",
            AcceleratorType.OPENCL: "OpenCL (GPU)",
            AcceleratorType.CPU: "CPU Multiprocessing",
        }
        return names.get(self.backend, "Unknown")

    def resize_image_batch(
        self,
        images: list[np.ndarray],
        target_size: tuple[int, int],
    ) -> list[np.ndarray]:
        """
        Resize a batch of images using GPU acceleration.

        Args:
            images: List of numpy arrays (H, W, C) or (H, W)
            target_size: Target size as (width, height)

        Returns:
            List of resized images as numpy arrays
        """
        if not images:
            return []

        if self.backend == AcceleratorType.CUDA and self._torch is not None:
            return self._resize_batch_torch(images, target_size)
        elif self.backend == AcceleratorType.CUDA and self._cp is not None:
            return self._resize_batch_cupy(images, target_size)
        else:
            return self._resize_batch_cpu(images, target_size)

    def _resize_batch_torch(
        self,
        images: list[np.ndarray],
        target_size: tuple[int, int],
    ) -> list[np.ndarray]:
        """Resize images using PyTorch GPU."""
        import torch.nn.functional as functional

        results = []
        target_h, target_w = target_size[1], target_size[0]

        for img in images:
            # Handle grayscale
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]

            # Convert to torch tensor (B, C, H, W)
            tensor = self._torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.to(self.torch_device)

            # Resize
            resized = functional.interpolate(
                tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

            # Convert back
            result = resized.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            if result.shape[2] == 1:
                result = result.squeeze(2)
            results.append(result)

        return results

    def _resize_batch_cupy(
        self,
        images: list[np.ndarray],
        target_size: tuple[int, int],
    ) -> list[np.ndarray]:
        """Resize images using CuPy."""
        from cupyx.scipy.ndimage import zoom

        results = []

        for img in images:
            # Calculate zoom factors
            if len(img.shape) == 2:
                zoom_factors = (target_size[1] / img.shape[0], target_size[0] / img.shape[1])
            else:
                zoom_factors = (
                    target_size[1] / img.shape[0],
                    target_size[0] / img.shape[1],
                    1,
                )

            # Transfer to GPU, resize, transfer back
            gpu_img = self._cp.asarray(img)
            resized = zoom(gpu_img, zoom_factors, order=1)
            results.append(self._cp.asnumpy(resized).astype(np.uint8))

        return results

    def _resize_batch_cpu(
        self,
        images: list[np.ndarray],
        target_size: tuple[int, int],
    ) -> list[np.ndarray]:
        """Resize images using CPU (OpenCV)."""
        import cv2

        results = []
        for img in images:
            resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            results.append(resized)

        return results

    def compute_dct_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """
        Compute DCT for a batch of images (used in pHash).

        Args:
            images: List of grayscale images as numpy arrays

        Returns:
            List of DCT coefficients
        """
        if self.backend == AcceleratorType.CUDA and self._torch is not None:
            return self._dct_batch_torch(images)
        elif self.backend == AcceleratorType.CUDA and self._cp is not None:
            return self._dct_batch_cupy(images)
        else:
            return self._dct_batch_cpu(images)

    def _dct_batch_torch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Compute DCT using PyTorch (via manual implementation)."""
        results = []

        for img in images:
            # Convert to float tensor
            tensor = self._torch.from_numpy(img.astype(np.float32)).to(self.torch_device)

            # Simple DCT approximation using FFT
            dct = self._torch.fft.fft2(tensor).real
            results.append(dct.cpu().numpy())

        return results

    def _dct_batch_cupy(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Compute DCT using CuPy."""
        results = []

        for img in images:
            gpu_img = self._cp.asarray(img.astype(np.float32))
            # Use FFT-based DCT approximation
            dct = self._cp.fft.fft2(gpu_img).real
            results.append(self._cp.asnumpy(dct))

        return results

    def _dct_batch_cpu(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Compute DCT using scipy on CPU."""
        from scipy.fftpack import dct

        results = []
        for img in images:
            # 2D DCT
            dct_result = dct(dct(img.astype(np.float32).T, norm="ortho").T, norm="ortho")
            results.append(dct_result)

        return results

    def compute_similarity_matrix(
        self,
        hashes: list[np.ndarray],
        threshold: float = 0.0,
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix for hash arrays.

        This is the O(n²) operation that benefits most from GPU acceleration.

        Args:
            hashes: List of binary hash arrays (flattened)
            threshold: Minimum similarity to store (0 = store all)

        Returns:
            Similarity matrix as numpy array
        """
        if not hashes:
            return np.array([])

        n = len(hashes)

        # Convert to binary matrix
        hash_matrix = np.vstack([h.flatten() for h in hashes]).astype(np.float32)

        if self.backend == AcceleratorType.CUDA and self._torch is not None:
            return self._similarity_matrix_torch(hash_matrix, n)
        elif self.backend == AcceleratorType.CUDA and self._cp is not None:
            return self._similarity_matrix_cupy(hash_matrix, n)
        else:
            return self._similarity_matrix_cpu(hash_matrix, n)

    def _similarity_matrix_torch(self, hash_matrix: np.ndarray, n: int) -> np.ndarray:
        """Compute similarity matrix using PyTorch GPU."""
        # Transfer to GPU
        gpu_hashes = self._torch.from_numpy(hash_matrix).to(self.torch_device)

        # Compute Hamming distances using XOR and sum
        h1 = gpu_hashes.unsqueeze(1)  # (n, 1, hash_size)
        h2 = gpu_hashes.unsqueeze(0)  # (1, n, hash_size)

        # XOR and count differences
        diff = (h1 != h2).float().sum(dim=2)

        # Convert to similarity (1 - normalized_distance)
        max_dist = hash_matrix.shape[1]
        similarity = 1.0 - (diff / max_dist)

        return similarity.cpu().numpy()

    def _similarity_matrix_cupy(self, hash_matrix: np.ndarray, n: int) -> np.ndarray:
        """Compute similarity matrix using CuPy."""
        gpu_hashes = self._cp.asarray(hash_matrix)

        # Compute all pairwise Hamming distances
        h1 = gpu_hashes[:, self._cp.newaxis, :]  # (n, 1, hash_size)
        h2 = gpu_hashes[self._cp.newaxis, :, :]  # (1, n, hash_size)

        diff = self._cp.sum(h1 != h2, axis=2).astype(self._cp.float32)

        max_dist = hash_matrix.shape[1]
        similarity = 1.0 - (diff / max_dist)

        return self._cp.asnumpy(similarity)

    def _similarity_matrix_cpu(self, hash_matrix: np.ndarray, n: int) -> np.ndarray:
        """Compute similarity matrix using CPU with multiprocessing."""
        from scipy.spatial.distance import cdist

        # Use Hamming distance from scipy (optimized C implementation)
        distances = cdist(hash_matrix, hash_matrix, metric="hamming")
        similarity = 1.0 - distances

        return similarity.astype(np.float32)

    def batch_hamming_distance(
        self,
        hashes1: list[str],
        hashes2: list[str],
    ) -> np.ndarray:
        """
        Compute Hamming distances between two lists of hex hash strings.

        Args:
            hashes1: First list of hex hash strings
            hashes2: Second list of hex hash strings

        Returns:
            Distance matrix (len(hashes1) x len(hashes2))
        """

        def hex_to_binary(hex_str: str) -> np.ndarray:
            return np.array([int(b) for b in bin(int(hex_str, 16))[2:].zfill(len(hex_str) * 4)])

        arr1 = np.vstack([hex_to_binary(h) for h in hashes1]).astype(np.float32)
        arr2 = np.vstack([hex_to_binary(h) for h in hashes2]).astype(np.float32)

        if self.backend == AcceleratorType.CUDA and self._torch is not None:
            return self._batch_hamming_torch(arr1, arr2)
        elif self.backend == AcceleratorType.CUDA and self._cp is not None:
            return self._batch_hamming_cupy(arr1, arr2)
        else:
            return self._batch_hamming_cpu(arr1, arr2)

    def _batch_hamming_torch(self, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Compute Hamming distances using PyTorch."""
        gpu_arr1 = self._torch.from_numpy(arr1).to(self.torch_device)
        gpu_arr2 = self._torch.from_numpy(arr2).to(self.torch_device)

        # Compute pairwise XOR and sum
        h1 = gpu_arr1.unsqueeze(1)
        h2 = gpu_arr2.unsqueeze(0)

        distances = (h1 != h2).float().sum(dim=2)

        return distances.cpu().numpy()

    def _batch_hamming_cupy(self, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Compute Hamming distances using CuPy."""
        gpu_arr1 = self._cp.asarray(arr1)
        gpu_arr2 = self._cp.asarray(arr2)

        h1 = gpu_arr1[:, self._cp.newaxis, :]
        h2 = gpu_arr2[self._cp.newaxis, :, :]

        distances = self._cp.sum(h1 != h2, axis=2)

        return self._cp.asnumpy(distances)

    def _batch_hamming_cpu(self, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Compute Hamming distances using scipy."""
        from scipy.spatial.distance import cdist

        # Scale by hash length to get actual bit differences
        distances = cdist(arr1, arr2, metric="hamming") * arr1.shape[1]

        return distances.astype(np.float32)


# Global accelerator instance
_accelerator: GPUAccelerator | None = None


def get_accelerator() -> GPUAccelerator:
    """Get the global GPU accelerator instance."""
    global _accelerator
    if _accelerator is None:
        _accelerator = GPUAccelerator()
    return _accelerator


def compute_phash_gpu(
    images: list[np.ndarray],
    hash_size: int = 8,
) -> list[np.ndarray]:
    """
    Compute perceptual hashes for images using GPU acceleration.

    Args:
        images: List of images as numpy arrays (grayscale or RGB)
        hash_size: Size of the hash (output will be hash_size x hash_size bits)

    Returns:
        List of binary hash arrays
    """
    acc = get_accelerator()

    if not images:
        return []

    # Target size for DCT (standard is 32x32 for hash_size=8)
    dct_size = hash_size * 4

    # Convert to grayscale if needed
    gray_images = []
    for img in images:
        if len(img.shape) == 3:
            # Simple luminance conversion
            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        else:
            gray = img
        gray_images.append(gray.astype(np.float32))

    # Resize images
    resized = acc.resize_image_batch(
        [img.astype(np.uint8) for img in gray_images],
        (dct_size, dct_size),
    )

    # Compute DCT
    dct_results = acc.compute_dct_batch([r.astype(np.float32) for r in resized])

    # Extract low-frequency components and compute hash
    hashes = []
    for dct in dct_results:
        # Get top-left hash_size x hash_size (low frequencies)
        low_freq = dct[:hash_size, :hash_size]

        # Compute median and threshold
        median = np.median(low_freq)
        hash_bits = (low_freq > median).astype(np.uint8)

        hashes.append(hash_bits.flatten())

    return hashes


def compute_similarity_matrix_gpu(
    hashes: list[str | np.ndarray],
    hash_size: int = 16,
) -> np.ndarray:
    """
    Compute pairwise similarity matrix for hashes using GPU.

    Args:
        hashes: List of hex strings or binary arrays
        hash_size: Size of hash (used for hex string conversion)

    Returns:
        Similarity matrix (n x n)
    """
    acc = get_accelerator()

    if not hashes:
        return np.array([])

    # Convert hex strings to binary arrays if needed
    if isinstance(hashes[0], str):

        def hex_to_binary(hex_str: str) -> np.ndarray:
            try:
                # imagehash format
                bits = bin(int(hex_str, 16))[2:].zfill(hash_size * hash_size)
                return np.array([int(b) for b in bits], dtype=np.uint8)
            except ValueError:
                return np.zeros(hash_size * hash_size, dtype=np.uint8)

        hash_arrays = [hex_to_binary(h) for h in hashes]
    else:
        hash_arrays = [h.flatten() if hasattr(h, "flatten") else np.array(h) for h in hashes]

    return acc.compute_similarity_matrix(hash_arrays)
