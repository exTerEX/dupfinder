Accelerator Module
==================

.. module:: dupfinder.accelerator
   :synopsis: GPU acceleration with automatic fallback

The accelerator module provides GPU-accelerated operations for media processing
with automatic detection and fallback through multiple backends.

Backend Priority
----------------

The accelerator tries backends in this order:

1. **CUDA** (NVIDIA GPUs via PyTorch or CuPy)
2. **ROCm** (AMD GPUs via PyTorch)
3. **OpenCL** (Any GPU with OpenCL support)
4. **CPU** (Multiprocessing fallback)

Classes
-------

AcceleratorType
^^^^^^^^^^^^^^^

.. autoclass:: dupfinder.accelerator.AcceleratorType
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

GPUAccelerator
^^^^^^^^^^^^^^

.. autoclass:: dupfinder.accelerator.GPUAccelerator
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Functions
---------

.. autofunction:: dupfinder.accelerator.get_accelerator

.. autofunction:: dupfinder.accelerator.compute_similarity_matrix_gpu

.. autofunction:: dupfinder.accelerator.compute_phash_gpu

Usage Examples
--------------

Getting the Accelerator
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from dupfinder import get_accelerator, AcceleratorType

   # Get accelerator (singleton)
   accel = get_accelerator()

   # Check backend
   print(f"Backend: {accel.backend}")
   print(f"Backend name: {accel.get_backend_name()}")
   print(f"GPU available: {accel.is_gpu_available}")
   print(f"CPU cores: {accel.num_cpus}")

   # Check specific backend
   if accel.backend == AcceleratorType.CUDA:
       print("Using NVIDIA GPU")
   elif accel.backend == AcceleratorType.CPU:
       print("Using CPU multiprocessing")

Batch Image Resizing
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from dupfinder import get_accelerator

   accel = get_accelerator()

   # Create sample images
   images = [
       np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8),
       np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8),
   ]

   # Resize batch to 64x64
   resized = accel.resize_image_batch(images, (64, 64))

   for i, img in enumerate(resized):
       print(f"Image {i}: {img.shape}")

Computing DCT
^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from dupfinder import get_accelerator

   accel = get_accelerator()

   # Grayscale images for DCT
   images = [
       np.random.rand(32, 32).astype(np.float32),
       np.random.rand(32, 32).astype(np.float32),
   ]

   # Compute DCT batch
   dct_results = accel.compute_dct_batch(images)

   for i, dct in enumerate(dct_results):
       print(f"DCT {i} shape: {dct.shape}")

Computing Similarity Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from dupfinder import get_accelerator

   accel = get_accelerator()

   # Create binary hash arrays
   hashes = [
       np.random.randint(0, 2, 256, dtype=np.uint8),
       np.random.randint(0, 2, 256, dtype=np.uint8),
       np.random.randint(0, 2, 256, dtype=np.uint8),
   ]

   # Compute pairwise similarity
   similarity_matrix = accel.compute_similarity_matrix(hashes)

   print(f"Matrix shape: {similarity_matrix.shape}")
   print(f"Similarity [0,1]: {similarity_matrix[0,1]:.3f}")

Using GPU Functions Directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from dupfinder.accelerator import (
       compute_similarity_matrix_gpu,
       compute_phash_gpu,
   )

   # Compute similarity from hex hashes
   hex_hashes = ["abc123def456", "abc123def457", "000000000000"]
   matrix = compute_similarity_matrix_gpu(hex_hashes, hash_size=8)

   # Compute perceptual hashes for images
   images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)]
   hash_arrays = compute_phash_gpu(images, hash_size=8)

GPU Memory Management
---------------------

The accelerator manages GPU memory automatically:

- PyTorch: Uses ``torch.cuda.empty_cache()``
- CuPy: Uses ``cp.get_default_memory_pool().free_all_blocks()``

For large datasets, consider processing in batches to avoid memory issues:

.. code-block:: python

   accel = get_accelerator()

   # Process in chunks
   all_images = [...]  # Large list
   chunk_size = 1000
   results = []

   for i in range(0, len(all_images), chunk_size):
       chunk = all_images[i:i+chunk_size]
       chunk_results = accel.resize_image_batch(chunk, (64, 64))
       results.extend(chunk_results)

Troubleshooting
---------------

CUDA Not Detected
^^^^^^^^^^^^^^^^^

If CUDA isn't detected despite having an NVIDIA GPU:

1. Verify CUDA is installed: ``nvidia-smi``
2. Install PyTorch with CUDA: ``pip install torch --index-url https://download.pytorch.org/whl/cu121``
3. Check GPU compute capability (requires CC >= 7.0 for recent PyTorch)

ROCm Not Detected
^^^^^^^^^^^^^^^^^

For AMD GPUs:

1. Verify ROCm installation: ``rocm-smi``
2. Install PyTorch for ROCm: ``pip install torch --index-url https://download.pytorch.org/whl/rocm5.6``

Fallback to CPU
^^^^^^^^^^^^^^^

If GPU acceleration isn't available, the accelerator falls back to CPU
multiprocessing. This is automatic and requires no code changes.

CPU performance depends on the number of available cores, controlled by
``accel.num_cpus``.
