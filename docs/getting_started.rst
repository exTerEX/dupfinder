Getting Started
===============

This guide will help you get started with dupfinder.

Installation
------------

Prerequisites
^^^^^^^^^^^^^

- Python 3.10 or higher
- FFmpeg (optional, for better video codec support)

Using UV (Recommended)
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/dupfinder.git
   cd dupfinder

   # Create virtual environment and install
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install .

Using pip
^^^^^^^^^

.. code-block:: bash

   pip install .

   # Or install in development mode
   pip install -e .

GPU Acceleration (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

GPU acceleration significantly speeds up processing for large media collections.

**NVIDIA GPUs (CUDA)**:

.. code-block:: bash

   uv pip install ".[cuda]"
   # Or manually:
   pip install torch --index-url https://download.pytorch.org/whl/cu121

**AMD GPUs (ROCm)**:

.. code-block:: bash

   uv pip install ".[rocm]"
   # Or manually:
   pip install torch --index-url https://download.pytorch.org/whl/rocm5.6

**OpenCL (any GPU)**:

.. code-block:: bash

   uv pip install ".[opencl]"
   # Or manually:
   pip install pyopencl

The acceleration priority is: CUDA → ROCm → OpenCL → CPU multiprocessing

Basic Usage
-----------

Finding Duplicate Images
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Basic usage
   dupfinder images /path/to/photos

   # With custom similarity threshold (0-1)
   dupfinder images /path/to/photos --threshold 0.95

   # Save results to JSON
   dupfinder images /path/to/photos --output duplicates.json

   # Disable GPU (use CPU only)
   dupfinder images /path/to/photos --no-gpu

Finding Duplicate Videos
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Basic usage
   dupfinder videos /path/to/videos

   # With more frames for better accuracy
   dupfinder videos /path/to/videos --frames 20

   # Combined options
   dupfinder videos /path/to/videos -t 0.90 -f 15 -w 8 -o results.json

Python API
----------

Basic Image Processing
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from dupfinder import ImageDuplicateFinder

   # Create finder with custom settings
   finder = ImageDuplicateFinder(
       similarity_threshold=0.90,  # 90% similarity threshold
       hash_size=16,               # Hash resolution
       num_workers=4,              # Parallel workers
       use_gpu=True,               # Enable GPU if available
   )

   # Find all images in folder (recursive)
   images = finder.find_images("/path/to/photos")
   print(f"Found {len(images)} images")

   # Process images and compute hashes
   finder.process_images(images)

   # Find duplicate groups
   duplicates = finder.find_duplicates_fast()

   # Print results
   for group in duplicates:
       print(f"\nDuplicate group ({len(group)} files):")
       for path, similarity in group:
           print(f"  - {path} ({similarity*100:.1f}%)")

Basic Video Processing
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from dupfinder import VideoDuplicateFinder

   # Create finder
   finder = VideoDuplicateFinder(
       similarity_threshold=0.85,
       num_frames=10,  # Frames to extract per video
       use_gpu=True,
   )

   # Find and process videos
   videos = finder.find_videos("/path/to/videos")
   finder.process_videos(videos)    

   # Find duplicates
   duplicates = finder.find_duplicates()

   for group in duplicates:
       print(f"\nDuplicate video group:")
       for video_path, similarity in group:
           info = finder.video_infos.get(video_path)
           if info:
               print(f"  - {video_path}")
               print(f"    Duration: {info.duration:.1f}s")
               print(f"    Resolution: {info.width}x{info.height}")

Using GPU Accelerator Directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from dupfinder import get_accelerator, AcceleratorType
   import numpy as np

   # Get accelerator instance
   accel = get_accelerator()

   # Check what backend is being used
   print(f"Backend: {accel.get_backend_name()}")
   print(f"GPU available: {accel.is_gpu_available}")

   # Use for batch operations
   images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(10)]
   resized = accel.resize_image_batch(images, (64, 64))

Understanding Output
--------------------

Console Output
^^^^^^^^^^^^^^

The tool displays grouped duplicates with metadata::

   ================================================================================
   DUPLICATE IMAGE GROUPS FOUND: 2 (5 duplicate files)
   ================================================================================

   Group 1:
   ----------------------------------------
     • /photos/vacation_original.jpg
       Resolution: 4000x3000
       Format: JPEG
       Size: 5.20 MB
       Similarity: 100.0%

     • /photos/vacation_resized.jpg
       Resolution: 1920x1440
       Format: JPEG
       Size: 1.10 MB  
       Similarity: 98.5%

   ================================================================================
   Potential space savings by removing duplicates: 3.50 MB
   ================================================================================

JSON Output
^^^^^^^^^^^

When using ``--output``, results are saved as JSON:

.. code-block:: json

   {
     "duplicate_groups": [
       [
         {
           "path": "/photos/vacation_original.jpg",
           "resolution": "4000x3000",
           "format": "JPEG",
           "size": 5452800,
           "similarity": 1.0
         },
         {
           "path": "/photos/vacation_resized.jpg",
           "resolution": "1920x1440",
           "format": "JPEG",
           "size": 1153024,
           "similarity": 0.985
         }
       ]
     ],
     "total_groups": 1,
     "total_duplicates": 1,
     "potential_savings_bytes": 1153024
   }

Next Steps
----------

- See :doc:`cli` for complete command-line reference
- See :doc:`configuration` for tuning options
- See :doc:`api/index` for full API documentation
