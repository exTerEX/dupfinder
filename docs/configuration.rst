Configuration
=============

This page describes configuration options for dupfinder.

Similarity Threshold
--------------------

The similarity threshold determines how similar two files must be to be considered duplicates.

.. list-table:: Recommended Thresholds
   :header-rows: 1
   :widths: 20 20 60

   * - Threshold
     - Use Case
     - Description
   * - 0.95 - 1.0
     - Exact duplicates
     - Nearly identical files, different formats/resolutions
   * - 0.85 - 0.95
     - Similar content
     - Same content with minor edits, watermarks, etc.
   * - 0.70 - 0.85
     - Related content
     - May include re-edits, crops, color changes
   * - < 0.70
     - Not recommended
     - High false positive rate

**Default values:**

- Images: 0.90
- Videos: 0.85 (videos need lower threshold due to encoding variations)

Hash Size
---------

Hash size determines the resolution of the perceptual hash.

.. list-table:: Hash Size Trade-offs
   :header-rows: 1
   :widths: 15 25 30 30

   * - Size
     - Bits
     - Pros
     - Cons
   * - 8
     - 64
     - Fastest, good for quick scans
     - Lower precision, more collisions
   * - 16
     - 256
     - Good balance (default)
     - Slightly slower than 8
   * - 32
     - 1024
     - High precision
     - Slower, larger memory usage

Number of Frames (Videos)
-------------------------

For videos, the number of extracted frames affects accuracy:

.. list-table:: Frame Count Recommendations
   :header-rows: 1
   :widths: 20 30 50

   * - Frames
     - Use Case
     - Notes
   * - 5
     - Quick scan
     - May miss duplicates with different lengths
   * - 10
     - General use (default)
     - Good balance of speed and accuracy
   * - 20
     - High accuracy
     - Better for trimmed/edited videos
   * - 30+
     - Maximum accuracy
     - Significantly slower

Hash Types (Images)
-------------------

Different hash algorithms have different characteristics:

pHash (Perceptual Hash)
^^^^^^^^^^^^^^^^^^^^^^^

- Based on DCT (Discrete Cosine Transform)
- Best for: Resized images, different compression
- Robust against: Scaling, brightness changes

aHash (Average Hash)
^^^^^^^^^^^^^^^^^^^^

- Compares pixels to average brightness
- Best for: Quick comparisons
- Less robust against: Minor edits

dHash (Difference Hash)
^^^^^^^^^^^^^^^^^^^^^^^

- Tracks gradients between adjacent pixels
- Best for: Detecting structure similarity
- Good for: Cropped images

wHash (Wavelet Hash)
^^^^^^^^^^^^^^^^^^^^

- Uses wavelet transforms
- Best for: Multi-resolution analysis
- Good for: Various image sizes

Combined (Default)
^^^^^^^^^^^^^^^^^^

Uses all four hash types and averages the similarity. Most accurate but slower.

Excluded Folders
----------------

By default, these folder types are excluded:

**System Folders:**

- ``$RECYCLE.BIN``, ``$Recycle``, ``Recycler``
- ``.Trash``, ``.Trashes``
- ``System Volume Information``
- ``lost+found``

**Development Folders:**

- ``.git``, ``.svn``, ``.hg``
- ``__pycache__``, ``.cache``
- ``node_modules``
- ``.venv``, ``venv``

**Media System Folders:**

- ``@eaDir`` (Synology NAS)
- ``.thumbnails``, ``.thumb``, ``thumbs``

GPU Configuration
-----------------

GPU acceleration is automatically detected in this order:

1. **CUDA (NVIDIA)** - PyTorch or CuPy
2. **ROCm (AMD)** - PyTorch with ROCm
3. **OpenCL** - Any GPU with OpenCL support
4. **CPU** - Multiprocessing fallback

Disable GPU
^^^^^^^^^^^

.. code-block:: bash

   # CLI
   dupfinder images /path --no-gpu

   # Python
   finder = ImageDuplicateFinder(use_gpu=False)

Batch Size
^^^^^^^^^^

For GPU processing, batch size controls memory usage:

.. code-block:: bash

   # Larger batch (faster, more memory)
   dupfinder images /path --batch-size 2000

   # Smaller batch (slower, less memory)
   dupfinder images /path --batch-size 500

Worker Configuration
--------------------

Workers control parallel CPU processing:

.. code-block:: bash

   # Match CPU cores
   dupfinder images /path --workers 8

   # Conservative (for I/O bound operations)
   dupfinder images /path --workers 2

**Recommendation:** Set workers to your CPU core count for optimal performance.

Performance Tuning
------------------

For Large Collections
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   finder = ImageDuplicateFinder(
       num_workers=8,          # Increase parallelism
       batch_size=2000,        # Larger GPU batches
       use_gpu=True,           # Enable GPU
   )

For Memory Constrained Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   finder = ImageDuplicateFinder(
       num_workers=2,          # Fewer parallel operations
       batch_size=500,         # Smaller GPU batches
       hash_size=8,            # Smaller hashes
   )

For Maximum Accuracy
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   finder = ImageDuplicateFinder(
       similarity_threshold=0.95,
       hash_size=32,
       hash_type="combined",
   )
