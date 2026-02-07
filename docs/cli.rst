Command Line Interface
======================

dupfinder provides several command-line tools for finding duplicate media files.

Commands Overview
-----------------

After installation, three commands are available:

- ``dupfinder`` - Unified CLI with subcommands
- ``find-duplicate-images`` - Standalone image finder
- ``find-duplicate-videos`` - Standalone video finder

Unified CLI (dupfinder)
-----------------------

The main entry point with subcommands for images and videos.

.. code-block:: bash

   dupfinder {images,videos} [options] folder

Images Subcommand
^^^^^^^^^^^^^^^^^

Find duplicate images in a folder.

.. code-block:: bash

   dupfinder images /path/to/images [options]

**Options:**

.. option:: folder

   Folder to scan for images (searches recursively). Required.

.. option:: -t, --threshold <float>

   Similarity threshold (0-1) to consider as duplicates. Default: 0.90

.. option:: -s, --hash-size <int>

   Hash size for perceptual hashing. Larger = more precise. Default: 16

.. option:: -w, --workers <int>

   Number of parallel workers for processing. Default: 4

.. option:: --hash-type {phash,ahash,dhash,whash,combined}

   Hash algorithm to use. Default: combined

.. option:: -o, --output <file>

   Output JSON file for results.

.. option:: --no-gpu

   Disable GPU acceleration (use CPU only).

.. option:: --batch-size <int>

   Batch size for GPU processing. Default: 1000

.. option:: -v, --verbose

   Enable verbose logging (debug level).

**Examples:**

.. code-block:: bash

   # Basic usage
   dupfinder images /photos

   # High precision matching
   dupfinder images /photos --threshold 0.95 --hash-size 32

   # Use only pHash (faster, less accurate)
   dupfinder images /photos --hash-type phash

   # Save results and use verbose output
   dupfinder images /photos -o results.json -v

Videos Subcommand
^^^^^^^^^^^^^^^^^

Find duplicate videos in a folder.

.. code-block:: bash

   dupfinder videos /path/to/videos [options]

**Options:**

.. option:: folder

   Folder to scan for videos (searches recursively). Required.

.. option:: -t, --threshold <float>

   Similarity threshold (0-1) to consider as duplicates. Default: 0.85

.. option:: -f, --frames <int>

   Number of frames to extract per video. More = more accurate but slower. Default: 10

.. option:: -s, --hash-size <int>

   Hash size for perceptual hashing. Default: 16

.. option:: -w, --workers <int>

   Number of parallel workers for processing. Default: 4

.. option:: -o, --output <file>

   Output JSON file for results.

.. option:: --no-gpu

   Disable GPU acceleration (use CPU only).

.. option:: -v, --verbose

   Enable verbose logging.

**Examples:**

.. code-block:: bash

   # Basic usage
   dupfinder videos /movies

   # More accurate matching (slower)
   dupfinder videos /movies --frames 20 --threshold 0.90

   # Fast processing with more workers
   dupfinder videos /movies -w 8 -f 5

Standalone Commands
-------------------

find-duplicate-images
^^^^^^^^^^^^^^^^^^^^^

Standalone command for finding duplicate images.

.. code-block:: bash

   find-duplicate-images /path/to/images [options]

All options are the same as ``dupfinder images``.

find-duplicate-videos
^^^^^^^^^^^^^^^^^^^^^

Standalone command for finding duplicate videos.

.. code-block:: bash

   find-duplicate-videos /path/to/videos [options]

All options are the same as ``dupfinder videos``.

Exit Codes
----------

======  ==========================================
Code    Description
======  ==========================================
0       Success
1       Error (invalid arguments, folder not found)
======  ==========================================

Environment Variables
---------------------

The tools respect the following environment variables:

``CUDA_VISIBLE_DEVICES``
   Control which GPUs are visible to CUDA.

``OMP_NUM_THREADS``
   Number of OpenMP threads for CPU operations.

Tips and Tricks
---------------

Processing Large Collections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For very large media collections (10,000+ files):

.. code-block:: bash

   # Use GPU acceleration
   dupfinder images /photos --batch-size 2000

   # Increase workers for CPU processing
   dupfinder images /photos --no-gpu --workers 8

   # Save results for later analysis
   dupfinder images /photos -o results.json

Optimizing Accuracy vs Speed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Higher Accuracy (slower):**

.. code-block:: bash

   dupfinder images /photos --threshold 0.95 --hash-size 32
   dupfinder videos /videos --frames 20 --threshold 0.90

**Faster Processing (may miss some):**

.. code-block:: bash

   dupfinder images /photos --threshold 0.80 --hash-type phash
   dupfinder videos /videos --frames 5 --threshold 0.80

Combining with Other Tools
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Process specific folders
   find /data -type d -name "photos" | xargs -I{} dupfinder images {}

   # Filter JSON results with jq
   dupfinder images /photos -o results.json
   cat results.json | jq '.duplicate_groups[] | .[].path'
