Constants Module
================

.. module:: dupfinder.constants
   :synopsis: Shared constants and default values

The constants module contains shared constants used throughout dupfinder.

File Extensions
---------------

IMAGE_EXTENSIONS
^^^^^^^^^^^^^^^^

.. py:data:: IMAGE_EXTENSIONS
   :type: frozenset[str]

   Supported image file extensions.

   Includes common formats:

   - JPEG: ``.jpg``, ``.jpeg``
   - PNG: ``.png``
   - GIF: ``.gif``
   - BMP: ``.bmp``
   - TIFF: ``.tiff``, ``.tif``
   - WebP: ``.webp``
   - HEIC/HEIF: ``.heic``, ``.heif``

   And RAW camera formats:

   - Canon: ``.cr2``
   - Nikon: ``.nef``
   - Sony: ``.arw``
   - Adobe: ``.dng``
   - And more...

VIDEO_EXTENSIONS
^^^^^^^^^^^^^^^^

.. py:data:: VIDEO_EXTENSIONS
   :type: frozenset[str]

   Supported video file extensions.

   Includes:

   - MP4: ``.mp4``, ``.m4v``
   - AVI: ``.avi``
   - MKV: ``.mkv``
   - MOV: ``.mov``
   - WebM: ``.webm``
   - Windows Media: ``.wmv``, ``.asf``
   - And more...

Excluded Folders
----------------

EXCLUDED_FOLDERS
^^^^^^^^^^^^^^^^

.. py:data:: EXCLUDED_FOLDERS
   :type: frozenset[str]

   Folder names to exclude from scanning.

   **System folders:**

   - Windows: ``$recycle.bin``, ``system volume information``
   - macOS: ``.trash``, ``.trashes``, ``.spotlight-v100``
   - Linux: ``lost+found``, ``trash``

   **Development folders:**

   - Version control: ``.git``, ``.svn``, ``.hg``
   - Build artifacts: ``__pycache__``, ``.cache``
   - Dependencies: ``node_modules``, ``.venv``, ``venv``

   **Media system folders:**

   - NAS: ``@eadir`` (Synology)
   - Thumbnails: ``.thumbnails``, ``.thumb``

Default Configuration
---------------------

DEFAULT_IMAGE_THRESHOLD
^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: DEFAULT_IMAGE_THRESHOLD
   :type: float
   :value: 0.90

   Default similarity threshold for image duplicate detection.
   Images must be at least 90% similar to be considered duplicates.

DEFAULT_VIDEO_THRESHOLD
^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: DEFAULT_VIDEO_THRESHOLD
   :type: float
   :value: 0.85

   Default similarity threshold for video duplicate detection.
   Videos need a lower threshold due to encoding variations.

DEFAULT_HASH_SIZE
^^^^^^^^^^^^^^^^^

.. py:data:: DEFAULT_HASH_SIZE
   :type: int
   :value: 16

   Default hash size for perceptual hashing.
   Results in 256-bit hashes (16x16).

DEFAULT_NUM_FRAMES
^^^^^^^^^^^^^^^^^^

.. py:data:: DEFAULT_NUM_FRAMES
   :type: int
   :value: 10

   Default number of frames to extract from each video for comparison.

DEFAULT_NUM_WORKERS
^^^^^^^^^^^^^^^^^^^

.. py:data:: DEFAULT_NUM_WORKERS
   :type: int
   :value: 4

   Default number of parallel workers for processing.

DEFAULT_BATCH_SIZE
^^^^^^^^^^^^^^^^^^

.. py:data:: DEFAULT_BATCH_SIZE
   :type: int
   :value: 1000

   Default batch size for GPU processing operations.

Usage
-----

.. code-block:: python

   from dupfinder.constants import (
       IMAGE_EXTENSIONS,
       VIDEO_EXTENSIONS,
       EXCLUDED_FOLDERS,
       DEFAULT_IMAGE_THRESHOLD,
       DEFAULT_VIDEO_THRESHOLD,
   )

   # Check if file is an image
   ext = ".jpg"
   if ext.lower() in IMAGE_EXTENSIONS:
       print("This is an image file")

   # Check if path should be excluded
   path_parts = ["photos", "$RECYCLE.BIN", "deleted.jpg"]
   if any(part.lower() in EXCLUDED_FOLDERS for part in path_parts):
       print("This path should be excluded")

   # Use default thresholds
   print(f"Image threshold: {DEFAULT_IMAGE_THRESHOLD}")
   print(f"Video threshold: {DEFAULT_VIDEO_THRESHOLD}")
