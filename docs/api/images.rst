Images Module
=============

.. module:: dupfinder.images
   :synopsis: Image duplicate detection

The images module provides functionality for detecting duplicate images
using perceptual hashing.

Classes
-------

ImageInfo
^^^^^^^^^

.. autoclass:: dupfinder.images.ImageInfo
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

ImageHasher
^^^^^^^^^^^

.. autoclass:: dupfinder.images.ImageHasher
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

ImageDuplicateFinder
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dupfinder.images.ImageDuplicateFinder
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Functions
---------

.. autofunction:: dupfinder.images.print_results

Usage Examples
--------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from dupfinder import ImageDuplicateFinder

   # Create finder with default settings
   finder = ImageDuplicateFinder()

   # Find images in folder
   images = finder.find_images("/path/to/photos")
   print(f"Found {len(images)} images")

   # Process and compute hashes
   finder.process_images(images)

   # Find duplicates
   groups = finder.find_duplicates_fast()

   for group in groups:
       print(f"Group with {len(group)} duplicates")

Custom Settings
^^^^^^^^^^^^^^^

.. code-block:: python

   finder = ImageDuplicateFinder(
       similarity_threshold=0.95,  # Stricter matching
       hash_size=32,               # Higher precision
       hash_type="phash",          # pHash only (faster)
       num_workers=8,              # More parallelism
       use_gpu=True,               # Enable GPU
   )

Using ImageHasher Directly
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from dupfinder.images import ImageHasher, ImageInfo

   hasher = ImageHasher(hash_size=16)
   info = hasher.compute_hashes("/path/to/image.jpg")

   print(f"pHash: {info.phash}")
   print(f"aHash: {info.ahash}")
   print(f"dHash: {info.dhash}")
   print(f"wHash: {info.whash}")
   print(f"Size: {info.width}x{info.height}")

Accessing Processed Data
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   finder = ImageDuplicateFinder()
   images = finder.find_images("/path/to/photos")
   finder.process_images(images)

   # Access processed image info
   for path, info in finder.image_infos.items():
       print(f"{path}:")
       print(f"  Resolution: {info.width}x{info.height}")
       print(f"  Format: {info.format}")
       print(f"  Size: {info.file_size} bytes")
