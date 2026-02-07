API Reference
=============

This section contains the complete API reference for dupfinder.

Main Classes
------------

The primary classes for finding duplicates:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   dupfinder.ImageDuplicateFinder
   dupfinder.VideoDuplicateFinder

Data Classes
------------

Classes for storing file information:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   dupfinder.ImageInfo
   dupfinder.VideoInfo

Hasher Classes
--------------

Low-level hashing functionality:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   dupfinder.ImageHasher
   dupfinder.VideoHasher

GPU Acceleration
----------------

GPU acceleration components:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   dupfinder.GPUAccelerator
   dupfinder.AcceleratorType
   dupfinder.get_accelerator

Module Index
------------

.. toctree::
   :maxdepth: 2

   images
   videos
   accelerator
   utils
   constants

Quick Reference
---------------

Finding Duplicate Images
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from dupfinder import ImageDuplicateFinder

   finder = ImageDuplicateFinder(similarity_threshold=0.90)
   images = finder.find_images("/path/to/images")
   finder.process_images(images)
   groups = finder.find_duplicates_fast()

Finding Duplicate Videos
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from dupfinder import VideoDuplicateFinder

   finder = VideoDuplicateFinder(similarity_threshold=0.85)
   videos = finder.find_videos("/path/to/videos")
   finder.process_videos(videos)
   groups = finder.find_duplicates()

Using GPU Accelerator
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from dupfinder import get_accelerator

   accel = get_accelerator()
   print(f"Using: {accel.get_backend_name()}")
   print(f"GPU available: {accel.is_gpu_available}")
