dupfinder Documentation
=======================

**dupfinder** is a Python package for detecting duplicate images and videos based on 
content similarity using perceptual hashing, with optional GPU acceleration.

.. image:: https://img.shields.io/pypi/v/dupfinder.svg
   :target: https://pypi.org/project/dupfinder/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/dupfinder.svg
   :target: https://pypi.org/project/dupfinder/
   :alt: Python versions

Features
--------

- **Content-Based Detection**: Finds duplicates even with different resolutions, codecs, or quality
- **Multiple Hash Algorithms**: pHash, aHash, dHash, wHash for robust detection  
- **GPU Acceleration**: Automatic detection of CUDA, ROCm, or OpenCL
- **Smart Fallback**: Falls back to CPU multiprocessing when no GPU is available
- **CLI & API**: Use as command-line tools or import as a Python library
- **Auto-Excludes System Folders**: Skips ``$RECYCLE.BIN``, ``.Trash``, ``node_modules``, etc.

Quick Start
-----------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   # Using UV (recommended)
   uv pip install .

   # Using pip
   pip install .

   # With GPU support (CUDA)
   uv pip install ".[cuda]"

Command Line Usage
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Find duplicate images
   dupfinder images /path/to/images

   # Find duplicate videos
   dupfinder videos /path/to/videos

   # With custom threshold
   dupfinder images /path/to/images --threshold 0.95

Python API
^^^^^^^^^^

.. code-block:: python

   from dupfinder import ImageDuplicateFinder, VideoDuplicateFinder

   # Find duplicate images
   finder = ImageDuplicateFinder(similarity_threshold=0.90)
   images = finder.find_images("/path/to/images")
   finder.process_images(images)
   duplicates = finder.find_duplicates_fast()

   for group in duplicates:
       print(f"Found {len(group)} duplicates")

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   cli
   configuration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
