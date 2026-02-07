Utilities Module
================

.. module:: dupfinder.utils
   :synopsis: Shared utility functions

The utils module provides shared utility functions used across dupfinder.

Functions
---------

.. autofunction:: dupfinder.utils.format_file_size

.. autofunction:: dupfinder.utils.format_duration

.. autofunction:: dupfinder.utils.is_excluded_path

.. autofunction:: dupfinder.utils.find_files_by_extension

.. autofunction:: dupfinder.utils.setup_logging

Usage Examples
--------------

Formatting File Sizes
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from dupfinder.utils import format_file_size

   print(format_file_size(1024))        # "1.00 KB"
   print(format_file_size(1048576))     # "1.00 MB"
   print(format_file_size(1073741824))  # "1.00 GB"

Formatting Durations
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from dupfinder.utils import format_duration

   print(format_duration(45))     # "45s"
   print(format_duration(90))     # "1m 30s"
   print(format_duration(3661))   # "1h 1m 1s"

Checking Excluded Paths
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from dupfinder.utils import is_excluded_path

   # Default exclusions
   print(is_excluded_path("/photos/vacation.jpg"))           # False
   print(is_excluded_path("/$RECYCLE.BIN/deleted.jpg"))      # True
   print(is_excluded_path("/project/.git/objects/file"))     # True

   # Custom exclusions
   custom = {"backup", "temp"}
   print(is_excluded_path("/data/backup/file.jpg", custom))  # True

Finding Files
^^^^^^^^^^^^^

.. code-block:: python

   from dupfinder.utils import find_files_by_extension
   from dupfinder.constants import IMAGE_EXTENSIONS

   # Find all images in a folder
   images = find_files_by_extension(
       "/path/to/photos",
       IMAGE_EXTENSIONS,
   )
   print(f"Found {len(images)} images")

   # Find specific file types
   pdfs = find_files_by_extension("/documents", {".pdf"})
