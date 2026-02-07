Videos Module
=============

.. module:: dupfinder.videos
   :synopsis: Video duplicate detection

The videos module provides functionality for detecting duplicate videos
using perceptual hashing of extracted frames.

Classes
-------

VideoInfo
^^^^^^^^^

.. autoclass:: dupfinder.videos.VideoInfo
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

VideoHasher
^^^^^^^^^^^

.. autoclass:: dupfinder.videos.VideoHasher
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

VideoDuplicateFinder
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dupfinder.videos.VideoDuplicateFinder
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Functions
---------

.. autofunction:: dupfinder.videos.print_results

Usage Examples
--------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from dupfinder import VideoDuplicateFinder

   # Create finder with default settings
   finder = VideoDuplicateFinder()

   # Find videos in folder
   videos = finder.find_videos("/path/to/videos")
   print(f"Found {len(videos)} videos")

   # Process and compute hashes
   finder.process_videos(videos)

   # Find duplicates
   groups = finder.find_duplicates()

   for group in groups:
       print(f"Group with {len(group)} duplicates")

Custom Settings
^^^^^^^^^^^^^^^

.. code-block:: python

   finder = VideoDuplicateFinder(
       similarity_threshold=0.90,  # Stricter matching
       num_frames=20,              # More frames for accuracy
       hash_size=32,               # Higher precision hashes
       num_workers=8,              # More parallelism
       use_gpu=True,               # Enable GPU
   )

Using VideoHasher Directly
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from dupfinder.videos import VideoHasher, VideoInfo

   hasher = VideoHasher(num_frames=10, hash_size=16)

   # Extract frames and get video info
   frames, info = hasher.extract_frames("/path/to/video.mp4")
   print(f"Duration: {info.duration}s")
   print(f"Resolution: {info.width}x{info.height}")
   print(f"FPS: {info.fps}")
   print(f"Extracted {len(frames)} frames")

   # Compute hash for a single frame
   if frames:
       frame_hash = hasher.compute_frame_hash(frames[0])
       print(f"Frame hash: {frame_hash}")

   # Or compute all hashes at once
   video_info = hasher.compute_video_hashes("/path/to/video.mp4")
   print(f"Frame hashes: {len(video_info.frame_hashes)}")
   print(f"Average hash: {video_info.average_hash}")

Comparing Videos Manually
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   finder = VideoDuplicateFinder()

   # Process videos
   finder.process_videos(["/path/video1.mp4", "/path/video2.mp4"])

   # Get info objects
   info1 = finder.video_infos["/path/video1.mp4"]
   info2 = finder.video_infos["/path/video2.mp4"]

   # Compute similarity
   similarity = finder.compute_similarity(info1, info2)
   print(f"Similarity: {similarity*100:.1f}%")

How Video Hashing Works
-----------------------

The video duplicate detection process:

1. **Frame Extraction**: Extracts N frames at regular intervals, skipping
   the first and last 5% (to avoid intros/outros).

2. **Perceptual Hashing**: Computes pHash for each extracted frame.

3. **Average Hash**: Combines frame hashes into a single representative hash.

4. **Similarity Comparison**: Compares frame hashes between video pairs
   using Hamming distance.

5. **Grouping**: Groups videos that exceed the similarity threshold.

This approach can detect duplicates even when videos have:

- Different resolutions
- Different lengths (trimmed versions)
- Different codecs/encodings
- Different bitrates/quality
- Minor edits or watermarks
