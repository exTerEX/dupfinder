# DupFinder - Video & Image Duplicate Finder

A Python package to detect duplicate videos and images based on **content similarity**, regardless of resolution, length, format, or encoding differences.

## Features

- **GPU Acceleration**: Automatic detection of CUDA (NVIDIA), ROCm (AMD), or OpenCL
- **Smart Fallback**: Falls back to CPU multiprocessing when no GPU is available
- **Content-Based Detection**: Finds duplicates even with different resolutions, codecs, or quality
- **Multiple Hash Algorithms**: pHash, aHash, dHash, wHash for robust detection
- **Installable Package**: Use as CLI tools or import as a Python library
- **Auto-Excludes System Folders**: Skips `$RECYCLE.BIN`, `.Trash`, `node_modules`, etc.

## Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/video-duplication-identifier.git
cd video-duplication-identifier

# Install with UV
uv pip install .

# Or install in development mode
uv pip install -e .
```

### Using pip

```bash
pip install .
```

### GPU Acceleration (Optional)

Install GPU extras based on your hardware:

```bash
# NVIDIA GPUs (CUDA)
uv pip install ".[cuda]"
# or: pip install torch --index-url https://download.pytorch.org/whl/cu121

# AMD GPUs (ROCm)
uv pip install ".[rocm]"

# OpenCL (any GPU)
uv pip install ".[opencl]"
```

The acceleration priority is: **CUDA → ROCm → OpenCL → CPU multiprocessing**

### Optional: Install FFmpeg

For better video format support:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## CLI Usage

After installation, three commands are available:

### Unified Command

```bash
# Find duplicate images
dupfinder images /path/to/images

# Find duplicate videos
dupfinder videos /path/to/videos
```

### Standalone Commands

```bash
# Find duplicate images
find-duplicate-images /path/to/images

# Find duplicate videos
find-duplicate-videos /path/to/videos
```

### CLI Options

```
Options:
  -t, --threshold FLOAT   Similarity threshold 0-1 (default: 0.90 images, 0.85 videos)
  -s, --hash-size INT     Hash size for perceptual hashing (default: 16)
  -w, --workers INT       Number of parallel workers (default: 4)
  -o, --output FILE       Output JSON file for results
  --no-gpu                Disable GPU acceleration
  --batch-size INT        Batch size for GPU processing (default: 1000)
  -v, --verbose           Enable verbose logging
```

### Examples

```bash
# Find image duplicates with 95% similarity
dupfinder images /path/to/photos --threshold 0.95

# Find video duplicates, save to JSON
dupfinder videos /path/to/videos --output duplicates.json

# Use CPU only (disable GPU)
dupfinder images /path/to/images --no-gpu

# Process with more workers
dupfinder videos /path/to/videos --workers 8
```

## Python API

Import and use the package in your own code:

```python
from dupfinder import ImageDuplicateFinder, VideoDuplicateFinder

# Find duplicate images
finder = ImageDuplicateFinder(threshold=0.90, use_gpu=True)
groups = finder.find_duplicates("/path/to/images")

for group in groups:
    print(f"Duplicate group with {len(group)} images:")
    for img in group:
        print(f"  - {img.path} ({img.width}x{img.height}, {img.file_size} bytes)")

# Find duplicate videos
video_finder = VideoDuplicateFinder(threshold=0.85, use_gpu=True)
video_groups = video_finder.find_duplicates("/path/to/videos")

for group in video_groups:
    print(f"Duplicate group with {len(group)} videos:")
    for vid in group:
        print(f"  - {vid.path} ({vid.width}x{vid.height}, {vid.duration:.1f}s)")
```

### Available Classes

```python
from dupfinder import (
    # Main finder classes
    ImageDuplicateFinder,
    VideoDuplicateFinder,
    
    # Data classes
    ImageInfo,
    VideoInfo,
    
    # Hasher classes (for custom use)
    ImageHasher,
    VideoHasher,
    
    # GPU acceleration
    GPUAccelerator,
    AcceleratorType,
    get_accelerator,
    
    # Constants
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    EXCLUDED_FOLDERS,
)
```

### GPU Accelerator

```python
from dupfinder import get_accelerator, AcceleratorType

# Get the best available accelerator
accel = get_accelerator()
print(f"Using: {accel.accelerator_type.name}")
print(f"Device: {accel.device_name}")

# Check accelerator type
if accel.accelerator_type == AcceleratorType.CUDA:
    print("Running on NVIDIA GPU")
elif accel.accelerator_type == AcceleratorType.CPU:
    print("Running on CPU with multiprocessing")
```

## How It Works

### Video Duplicate Detection

1. **Frame Extraction**: Extracts 10 frames at regular intervals (skipping 5% intro/outro)
2. **Perceptual Hashing**: Computes pHash for each frame
3. **Similarity Comparison**: Compares frame hashes using Hamming distance
4. **Duplicate Grouping**: Groups videos exceeding the similarity threshold

### Image Duplicate Detection

1. **Hash Computation**: Computes multiple perceptual hashes (pHash, aHash, dHash, wHash)
2. **Hash Bucketing**: Groups images with identical hashes
3. **Near-Duplicate Detection**: GPU-accelerated similarity matrix for remaining images
4. **Duplicate Grouping**: Groups images exceeding the similarity threshold

## Supported Formats

### Images
JPEG, PNG, GIF, BMP, TIFF, WebP, HEIC, HEIF, ICO, SVG, and RAW formats (CR2, NEF, ARW, DNG, ORF, RW2, PEF, SRW)

### Videos
MP4, AVI, MKV, MOV, WMV, FLV, WebM, M4V, MPEG, MPG, 3GP, OGV, TS, MTS, M2TS, VOB, DIVX, XVID, ASF, RM, RMVB

## Output Example

```
================================================================================
DUPLICATE IMAGE GROUPS FOUND: 3 (7 duplicate files)
================================================================================

Group 1:
----------------------------------------
  * /photos/vacation_original.jpg
    Resolution: 4000x3000
    Format: JPEG
    Size: 5.20 MB
    Similarity: 100.0%

  * /photos/vacation_resized.jpg
    Resolution: 1920x1440
    Format: JPEG
    Size: 1.10 MB
    Similarity: 98.5%

================================================================================
Potential space savings by removing duplicates: 3.50 MB
================================================================================
```

## Project Structure

```
video-duplication-identifier/
├── pyproject.toml          # Package configuration
├── README.md
├── src/
│   └── dupfinder/
│       ├── __init__.py     # Package exports
│       ├── accelerator.py  # GPU acceleration
│       ├── cli.py          # CLI entry points
│       ├── constants.py    # Shared constants
│       ├── images.py       # Image duplicate finder
│       ├── utils.py        # Utility functions
│       └── videos.py       # Video duplicate finder
└── tests/
    └── __init__.py
```

## Performance Tips

1. **Threshold**: Lower (0.75) finds more duplicates but may have false positives. Higher (0.95) is more strict.
2. **Workers**: Match to your CPU core count for optimal performance.
3. **GPU**: Significantly faster for large collections (1000+ files).
4. **Batch Size**: Adjust `--batch-size` based on GPU memory.

## License

MIT License - Feel free to use and modify as needed.
