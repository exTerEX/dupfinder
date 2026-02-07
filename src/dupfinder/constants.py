"""
Shared constants for dupfinder.
"""

# Supported image extensions
IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".heic",
        ".heif",
        ".ico",
        ".svg",
        ".raw",
        ".cr2",
        ".nef",
        ".arw",
        ".dng",
        ".orf",
        ".rw2",
        ".pef",
        ".srw",
    }
)

# Supported video extensions
VIDEO_EXTENSIONS = frozenset(
    {
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".mpeg",
        ".mpg",
        ".3gp",
        ".ogv",
        ".ts",
        ".mts",
        ".m2ts",
        ".vob",
        ".divx",
        ".xvid",
        ".asf",
        ".rm",
        ".rmvb",
    }
)

# Folders to exclude from scanning
EXCLUDED_FOLDERS = frozenset(
    {
        # Windows
        "$recycle.bin",
        "$recycle",
        "recycler",
        "recycled",
        "system volume information",
        "windows",
        "appdata",
        # macOS
        ".trash",
        ".trashes",
        ".spotlight-v100",
        ".fseventsd",
        ".ds_store",
        # Linux
        "lost+found",
        "trash",
        # Thumbnails
        ".thumbnails",
        ".thumb",
        "thumbs",
        # NAS
        "@eadir",
        # Version control
        ".git",
        ".svn",
        ".hg",
        # Development
        "__pycache__",
        ".cache",
        "node_modules",
        ".venv",
        "venv",
    }
)

# Default configuration values
DEFAULT_IMAGE_THRESHOLD = 0.90
DEFAULT_VIDEO_THRESHOLD = 0.85
DEFAULT_HASH_SIZE = 16
DEFAULT_NUM_FRAMES = 10
DEFAULT_NUM_WORKERS = 4
DEFAULT_BATCH_SIZE = 1000
