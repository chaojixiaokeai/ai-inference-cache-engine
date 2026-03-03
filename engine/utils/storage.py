"""Storage path management and disk space utilities."""

import os
import shutil
from pathlib import Path


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    p = Path(os.path.expanduser(path))
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_dir_size_bytes(path: str) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    p = Path(path)
    if not p.exists():
        return 0
    for entry in p.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def get_dir_size_mb(path: str) -> float:
    return get_dir_size_bytes(path) / (1024 * 1024)


def get_dir_size_gb(path: str) -> float:
    return get_dir_size_bytes(path) / (1024 ** 3)


def get_free_disk_gb(path: str) -> float:
    """Get free disk space in GB for the filesystem containing path."""
    p = os.path.expanduser(path)
    if not os.path.exists(p):
        p = os.path.expanduser("~")
    usage = shutil.disk_usage(p)
    return usage.free / (1024 ** 3)


def safe_remove_dir(path: str) -> bool:
    """Safely remove a directory and all its contents."""
    p = Path(os.path.expanduser(path))
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
        return True
    return False


def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"
