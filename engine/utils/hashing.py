"""Fast hashing utilities based on xxhash64 for cache key generation."""

import hashlib
from pathlib import Path
from typing import List, Optional

import xxhash


def hash_token_sequence(tokens: List[int]) -> int:
    """Generate a 64-bit hash from a token sequence for exact cache matching."""
    h = xxhash.xxh64()
    h.update(bytes(len(tokens).to_bytes(8, "little")))
    for t in tokens:
        h.update(t.to_bytes(4, "little", signed=True))
    return h.intdigest()


def hash_bytes(data: bytes) -> int:
    """Generate a 64-bit xxhash from raw bytes."""
    return xxhash.xxh64(data).intdigest()


def sha256_file(filepath: str, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hash of a file for model integrity verification."""
    h = hashlib.sha256()
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


_model_hash_cache: dict[str, str] = {}


def get_model_hash(filepath: str) -> str:
    """Get model file SHA-256 hash with caching to avoid repeated computation."""
    filepath = str(Path(filepath).resolve())
    if filepath not in _model_hash_cache:
        _model_hash_cache[filepath] = sha256_file(filepath)
    return _model_hash_cache[filepath]


def make_cache_key(model_hash: str, context_hash: int) -> bytes:
    """Create a composite cache key combining model hash and context hash."""
    return f"{model_hash}:{context_hash}".encode("utf-8")
