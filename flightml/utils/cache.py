"""
FlightML Persistent Disk Caching and Hashing Engine.

Eliminates repetitive computationally intensive recalculations across historical
Eurocontrol archives, terrestrial radio horizon evaluations, kinematic WRAP analysis,
and optimization routines.

Features:
- Deterministic SHA-256 hash key generation across inputs, periods, and parameters
- Atomic write-replace serialization using pickle with highest protocol
- Cache statistics reporting (disk usage, entry count, namespaces)
- One-click / programmatic cache invalidation and reset
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Root cache directory for FlightML evaluations and analysis
CACHE_ROOT = Path(".cache") / "analysis"


def get_cache_root() -> Path:
    """Ensure and return the root directory for persistent disk caching."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT


def _normalize_for_hash(val: Any) -> Any:
    """Convert arbitrary Python arguments into JSON-serializable deterministic structures."""
    if val is None or isinstance(val, (int, float, str, bool)):
        return val
    elif isinstance(val, (list, tuple, set)):
        return [_normalize_for_hash(x) for x in sorted(list(val), key=str)]
    elif isinstance(val, dict):
        return {str(k): _normalize_for_hash(v) for k, v in sorted(val.items(), key=lambda item: str(item[0]))}
    elif hasattr(val, "period"):  # e.g. FlightDataset instance
        return f"FlightDataset(period={getattr(val, 'period')})"
    elif hasattr(val, "period_id"):
        return f"Dataset(period_id={getattr(val, 'period_id')})"
    elif hasattr(val, "to_dict"):
        return _normalize_for_hash(val.to_dict())
    else:
        return f"{type(val).__name__}:{repr(val)}"


def compute_cache_key(func_name: str, *args: Any, **kwargs: Any) -> str:
    """
    Generate a deterministic SHA-256 hex digest key for a function invocation.
    """
    normalized_args = [_normalize_for_hash(a) for a in args]
    normalized_kwargs = {str(k): _normalize_for_hash(v) for k, v in sorted(kwargs.items())}

    payload = {
        "func": func_name,
        "args": normalized_args,
        "kwargs": normalized_kwargs,
    }
    raw_json = json.dumps(payload, sort_keys=True)
    hash_digest = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    return f"{func_name}_{hash_digest[:20]}"


def disk_cache(namespace: str = "eval", enabled: bool = True) -> Callable[[F], F]:
    """
    Decorator for caching deterministic analytics and evaluations to disk.
    If a cached result exists for the computed SHA-256 hash, loads and returns it
    immediately without rerunning the underlying algorithm.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enabled:
                return func(*args, **kwargs)

            ns_dir = get_cache_root() / namespace
            ns_dir.mkdir(parents=True, exist_ok=True)

            cache_key = compute_cache_key(func.__name__, *args, **kwargs)
            cache_file = ns_dir / f"{cache_key}.pkl"

            # Check if cached file exists
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    # In case of corruption, fall back to recomputing
                    pass

            # Compute result
            result = func(*args, **kwargs)

            # Atomic write to disk cache
            try:
                temp_fd, temp_path = tempfile.mkstemp(dir=str(ns_dir), prefix="tmp_cache_")
                with os.fdopen(temp_fd, "wb") as f:
                    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temp_path, cache_file)
            except Exception:
                # If writing to cache fails, still return the computed result safely
                pass

            return result

        return wrapper  # type: ignore

    return decorator


def clear_disk_cache(namespace: str | None = None) -> int:
    """
    Clear all cached analysis objects or only those within a specific namespace.
    Returns the count of deleted cache files.
    """
    root = get_cache_root()
    target_dir = root / namespace if namespace else root
    if not target_dir.exists():
        return 0

    count = 0
    for root_path, _, files in os.walk(target_dir):
        for f in files:
            if f.endswith(".pkl"):
                try:
                    os.remove(os.path.join(root_path, f))
                    count += 1
                except Exception:
                    pass
    return count


def get_disk_cache_stats() -> dict[str, Any]:
    """
    Retrieve current disk cache statistics: total entries, storage in MB, and namespaces.
    """
    root = get_cache_root()
    if not root.exists():
        return {"total_files": 0, "total_size_mb": 0.0, "namespaces": {}}

    total_files = 0
    total_bytes = 0
    namespaces: dict[str, dict[str, Any]] = {}

    for ns_path in root.iterdir():
        if ns_path.is_dir():
            ns_files = 0
            ns_bytes = 0
            for f in ns_path.glob("*.pkl"):
                try:
                    sz = f.stat().st_size
                    ns_files += 1
                    ns_bytes += sz
                except Exception:
                    pass
            total_files += ns_files
            total_bytes += ns_bytes
            namespaces[ns_path.name] = {
                "files": ns_files,
                "size_mb": round(ns_bytes / (1024 * 1024), 2),
            }

    return {
        "total_files": total_files,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "namespaces": namespaces,
    }
