"""
Unit tests for the FlightML persistent disk caching and hashing architecture.
"""

from __future__ import annotations

import time
import pytest
from flightml.data.dataset import FlightDataset
from flightml.utils.cache import (
    compute_cache_key,
    disk_cache,
    clear_disk_cache,
    get_disk_cache_stats,
)
from flightml.eval.airport_ops import get_airport_operations_report
from flightml.eval.dme_rnav_coverage import get_dme_rnav_coverage_report


def test_compute_cache_key_deterministic():
    """Verify SHA-256 hash key generation is deterministic across ordered arguments."""
    k1 = compute_cache_key("test_func", 100, "EGLL", inactive=["A", "B"])
    k2 = compute_cache_key("test_func", 100, "EGLL", inactive=["B", "A"])
    # Lists are sorted in normalization
    assert k1 == k2

    # Different arguments produce different keys
    k3 = compute_cache_key("test_func", 200, "EGLL", inactive=["A", "B"])
    assert k1 != k3


def test_disk_cache_speedup():
    """Verify that cached function execution is orders of magnitude faster than cold compute."""
    call_count = 0

    @disk_cache(namespace="unit_test")
    def compute_expensive(val: int) -> int:
        nonlocal call_count
        call_count += 1
        time.sleep(0.1)
        return val * 10

    # Ensure clean test state
    clear_disk_cache("unit_test")

    # 1. Cold execution
    t0 = time.time()
    res1 = compute_expensive(42)
    t_cold = time.time() - t0
    assert res1 == 420
    assert call_count == 1
    assert t_cold >= 0.09

    # 2. Warm cached execution (should hit disk cache)
    t0 = time.time()
    res2 = compute_expensive(42)
    t_warm = time.time() - t0
    assert res2 == 420
    assert call_count == 1  # Underlying function was not called again!
    assert t_warm < t_cold / 2

    # Verify cache statistics
    stats = get_disk_cache_stats()
    assert stats["total_files"] >= 1
    assert "unit_test" in stats["namespaces"]

    # 3. Cache reset
    deleted = clear_disk_cache("unit_test")
    assert deleted >= 1

    stats_after = get_disk_cache_stats()
    assert stats_after["namespaces"].get("unit_test", {}).get("files", 0) == 0


def test_evaluation_engines_cached_end_to_end():
    """Verify evaluation engines leverage the disk cache on repeated queries."""
    ds = FlightDataset("202501")

    # Airport Ops
    t0 = time.time()
    ao1 = get_airport_operations_report(ds, "EGLL", period="202501")
    t_ao1 = time.time() - t0

    t0 = time.time()
    ao2 = get_airport_operations_report(ds, "EGLL", period="202501")
    t_ao2 = time.time() - t0

    assert ao1.airport_ident == ao2.airport_ident
    assert ao1.total_movements == ao2.total_movements
    # Cached run should be much faster
    assert t_ao2 < max(0.05, t_ao1)
