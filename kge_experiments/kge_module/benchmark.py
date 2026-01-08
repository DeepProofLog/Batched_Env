"""
KGE Module Performance Benchmarking.

Collects timing statistics for KGE integration modules to measure overhead.
Can be enabled via config to track performance impact of each module.

Usage:
    from kge_module.benchmark import KGEBenchmark, get_benchmark

    # Create or get global benchmark
    bench = get_benchmark()

    # Time an operation
    with bench.time("kge_inference"):
        scores = kge_engine.predict_batch(atoms)

    # Print report at end of training
    bench.print_report(total_training_time=training_seconds)
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional
from functools import wraps

import torch


class KGEBenchmark:
    """Collect timing statistics for KGE modules.

    Thread-safe timing collection with per-module statistics.
    Supports both context manager and decorator usage patterns.

    Attributes:
        timings: Dict mapping module names to list of elapsed times.
        call_counts: Dict mapping module names to call counts.
        enabled: Whether benchmarking is active.
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize benchmark.

        Args:
            enabled: Whether to collect timings (can be disabled for production).
        """
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = {}
        self.call_counts: Dict[str, int] = {}
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Mark the start of benchmarked training."""
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Mark the end of benchmarked training.

        Returns:
            Total elapsed time in seconds.
        """
        if self._start_time is None:
            return 0.0
        elapsed = time.perf_counter() - self._start_time
        self._start_time = None
        return elapsed

    @contextmanager
    def time(self, name: str) -> Generator[None, None, None]:
        """Context manager for timing a block.

        Args:
            name: Name of the module/operation being timed.

        Yields:
            None
        """
        if not self.enabled:
            yield
            return

        # Sync CUDA before timing for accurate GPU measurements
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            self.timings.setdefault(name, []).append(elapsed)
            self.call_counts[name] = self.call_counts.get(name, 0) + 1

    def timed(self, name: str):
        """Decorator for timing a function.

        Args:
            name: Name for the timing category.

        Returns:
            Decorated function.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.time(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a module.

        Args:
            name: Module name.

        Returns:
            Dict with total, mean, min, max, count statistics.
        """
        times = self.timings.get(name, [])
        if not times:
            return {"total": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        return {
            "total": sum(times),
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "count": len(times),
        }

    def get_total_overhead(self) -> float:
        """Get total time spent in all timed operations.

        Returns:
            Total overhead in seconds.
        """
        return sum(sum(times) for times in self.timings.values())

    def report(self) -> Dict[str, Dict[str, float]]:
        """Generate full benchmark report.

        Returns:
            Dict mapping module names to their statistics.
        """
        return {name: self.get_stats(name) for name in sorted(self.timings.keys())}

    def print_report(
        self,
        total_training_time: Optional[float] = None,
        print_fn=print,
    ) -> None:
        """Print formatted benchmark report.

        Args:
            total_training_time: Optional total training time for overhead %.
            print_fn: Print function to use (default: print).
        """
        if not self.timings:
            print_fn("[KGEBenchmark] No timings collected")
            return

        print_fn("\n" + "=" * 60)
        print_fn("KGE Module Performance Report")
        print_fn("=" * 60)

        total_overhead = self.get_total_overhead()

        # Sort by total time (descending)
        sorted_modules = sorted(
            self.timings.keys(),
            key=lambda n: sum(self.timings[n]),
            reverse=True,
        )

        for name in sorted_modules:
            stats = self.get_stats(name)
            mean_ms = stats["mean"] * 1000
            total_s = stats["total"]
            count = stats["count"]

            print_fn(f"  {name:25s}: {total_s:7.2f}s ({mean_ms:6.2f}ms/call, {count:6d} calls)")

        print_fn("-" * 60)
        print_fn(f"  {'Total KGE overhead':25s}: {total_overhead:7.2f}s")

        if total_training_time is not None and total_training_time > 0:
            pct = 100.0 * total_overhead / total_training_time
            print_fn(f"  {'Overhead percentage':25s}: {pct:7.2f}% of training time")

        print_fn("=" * 60 + "\n")

    def reset(self) -> None:
        """Clear all collected timings."""
        self.timings.clear()
        self.call_counts.clear()
        self._start_time = None


# Global benchmark instance
_global_benchmark: Optional[KGEBenchmark] = None


def get_benchmark(enabled: bool = True) -> KGEBenchmark:
    """Get or create global benchmark instance.

    Args:
        enabled: Whether benchmarking should be enabled.

    Returns:
        Global KGEBenchmark instance.
    """
    global _global_benchmark
    if _global_benchmark is None:
        _global_benchmark = KGEBenchmark(enabled=enabled)
    return _global_benchmark


def set_benchmark(benchmark: Optional[KGEBenchmark]) -> None:
    """Set the global benchmark instance.

    Args:
        benchmark: Benchmark instance to use, or None to reset.
    """
    global _global_benchmark
    _global_benchmark = benchmark


def create_benchmark(config: Any) -> Optional[KGEBenchmark]:
    """Create benchmark from config.

    Args:
        config: TrainConfig with benchmark settings.

    Returns:
        KGEBenchmark if enabled, None otherwise.
    """
    enabled = getattr(config, 'kge_benchmark', False)
    if not enabled:
        return None

    verbose = getattr(config, 'verbose', True)
    bench = KGEBenchmark(enabled=True)

    if verbose:
        print("[KGEBenchmark] Performance benchmarking enabled")

    # Set as global
    set_benchmark(bench)
    return bench


__all__ = [
    "KGEBenchmark",
    "get_benchmark",
    "set_benchmark",
    "create_benchmark",
]
