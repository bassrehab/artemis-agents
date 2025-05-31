"""
ARTEMIS Benchmark Base Classes

Provides the foundation for running benchmarks and collecting metrics.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from artemis.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkMetrics:
    """Metrics collected during a benchmark run."""

    # Timing metrics
    total_time_seconds: float = 0.0
    avg_round_time_seconds: float = 0.0
    min_round_time_seconds: float = 0.0
    max_round_time_seconds: float = 0.0

    # Quality metrics
    argument_quality_score: float = 0.0
    evidence_usage_score: float = 0.0
    logical_coherence_score: float = 0.0
    verdict_confidence: float = 0.0

    # Token metrics
    total_tokens_used: int = 0
    avg_tokens_per_turn: float = 0.0
    reasoning_tokens_used: int = 0

    # Safety metrics
    safety_violations: int = 0
    deception_detections: int = 0
    sandbagging_detections: int = 0

    # Custom metrics
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timing": {
                "total_seconds": self.total_time_seconds,
                "avg_round_seconds": self.avg_round_time_seconds,
                "min_round_seconds": self.min_round_time_seconds,
                "max_round_seconds": self.max_round_time_seconds,
            },
            "quality": {
                "argument_score": self.argument_quality_score,
                "evidence_score": self.evidence_usage_score,
                "coherence_score": self.logical_coherence_score,
                "verdict_confidence": self.verdict_confidence,
            },
            "tokens": {
                "total": self.total_tokens_used,
                "avg_per_turn": self.avg_tokens_per_turn,
                "reasoning": self.reasoning_tokens_used,
            },
            "safety": {
                "violations": self.safety_violations,
                "deception": self.deception_detections,
                "sandbagging": self.sandbagging_detections,
            },
            "custom": self.custom,
        }


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    success: bool
    metrics: BenchmarkMetrics
    timestamp: datetime = field(default_factory=datetime.now)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
            "metrics": self.metrics.to_dict(),
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Get a summary string."""
        status = "PASS" if self.success else "FAIL"
        return (
            f"{self.name}: {status} "
            f"({self.metrics.total_time_seconds:.2f}s, "
            f"quality={self.metrics.argument_quality_score:.2f})"
        )


class Benchmark(ABC):
    """
    Abstract base class for benchmarks.

    Subclass this to create specific benchmarks.

    Example:
        >>> class MyBenchmark(Benchmark):
        ...     name = "my_benchmark"
        ...
        ...     async def run(self) -> BenchmarkResult:
        ...         # Run benchmark
        ...         metrics = BenchmarkMetrics(...)
        ...         return BenchmarkResult(name=self.name, success=True, metrics=metrics)
    """

    name: str = "base_benchmark"
    description: str = "Base benchmark"

    def __init__(self, **config: Any) -> None:
        """Initialize with configuration."""
        self.config = config

    @abstractmethod
    async def run(self) -> BenchmarkResult:
        """
        Run the benchmark.

        Returns:
            BenchmarkResult with metrics.
        """
        ...

    async def setup(self) -> None:  # noqa: B027
        """Setup before running. Override if needed."""

    async def teardown(self) -> None:  # noqa: B027
        """Teardown after running. Override if needed."""

    async def execute(self) -> BenchmarkResult:
        """Execute the full benchmark lifecycle."""
        try:
            await self.setup()
            result = await self.run()
            await self.teardown()
            return result
        except Exception as e:
            logger.error("Benchmark failed", name=self.name, error=str(e))
            return BenchmarkResult(
                name=self.name,
                success=False,
                metrics=BenchmarkMetrics(),
                error=str(e),
            )


class BenchmarkSuite:
    """
    A collection of benchmarks to run together.

    Example:
        >>> suite = BenchmarkSuite("my_suite")
        >>> suite.add(MyBenchmark())
        >>> suite.add(AnotherBenchmark())
        >>> results = await suite.run_all()
    """

    def __init__(self, name: str, description: str = "") -> None:
        """Initialize the suite."""
        self.name = name
        self.description = description
        self.benchmarks: list[Benchmark] = []

    def add(self, benchmark: Benchmark) -> "BenchmarkSuite":
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)
        return self

    async def run_all(
        self,
        parallel: bool = False,
    ) -> list[BenchmarkResult]:
        """
        Run all benchmarks.

        Args:
            parallel: Whether to run benchmarks in parallel.

        Returns:
            List of results.
        """
        logger.info(
            "Running benchmark suite",
            name=self.name,
            count=len(self.benchmarks),
        )

        if parallel:
            tasks = [b.execute() for b in self.benchmarks]
            results = await asyncio.gather(*tasks)
        else:
            results = []
            for benchmark in self.benchmarks:
                result = await benchmark.execute()
                results.append(result)

        return list(results)

    def summary(self, results: list[BenchmarkResult]) -> str:
        """Generate a summary of results."""
        lines = [
            f"Suite: {self.name}",
            f"Description: {self.description}",
            f"Benchmarks: {len(results)}",
            "-" * 40,
        ]

        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed

        for result in results:
            lines.append(result.summary())

        lines.extend([
            "-" * 40,
            f"Passed: {passed}/{len(results)}",
            f"Failed: {failed}/{len(results)}",
        ])

        return "\n".join(lines)


class BenchmarkRunner:
    """
    Runs benchmarks and collects results.

    Example:
        >>> runner = BenchmarkRunner()
        >>> runner.add_suite(suite)
        >>> all_results = await runner.run()
        >>> runner.print_report(all_results)
    """

    def __init__(self) -> None:
        """Initialize the runner."""
        self.suites: list[BenchmarkSuite] = []

    def add_suite(self, suite: BenchmarkSuite) -> "BenchmarkRunner":
        """Add a suite to run."""
        self.suites.append(suite)
        return self

    def add_benchmark(
        self,
        benchmark: Benchmark,
        suite_name: str = "default",
    ) -> "BenchmarkRunner":
        """Add a single benchmark to a suite."""
        # Find or create suite
        suite = next(
            (s for s in self.suites if s.name == suite_name),
            None,
        )
        if not suite:
            suite = BenchmarkSuite(suite_name)
            self.suites.append(suite)

        suite.add(benchmark)
        return self

    async def run(
        self,
        parallel_suites: bool = False,
    ) -> dict[str, list[BenchmarkResult]]:
        """
        Run all suites.

        Returns:
            Dictionary mapping suite names to results.
        """
        logger.info("Starting benchmark run", suites=len(self.suites))
        start_time = time.time()

        all_results: dict[str, list[BenchmarkResult]] = {}

        if parallel_suites:
            tasks = [(s.name, s.run_all()) for s in self.suites]
            suite_results = await asyncio.gather(*[t[1] for t in tasks])
            for i, suite in enumerate(self.suites):
                all_results[suite.name] = suite_results[i]
        else:
            for suite in self.suites:
                results = await suite.run_all()
                all_results[suite.name] = results

        elapsed = time.time() - start_time
        logger.info("Benchmark run complete", elapsed=f"{elapsed:.2f}s")

        return all_results

    def print_report(
        self,
        results: dict[str, list[BenchmarkResult]],
    ) -> None:
        """Print a formatted report."""
        print("\n" + "=" * 60)
        print("BENCHMARK REPORT")
        print("=" * 60)

        total_passed = 0
        total_failed = 0

        for suite_name, suite_results in results.items():
            suite = next(
                (s for s in self.suites if s.name == suite_name),
                None,
            )
            if suite:
                print("\n" + suite.summary(suite_results))
                total_passed += sum(1 for r in suite_results if r.success)
                total_failed += sum(1 for r in suite_results if not r.success)

        print("\n" + "=" * 60)
        print(f"TOTAL: {total_passed} passed, {total_failed} failed")
        print("=" * 60 + "\n")

    def export_json(
        self,
        results: dict[str, list[BenchmarkResult]],
        filepath: str,
    ) -> None:
        """Export results to JSON file."""
        import json

        data = {
            "timestamp": datetime.now().isoformat(),
            "suites": {
                name: [r.to_dict() for r in suite_results]
                for name, suite_results in results.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Results exported", filepath=filepath)
