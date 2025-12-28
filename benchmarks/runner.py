"""Benchmark runner for framework comparisons."""

import asyncio
import json
import os
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.adapters import (
    ArtemisAdapter,
    AutoGenAdapter,
    CAMELAdapter,
    CrewAIAdapter,
    DebateAdapter,
)
from benchmarks.evaluator import DebateEvaluator, EvaluationScores


# Debate topics for benchmarking
BENCHMARK_TOPICS = [
    {
        "topic": "Should governments regulate AI development?",
        "pro": "supports government regulation of AI",
        "con": "opposes government regulation of AI",
    },
    {
        "topic": "Is it ethical to use AI for criminal sentencing decisions?",
        "pro": "supports AI use in criminal sentencing",
        "con": "opposes AI use in criminal sentencing",
    },
    {
        "topic": "Will AI create more jobs than it destroys in the next decade?",
        "pro": "believes AI will create net positive jobs",
        "con": "believes AI will cause net job losses",
    },
    {
        "topic": "Should social media platforms be legally liable for user-generated content?",
        "pro": "supports platform liability for content",
        "con": "opposes platform liability for content",
    },
    {
        "topic": "Is nuclear energy essential for addressing climate change?",
        "pro": "supports nuclear energy for climate goals",
        "con": "opposes reliance on nuclear energy",
    },
]


@dataclass
class BenchmarkRun:
    """Single benchmark run result."""

    framework: str
    topic: str
    trial: int
    argument_quality: float
    decision_accuracy: float
    reasoning_depth: float
    time_seconds: float
    error: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class AggregatedResults:
    """Aggregated results for a framework."""

    framework: str
    argument_quality_mean: float
    argument_quality_std: float
    decision_accuracy_mean: float
    decision_accuracy_std: float
    reasoning_depth_mean: float
    reasoning_depth_std: float
    total_runs: int
    successful_runs: int
    avg_time_seconds: float


class BenchmarkRunner:
    """Orchestrates benchmark runs across frameworks."""

    def __init__(
        self,
        model: str = "gpt-4o",
        rounds: int = 3,
        trials: int = 3,
        output_dir: str = "benchmarks/results",
    ):
        self.model = model
        self.rounds = rounds
        self.trials = trials
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = DebateEvaluator(model=model)

        # Initialize adapters
        self.adapters: dict[str, DebateAdapter] = {
            "artemis": ArtemisAdapter(model=model, rounds=rounds),
            "autogen": AutoGenAdapter(model=model, rounds=rounds),
            "crewai": CrewAIAdapter(model=model, rounds=rounds),
            "camel": CAMELAdapter(model=model, rounds=rounds),
        }

    def check_frameworks(self) -> dict[str, bool]:
        """Check which frameworks are available."""
        return {name: adapter.is_available() for name, adapter in self.adapters.items()}

    async def run_single(
        self,
        framework: str,
        topic_config: dict,
        trial: int,
    ) -> BenchmarkRun:
        """Run a single benchmark."""
        adapter = self.adapters.get(framework)
        if not adapter:
            return BenchmarkRun(
                framework=framework,
                topic=topic_config["topic"],
                trial=trial,
                argument_quality=0,
                decision_accuracy=0,
                reasoning_depth=0,
                time_seconds=0,
                error=f"Unknown framework: {framework}",
            )

        if not adapter.is_available():
            return BenchmarkRun(
                framework=framework,
                topic=topic_config["topic"],
                trial=trial,
                argument_quality=0,
                decision_accuracy=0,
                reasoning_depth=0,
                time_seconds=0,
                error=f"Framework not installed: {framework}",
            )

        print(f"  Running {framework} - Trial {trial + 1}...")

        # Run debate
        result = await adapter.run_debate(
            topic=topic_config["topic"],
            pro_position=topic_config["pro"],
            con_position=topic_config["con"],
        )

        if result.error:
            return BenchmarkRun(
                framework=framework,
                topic=topic_config["topic"],
                trial=trial,
                argument_quality=0,
                decision_accuracy=0,
                reasoning_depth=0,
                time_seconds=result.time_seconds,
                error=result.error,
            )

        # Evaluate
        print(f"    Evaluating {framework} debate...")
        scores = await self.evaluator.evaluate(result)

        return BenchmarkRun(
            framework=framework,
            topic=topic_config["topic"],
            trial=trial,
            argument_quality=scores.argument_quality,
            decision_accuracy=scores.decision_accuracy,
            reasoning_depth=scores.reasoning_depth,
            time_seconds=result.time_seconds,
            metadata=result.metadata,
        )

    async def run_framework(
        self,
        framework: str,
        topics: list[dict] | None = None,
    ) -> list[BenchmarkRun]:
        """Run all benchmarks for a single framework."""
        topics = topics or BENCHMARK_TOPICS
        results = []

        for topic_config in topics:
            print(f"\nTopic: {topic_config['topic'][:50]}...")

            for trial in range(self.trials):
                run = await self.run_single(framework, topic_config, trial)
                results.append(run)

                # Small delay between runs
                await asyncio.sleep(1)

        return results

    async def run_all(
        self,
        frameworks: list[str] | None = None,
        topics: list[dict] | None = None,
    ) -> dict[str, list[BenchmarkRun]]:
        """Run benchmarks for all frameworks."""
        frameworks = frameworks or list(self.adapters.keys())
        topics = topics or BENCHMARK_TOPICS

        all_results = {}

        print("=" * 60)
        print("ARTEMIS FRAMEWORK BENCHMARK")
        print("=" * 60)
        print(f"Model: {self.model}")
        print(f"Rounds per debate: {self.rounds}")
        print(f"Trials per topic: {self.trials}")
        print(f"Topics: {len(topics)}")
        print(f"Frameworks: {', '.join(frameworks)}")
        print("=" * 60)

        # Check availability
        availability = self.check_frameworks()
        print("\nFramework availability:")
        for fw, available in availability.items():
            status = "OK" if available else "NOT INSTALLED"
            print(f"  {fw}: {status}")

        for framework in frameworks:
            print(f"\n{'=' * 40}")
            print(f"FRAMEWORK: {framework.upper()}")
            print("=" * 40)

            results = await self.run_framework(framework, topics)
            all_results[framework] = results

        return all_results

    def aggregate_results(
        self,
        results: dict[str, list[BenchmarkRun]],
    ) -> dict[str, AggregatedResults]:
        """Aggregate results by framework."""
        aggregated = {}

        for framework, runs in results.items():
            successful = [r for r in runs if r.error is None]

            if not successful:
                aggregated[framework] = AggregatedResults(
                    framework=framework,
                    argument_quality_mean=0,
                    argument_quality_std=0,
                    decision_accuracy_mean=0,
                    decision_accuracy_std=0,
                    reasoning_depth_mean=0,
                    reasoning_depth_std=0,
                    total_runs=len(runs),
                    successful_runs=0,
                    avg_time_seconds=0,
                )
                continue

            arg_scores = [r.argument_quality for r in successful]
            dec_scores = [r.decision_accuracy for r in successful]
            rea_scores = [r.reasoning_depth for r in successful]
            times = [r.time_seconds for r in successful]

            aggregated[framework] = AggregatedResults(
                framework=framework,
                argument_quality_mean=statistics.mean(arg_scores),
                argument_quality_std=statistics.stdev(arg_scores) if len(arg_scores) > 1 else 0,
                decision_accuracy_mean=statistics.mean(dec_scores),
                decision_accuracy_std=statistics.stdev(dec_scores) if len(dec_scores) > 1 else 0,
                reasoning_depth_mean=statistics.mean(rea_scores),
                reasoning_depth_std=statistics.stdev(rea_scores) if len(rea_scores) > 1 else 0,
                total_runs=len(runs),
                successful_runs=len(successful),
                avg_time_seconds=statistics.mean(times),
            )

        return aggregated

    def save_results(
        self,
        results: dict[str, list[BenchmarkRun]],
        aggregated: dict[str, AggregatedResults],
    ) -> Path:
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_{timestamp}.json"

        output = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": self.model,
                "rounds": self.rounds,
                "trials": self.trials,
            },
            "raw_results": {
                fw: [asdict(r) for r in runs]
                for fw, runs in results.items()
            },
            "aggregated": {
                fw: asdict(agg)
                for fw, agg in aggregated.items()
            },
        }

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        return output_file

    def print_summary(self, aggregated: dict[str, AggregatedResults]) -> None:
        """Print summary table."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 80)
        print(
            f"{'Framework':<12} | {'Arg Quality':<15} | {'Decision Acc':<15} | {'Reasoning':<15} | {'Runs':<6}"
        )
        print("-" * 80)

        for fw, agg in sorted(aggregated.items()):
            arg = f"{agg.argument_quality_mean:.1f} +/- {agg.argument_quality_std:.1f}"
            dec = f"{agg.decision_accuracy_mean:.1f} +/- {agg.decision_accuracy_std:.1f}"
            rea = f"{agg.reasoning_depth_mean:.1f} +/- {agg.reasoning_depth_std:.1f}"
            runs = f"{agg.successful_runs}/{agg.total_runs}"
            print(f"{fw:<12} | {arg:<15} | {dec:<15} | {rea:<15} | {runs:<6}")

        print("=" * 80)

    def generate_readme_table(self, aggregated: dict[str, AggregatedResults]) -> str:
        """Generate markdown table for README."""
        lines = [
            "| Framework | Argument Quality | Decision Accuracy | Reasoning Depth |",
            "|-----------|-----------------|-------------------|-----------------|",
        ]

        # Sort by argument quality (descending)
        sorted_frameworks = sorted(
            aggregated.items(),
            key=lambda x: x[1].argument_quality_mean,
            reverse=True,
        )

        for fw, agg in sorted_frameworks:
            if agg.successful_runs == 0:
                continue

            name = f"**{fw.upper()}**" if fw == "artemis" else fw.title()
            arg = f"{agg.argument_quality_mean:.1f}%"
            dec = f"{agg.decision_accuracy_mean:.1f}%"
            rea = f"{agg.reasoning_depth_mean:.1f}%"

            lines.append(f"| {name} | {arg} | {dec} | {rea} |")

        return "\n".join(lines)


async def run_benchmark(
    frameworks: list[str] | None = None,
    topics: list[dict] | None = None,
    model: str = "gpt-4o",
    rounds: int = 3,
    trials: int = 3,
) -> dict[str, AggregatedResults]:
    """
    Run the full benchmark suite.

    Args:
        frameworks: List of frameworks to benchmark (default: all).
        topics: List of topic configs (default: BENCHMARK_TOPICS).
        model: LLM model to use.
        rounds: Debate rounds per benchmark.
        trials: Number of trials per topic/framework.

    Returns:
        Aggregated results by framework.
    """
    runner = BenchmarkRunner(
        model=model,
        rounds=rounds,
        trials=trials,
    )

    results = await runner.run_all(frameworks=frameworks, topics=topics)
    aggregated = runner.aggregate_results(results)

    runner.save_results(results, aggregated)
    runner.print_summary(aggregated)

    print("\nREADME Table:")
    print(runner.generate_readme_table(aggregated))

    return aggregated


if __name__ == "__main__":
    # Run with: python -m benchmarks.runner
    asyncio.run(run_benchmark())
