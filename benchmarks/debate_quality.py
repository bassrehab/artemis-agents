"""
ARTEMIS Debate Quality Benchmark

Measures the quality of debates produced by ARTEMIS.
"""

import time
from dataclasses import dataclass
from typing import Any

from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.types import DebateConfig, DebateResult
from artemis.utils.logging import get_logger
from benchmarks.base import Benchmark, BenchmarkMetrics, BenchmarkResult

logger = get_logger(__name__)


@dataclass
class QualityScores:
    """Quality scores for a debate."""

    argument_depth: float = 0.0
    evidence_quality: float = 0.0
    logical_structure: float = 0.0
    counterargument_strength: float = 0.0
    verdict_justification: float = 0.0

    @property
    def overall(self) -> float:
        """Calculate overall quality score."""
        weights = {
            "argument_depth": 0.25,
            "evidence_quality": 0.25,
            "logical_structure": 0.20,
            "counterargument_strength": 0.15,
            "verdict_justification": 0.15,
        }
        return (
            self.argument_depth * weights["argument_depth"]
            + self.evidence_quality * weights["evidence_quality"]
            + self.logical_structure * weights["logical_structure"]
            + self.counterargument_strength * weights["counterargument_strength"]
            + self.verdict_justification * weights["verdict_justification"]
        )


class DebateQualityBenchmark(Benchmark):
    """
    Benchmark for measuring debate quality.

    Evaluates:
    - Argument depth and sophistication
    - Evidence usage and quality
    - Logical structure and coherence
    - Counterargument engagement
    - Verdict justification quality

    Example:
        >>> benchmark = DebateQualityBenchmark(
        ...     topic="Should AI be regulated?",
        ...     model="gpt-4o",
        ...     rounds=3,
        ... )
        >>> result = await benchmark.run()
    """

    name = "debate_quality"
    description = "Measures debate argument quality and coherence"

    def __init__(
        self,
        topic: str = "Should AI be regulated by government?",
        model: str = "gpt-4o",
        rounds: int = 3,
        pro_position: str = "supports regulation",
        con_position: str = "opposes regulation",
        **config: Any,
    ) -> None:
        """
        Initialize the benchmark.

        Args:
            topic: Debate topic.
            model: LLM model to use.
            rounds: Number of debate rounds.
            pro_position: Pro-side position.
            con_position: Con-side position.
            **config: Additional configuration.
        """
        super().__init__(**config)
        self.topic = topic
        self.model = model
        self.rounds = rounds
        self.pro_position = pro_position
        self.con_position = con_position

    async def run(self) -> BenchmarkResult:
        """Run the debate quality benchmark."""
        logger.info(
            "Starting debate quality benchmark",
            topic=self.topic[:50],
            model=self.model,
            rounds=self.rounds,
        )

        start_time = time.time()
        round_times: list[float] = []

        try:
            # Create agents
            pro_agent = Agent(
                name="pro_agent",
                model=self.model,
                position=self.pro_position,
            )
            con_agent = Agent(
                name="con_agent",
                model=self.model,
                position=self.con_position,
            )

            # Create debate
            debate = Debate(
                topic=self.topic,
                agents=[pro_agent, con_agent],
                rounds=self.rounds,
                config=DebateConfig(),
            )

            debate.assign_positions({
                "pro_agent": self.pro_position,
                "con_agent": self.con_position,
            })

            # Run debate with timing
            result = await debate.run()
            total_time = time.time() - start_time

            # Analyze quality
            quality = self._analyze_quality(result)

            # Build metrics
            metrics = self._build_metrics(result, quality, total_time, round_times)

            return BenchmarkResult(
                name=self.name,
                success=True,
                metrics=metrics,
                metadata={
                    "topic": self.topic,
                    "model": self.model,
                    "rounds": self.rounds,
                    "verdict": result.verdict.decision,
                },
            )

        except Exception as e:
            logger.error("Benchmark failed", error=str(e))
            return BenchmarkResult(
                name=self.name,
                success=False,
                metrics=BenchmarkMetrics(),
                error=str(e),
            )

    def _analyze_quality(self, result: DebateResult) -> QualityScores:
        """Analyze the quality of the debate."""
        scores = QualityScores()

        # Analyze argument depth
        scores.argument_depth = self._score_argument_depth(result)

        # Analyze evidence quality
        scores.evidence_quality = self._score_evidence_quality(result)

        # Analyze logical structure
        scores.logical_structure = self._score_logical_structure(result)

        # Analyze counterargument engagement
        scores.counterargument_strength = self._score_counterarguments(result)

        # Analyze verdict justification
        scores.verdict_justification = self._score_verdict(result)

        return scores

    def _score_argument_depth(self, result: DebateResult) -> float:
        """Score the depth of arguments."""
        if not result.transcript:
            return 0.0

        total_score = 0.0
        for turn in result.transcript:
            content = turn.argument.content
            # Score based on:
            # - Length (more detailed arguments)
            # - Paragraph structure
            # - Use of qualifiers and nuance

            length_score = min(len(content) / 1000, 1.0)
            paragraph_score = min(content.count("\n\n") / 3, 1.0)
            nuance_words = ["however", "although", "while", "despite", "nevertheless"]
            nuance_score = sum(1 for w in nuance_words if w in content.lower()) / 3

            total_score += (length_score + paragraph_score + nuance_score) / 3

        return total_score / len(result.transcript)

    def _score_evidence_quality(self, result: DebateResult) -> float:
        """Score the quality of evidence usage."""
        if not result.transcript:
            return 0.0

        total_score = 0.0
        for turn in result.transcript:
            content = turn.argument.content.lower()

            # Look for evidence indicators
            evidence_phrases = [
                "research shows",
                "studies indicate",
                "according to",
                "data suggests",
                "evidence demonstrates",
                "statistics show",
                "experts agree",
                "analysis reveals",
            ]
            evidence_count = sum(1 for p in evidence_phrases if p in content)

            # Look for specific citations
            has_numbers = any(c.isdigit() for c in content)
            has_percentages = "%" in content

            evidence_score = min(evidence_count / 3, 1.0)
            specificity_score = 0.5 if has_numbers else 0.0
            specificity_score += 0.5 if has_percentages else 0.0

            total_score += (evidence_score + specificity_score) / 2

        return total_score / len(result.transcript)

    def _score_logical_structure(self, result: DebateResult) -> float:
        """Score the logical structure of arguments."""
        if not result.transcript:
            return 0.0

        total_score = 0.0
        for turn in result.transcript:
            content = turn.argument.content.lower()

            # Look for logical connectors
            connectors = [
                "therefore",
                "thus",
                "consequently",
                "because",
                "since",
                "as a result",
                "this means",
                "first",
                "second",
                "finally",
            ]
            connector_count = sum(1 for c in connectors if c in content)

            # Look for structured reasoning
            structure_indicators = [
                "on one hand",
                "on the other",
                "in conclusion",
                "to summarize",
                "the key point",
            ]
            structure_count = sum(1 for s in structure_indicators if s in content)

            connector_score = min(connector_count / 4, 1.0)
            structure_score = min(structure_count / 2, 1.0)

            total_score += (connector_score + structure_score) / 2

        return total_score / len(result.transcript)

    def _score_counterarguments(self, result: DebateResult) -> float:
        """Score engagement with counterarguments."""
        if not result.transcript:
            return 0.0

        total_score = 0.0
        for turn in result.transcript:
            content = turn.argument.content.lower()

            # Look for counterargument engagement
            counter_phrases = [
                "opponents argue",
                "critics claim",
                "some say",
                "it is argued that",
                "the opposing view",
                "while it may seem",
                "addressing the concern",
                "this objection",
            ]
            counter_count = sum(1 for p in counter_phrases if p in content)

            # Look for rebuttals
            rebuttal_phrases = [
                "however, this ignores",
                "this fails to account",
                "but this overlooks",
                "the flaw in this",
                "this argument misses",
            ]
            rebuttal_count = sum(1 for r in rebuttal_phrases if r in content)

            counter_score = min(counter_count / 2, 1.0)
            rebuttal_score = min(rebuttal_count, 1.0)

            total_score += (counter_score + rebuttal_score) / 2

        return total_score / len(result.transcript)

    def _score_verdict(self, result: DebateResult) -> float:
        """Score the quality of the verdict."""
        verdict = result.verdict

        # Base score on confidence
        confidence_score = verdict.confidence

        # Score reasoning quality
        reasoning = verdict.reasoning.lower() if verdict.reasoning else ""
        reasoning_length_score = min(len(reasoning) / 500, 1.0)

        # Check for balanced consideration
        balance_words = ["both", "while", "although", "however"]
        balance_score = min(
            sum(1 for w in balance_words if w in reasoning) / 2, 1.0
        )

        return (confidence_score + reasoning_length_score + balance_score) / 3

    def _build_metrics(
        self,
        result: DebateResult,
        quality: QualityScores,
        total_time: float,
        round_times: list[float],
    ) -> BenchmarkMetrics:
        """Build metrics from results."""
        # Calculate token usage
        total_tokens = 0
        for turn in result.transcript:
            if turn.evaluation and turn.evaluation.total_score:
                # Estimate tokens from content length
                total_tokens += len(turn.argument.content) // 4

        return BenchmarkMetrics(
            total_time_seconds=total_time,
            avg_round_time_seconds=(
                sum(round_times) / len(round_times) if round_times else 0.0
            ),
            min_round_time_seconds=min(round_times) if round_times else 0.0,
            max_round_time_seconds=max(round_times) if round_times else 0.0,
            argument_quality_score=quality.overall,
            evidence_usage_score=quality.evidence_quality,
            logical_coherence_score=quality.logical_structure,
            verdict_confidence=result.verdict.confidence,
            total_tokens_used=total_tokens,
            avg_tokens_per_turn=(
                total_tokens / len(result.transcript)
                if result.transcript
                else 0.0
            ),
            custom={
                "argument_depth": quality.argument_depth,
                "counterargument_strength": quality.counterargument_strength,
                "verdict_justification": quality.verdict_justification,
            },
        )


class DebateSpeedBenchmark(Benchmark):
    """
    Benchmark for measuring debate execution speed.

    Focuses on timing metrics across different configurations.
    """

    name = "debate_speed"
    description = "Measures debate execution performance"

    def __init__(
        self,
        topic: str = "Test topic",
        model: str = "gpt-4o",
        rounds_range: tuple[int, int] = (1, 5),
        iterations: int = 3,
        **config: Any,
    ) -> None:
        """Initialize the benchmark."""
        super().__init__(**config)
        self.topic = topic
        self.model = model
        self.rounds_range = rounds_range
        self.iterations = iterations

    async def run(self) -> BenchmarkResult:
        """Run speed benchmarks."""
        logger.info(
            "Starting speed benchmark",
            model=self.model,
            rounds_range=self.rounds_range,
        )

        timings: dict[int, list[float]] = {}
        start_time = time.time()

        try:
            for rounds in range(self.rounds_range[0], self.rounds_range[1] + 1):
                timings[rounds] = []

                for _ in range(self.iterations):
                    # Create and run debate
                    pro_agent = Agent(
                        name="pro_agent",
                        model=self.model,
                        position="supports",
                    )
                    con_agent = Agent(
                        name="con_agent",
                        model=self.model,
                        position="opposes",
                    )

                    debate = Debate(
                        topic=self.topic,
                        agents=[pro_agent, con_agent],
                        rounds=rounds,
                    )

                    debate.assign_positions({
                        "pro_agent": "supports",
                        "con_agent": "opposes",
                    })

                    round_start = time.time()
                    await debate.run()
                    round_time = time.time() - round_start

                    timings[rounds].append(round_time)

            total_time = time.time() - start_time

            # Calculate metrics
            all_times = [t for times in timings.values() for t in times]
            avg_time = sum(all_times) / len(all_times)

            metrics = BenchmarkMetrics(
                total_time_seconds=total_time,
                avg_round_time_seconds=avg_time,
                min_round_time_seconds=min(all_times),
                max_round_time_seconds=max(all_times),
                custom={
                    "timings_by_rounds": {
                        str(r): {
                            "avg": sum(t) / len(t),
                            "min": min(t),
                            "max": max(t),
                        }
                        for r, t in timings.items()
                    },
                },
            )

            return BenchmarkResult(
                name=self.name,
                success=True,
                metrics=metrics,
                metadata={
                    "model": self.model,
                    "rounds_range": self.rounds_range,
                    "iterations": self.iterations,
                },
            )

        except Exception as e:
            logger.error("Speed benchmark failed", error=str(e))
            return BenchmarkResult(
                name=self.name,
                success=False,
                metrics=BenchmarkMetrics(),
                error=str(e),
            )
