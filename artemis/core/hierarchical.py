"""Hierarchical debate orchestration.

Provides the HierarchicalDebate class for managing debates with
sub-debates that decompose complex topics.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from artemis.core.aggregation import VerdictAggregator, WeightedAverageAggregator
from artemis.core.debate import Debate
from artemis.core.decomposition import (
    RuleBasedDecomposer,
    TopicDecomposer,
)
from artemis.core.types import (
    CompoundVerdict,
    DebateConfig,
    DebateResult,
    HierarchicalContext,
    Verdict,
)
from artemis.utils.logging import get_logger

if TYPE_CHECKING:
    from artemis.core.agent import Agent
    from artemis.core.evaluation import AdaptiveEvaluator
    from artemis.core.jury import JuryPanel

logger = get_logger(__name__)


class HierarchicalDebate:
    """Orchestrates hierarchical debates with sub-debates.

    Breaks complex topics into sub-topics, runs focused debates
    on each, and aggregates the results into a final verdict.

    Example:
        ```python
        from artemis.core import Agent, HierarchicalDebate
        from artemis.core.decomposition import LLMTopicDecomposer

        debate = HierarchicalDebate(
            topic="Should universal basic income be implemented?",
            agents=[agent1, agent2],
            decomposer=LLMTopicDecomposer(),
            max_depth=2,
        )

        result = await debate.run()
        print(f"Final decision: {result.final_decision}")
        for sub in result.sub_verdicts:
            print(f"  - {sub.decision}")
        ```
    """

    def __init__(
        self,
        topic: str,
        agents: list[Agent],
        decomposer: TopicDecomposer | None = None,
        aggregator: VerdictAggregator | None = None,
        rounds: int = 3,
        max_depth: int = 2,
        config: DebateConfig | None = None,
        jury: JuryPanel | None = None,
        evaluator: AdaptiveEvaluator | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a hierarchical debate.

        Args:
            topic: The main debate topic.
            agents: Agents participating in debates.
            decomposer: Topic decomposer for creating sub-debates.
            aggregator: Verdict aggregator for combining results.
            rounds: Number of rounds per sub-debate.
            max_depth: Maximum hierarchy depth.
            config: Debate configuration.
            jury: Jury panel for verdicts.
            evaluator: Argument evaluator.
            **kwargs: Additional arguments for sub-debates.
        """
        if len(agents) < 2:
            raise ValueError("Need at least 2 agents for a debate")

        self.debate_id = str(uuid4())
        self.topic = topic
        self.agents = agents
        self.rounds = rounds
        self.max_depth = max_depth
        self.config = config or DebateConfig()

        self.decomposer = decomposer or RuleBasedDecomposer()
        self.aggregator = aggregator or WeightedAverageAggregator()

        self._jury = jury
        self._evaluator = evaluator
        self._kwargs = kwargs

        # State tracking
        self._sub_debates: list[Debate] = []
        self._sub_results: list[DebateResult] = []
        self._started_at: datetime | None = None
        self._ended_at: datetime | None = None

        logger.info(
            "HierarchicalDebate initialized",
            debate_id=self.debate_id,
            topic=topic[:50],
            agents=[a.name for a in agents],
            max_depth=max_depth,
        )

    async def run(self) -> CompoundVerdict:
        """Run the hierarchical debate.

        Returns:
            CompoundVerdict with aggregated results.
        """
        self._started_at = datetime.utcnow()

        logger.info(
            "Starting hierarchical debate",
            debate_id=self.debate_id,
            topic=self.topic[:50],
        )

        # Create root context
        context = HierarchicalContext(
            parent_topic=self.topic,
            depth=0,
            max_depth=self.max_depth,
            path=[self.topic],
        )

        # Run recursive debate
        result = await self._run_at_depth(self.topic, context)

        self._ended_at = datetime.utcnow()

        logger.info(
            "Hierarchical debate complete",
            debate_id=self.debate_id,
            final_decision=result.final_decision,
            sub_debates=len(result.sub_verdicts),
        )

        return result

    async def _run_at_depth(
        self,
        topic: str,
        context: HierarchicalContext,
    ) -> CompoundVerdict:
        """Run debate at a specific depth in the hierarchy.

        Args:
            topic: Topic for this level.
            context: Hierarchical context.

        Returns:
            CompoundVerdict from this level.
        """
        # Determine if we should decompose further
        if context.depth >= context.max_depth:
            # Run a leaf debate
            result = await self._run_leaf_debate(topic, context)
            return CompoundVerdict(
                final_decision=result.verdict.decision,
                confidence=result.verdict.confidence,
                reasoning=result.verdict.reasoning,
                sub_verdicts=[result.verdict],
                sub_topics=[topic],
                aggregation_method="leaf",
                depth=context.depth,
            )

        # Decompose topic
        specs = await self.decomposer.decompose(topic, context, max_subtopics=5)

        if not specs or len(specs) < 2:
            # Not enough sub-topics - run as leaf
            logger.debug(
                "Insufficient sub-topics for decomposition, running as leaf",
                topic=topic[:30],
            )
            result = await self._run_leaf_debate(topic, context)
            return CompoundVerdict(
                final_decision=result.verdict.decision,
                confidence=result.verdict.confidence,
                reasoning=result.verdict.reasoning,
                sub_verdicts=[result.verdict],
                sub_topics=[topic],
                aggregation_method="leaf",
                depth=context.depth,
            )

        # Run sub-debates
        sub_verdicts: list[Verdict] = []
        sub_topics: list[str] = []

        for spec in specs:
            # Create child context
            child_context = HierarchicalContext(
                parent_topic=topic,
                sibling_verdicts=sub_verdicts.copy(),
                sibling_topics=sub_topics.copy(),
                depth=context.depth + 1,
                max_depth=context.max_depth,
                path=context.path + [spec.aspect],
            )

            logger.debug(
                "Running sub-debate",
                aspect=spec.aspect[:30],
                depth=child_context.depth,
            )

            # Recursively run sub-debate
            sub_result = await self._run_at_depth(spec.aspect, child_context)

            # Collect verdict
            verdict = Verdict(
                decision=sub_result.final_decision,
                confidence=sub_result.confidence,
                reasoning=sub_result.reasoning,
                unanimous=len(sub_result.sub_verdicts) == 1,
            )
            sub_verdicts.append(verdict)
            sub_topics.append(spec.aspect)

        # Aggregate sub-verdicts
        compound = self.aggregator.aggregate(sub_verdicts, specs)
        compound.depth = context.depth

        return compound

    async def _run_leaf_debate(
        self,
        topic: str,
        context: HierarchicalContext,
    ) -> DebateResult:
        """Run a leaf debate (no further decomposition).

        Args:
            topic: Topic for the debate.
            context: Hierarchical context.

        Returns:
            DebateResult from the leaf debate.
        """
        # Adjust rounds for depth
        rounds = max(1, self.rounds - context.depth)

        # Create and run debate
        debate = Debate(
            topic=topic,
            agents=self.agents,
            rounds=rounds,
            config=self.config,
            jury=self._jury,
            evaluator=self._evaluator,
            **self._kwargs,
        )

        self._sub_debates.append(debate)

        result = await debate.run()
        self._sub_results.append(result)

        return result

    def get_debate_tree(self) -> dict:
        """Get the tree structure of debates.

        Returns:
            Dictionary representing the debate hierarchy.
        """
        return {
            "topic": self.topic,
            "debate_id": self.debate_id,
            "max_depth": self.max_depth,
            "sub_debates": len(self._sub_debates),
            "sub_results": [
                {
                    "topic": r.topic,
                    "verdict": r.verdict.decision,
                    "confidence": r.verdict.confidence,
                }
                for r in self._sub_results
            ],
        }

    @property
    def sub_debates(self) -> list[Debate]:
        """All sub-debates that were run."""
        return self._sub_debates.copy()

    @property
    def sub_results(self) -> list[DebateResult]:
        """Results from all sub-debates."""
        return self._sub_results.copy()

    def __repr__(self) -> str:
        return (
            f"HierarchicalDebate(topic={self.topic[:30]!r}..., "
            f"max_depth={self.max_depth}, "
            f"agents={[a.name for a in self.agents]})"
        )
