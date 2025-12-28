"""Verdict aggregation for hierarchical debates.

Provides strategies for combining verdicts from sub-debates
into a final compound verdict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter

from artemis.core.types import (
    AggregationMethod,
    CompoundVerdict,
    SubDebateSpec,
    Verdict,
)
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


class VerdictAggregator(ABC):
    """Abstract base class for verdict aggregation.

    Implementations determine how sub-debate verdicts are
    combined into a final verdict.
    """

    @property
    @abstractmethod
    def method(self) -> AggregationMethod:
        """The aggregation method used."""
        pass

    @abstractmethod
    def aggregate(
        self,
        verdicts: list[Verdict],
        specs: list[SubDebateSpec],
    ) -> CompoundVerdict:
        """Aggregate sub-debate verdicts into a compound verdict.

        Args:
            verdicts: Verdicts from each sub-debate.
            specs: Specifications for each sub-debate (for weights).

        Returns:
            CompoundVerdict combining all sub-verdicts.
        """
        pass


class WeightedAverageAggregator(VerdictAggregator):
    """Aggregates verdicts using weighted averaging.

    Uses the weights from SubDebateSpec to compute
    a weighted average of verdict confidences.

    Example:
        ```python
        aggregator = WeightedAverageAggregator()
        compound = aggregator.aggregate(verdicts, specs)
        print(f"Winner: {compound.final_decision}")
        ```
    """

    @property
    def method(self) -> AggregationMethod:
        return AggregationMethod.WEIGHTED_AVERAGE

    def aggregate(
        self,
        verdicts: list[Verdict],
        specs: list[SubDebateSpec],
    ) -> CompoundVerdict:
        """Aggregate using weighted averages."""
        if not verdicts:
            return CompoundVerdict(
                final_decision="no_winner",
                confidence=0.0,
                reasoning="No sub-debate verdicts to aggregate.",
                sub_verdicts=[],
            )

        # Collect scores per decision (usually agent names)
        decision_scores: dict[str, float] = {}
        total_weight = 0.0

        for verdict, spec in zip(verdicts, specs):
            weight = spec.weight if spec else 1.0
            total_weight += weight

            decision = verdict.decision.lower()
            if decision not in decision_scores:
                decision_scores[decision] = 0.0

            # Weight by both spec weight and verdict confidence
            decision_scores[decision] += weight * verdict.confidence

        # Normalize
        if total_weight > 0:
            for decision in decision_scores:
                decision_scores[decision] /= total_weight

        # Determine winner
        if decision_scores:
            winner = max(decision_scores, key=decision_scores.get)
            confidence = decision_scores[winner]
        else:
            winner = "draw"
            confidence = 0.0

        # Build aggregation weights dict
        weights_dict = {spec.aspect: spec.weight for spec in specs}

        # Build reasoning
        reasoning_parts = [
            f"Aggregated {len(verdicts)} sub-debate verdicts using weighted averaging."
        ]
        for verdict, spec in zip(verdicts, specs):
            reasoning_parts.append(
                f"- {spec.aspect}: {verdict.decision} "
                f"(confidence {verdict.confidence:.2f}, weight {spec.weight:.2f})"
            )

        return CompoundVerdict(
            final_decision=winner,
            confidence=confidence,
            reasoning="\n".join(reasoning_parts),
            sub_verdicts=verdicts,
            sub_topics=[s.aspect for s in specs],
            aggregation_method=self.method.value,
            aggregation_weights=weights_dict,
        )


class MajorityVoteAggregator(VerdictAggregator):
    """Aggregates verdicts using simple majority vote.

    Each sub-debate gets one vote for its verdict decision.

    Example:
        ```python
        aggregator = MajorityVoteAggregator()
        compound = aggregator.aggregate(verdicts, specs)
        ```
    """

    @property
    def method(self) -> AggregationMethod:
        return AggregationMethod.MAJORITY_VOTE

    def aggregate(
        self,
        verdicts: list[Verdict],
        specs: list[SubDebateSpec],
    ) -> CompoundVerdict:
        """Aggregate using majority vote."""
        if not verdicts:
            return CompoundVerdict(
                final_decision="no_winner",
                confidence=0.0,
                reasoning="No sub-debate verdicts to aggregate.",
                sub_verdicts=[],
            )

        # Count votes
        vote_counts = Counter(v.decision.lower() for v in verdicts)
        total_votes = len(verdicts)

        # Get winner
        winner, count = vote_counts.most_common(1)[0]
        confidence = count / total_votes

        # Check for tie
        top_votes = vote_counts.most_common(2)
        if len(top_votes) > 1 and top_votes[0][1] == top_votes[1][1]:
            # Tie - use confidence as tiebreaker
            tied_decisions = [d for d, c in top_votes if c == top_votes[0][1]]
            avg_confidences = {}
            for decision in tied_decisions:
                matching = [v for v in verdicts if v.decision.lower() == decision]
                avg_confidences[decision] = (
                    sum(v.confidence for v in matching) / len(matching)
                )
            winner = max(avg_confidences, key=avg_confidences.get)

        reasoning_parts = [
            f"Aggregated {len(verdicts)} sub-debate verdicts using majority vote."
        ]
        for decision, count in vote_counts.most_common():
            reasoning_parts.append(f"- {decision}: {count} votes")

        return CompoundVerdict(
            final_decision=winner,
            confidence=confidence,
            reasoning="\n".join(reasoning_parts),
            sub_verdicts=verdicts,
            sub_topics=[s.aspect for s in specs],
            aggregation_method=self.method.value,
            aggregation_weights={},
        )


class ConfidenceWeightedAggregator(VerdictAggregator):
    """Aggregates verdicts weighted by their confidence scores.

    Higher confidence verdicts have more influence on the final decision.

    Example:
        ```python
        aggregator = ConfidenceWeightedAggregator()
        compound = aggregator.aggregate(verdicts, specs)
        ```
    """

    @property
    def method(self) -> AggregationMethod:
        return AggregationMethod.CONFIDENCE_WEIGHTED

    def aggregate(
        self,
        verdicts: list[Verdict],
        specs: list[SubDebateSpec],
    ) -> CompoundVerdict:
        """Aggregate weighted by confidence."""
        if not verdicts:
            return CompoundVerdict(
                final_decision="no_winner",
                confidence=0.0,
                reasoning="No sub-debate verdicts to aggregate.",
                sub_verdicts=[],
            )

        # Sum confidences per decision
        decision_confidence: dict[str, float] = {}
        for verdict in verdicts:
            decision = verdict.decision.lower()
            if decision not in decision_confidence:
                decision_confidence[decision] = 0.0
            decision_confidence[decision] += verdict.confidence

        # Normalize
        total_confidence = sum(decision_confidence.values())
        if total_confidence > 0:
            for decision in decision_confidence:
                decision_confidence[decision] /= total_confidence

        # Determine winner
        winner = max(decision_confidence, key=decision_confidence.get)
        final_confidence = decision_confidence[winner]

        reasoning_parts = [
            f"Aggregated {len(verdicts)} sub-debate verdicts weighted by confidence."
        ]
        for decision, conf in sorted(
            decision_confidence.items(), key=lambda x: x[1], reverse=True
        ):
            reasoning_parts.append(f"- {decision}: {conf:.2%} weighted support")

        return CompoundVerdict(
            final_decision=winner,
            confidence=final_confidence,
            reasoning="\n".join(reasoning_parts),
            sub_verdicts=verdicts,
            sub_topics=[s.aspect for s in specs],
            aggregation_method=self.method.value,
            aggregation_weights={},
        )


class UnanimousAggregator(VerdictAggregator):
    """Requires unanimous agreement across all sub-debates.

    Returns the decision only if all sub-debates agree.

    Example:
        ```python
        aggregator = UnanimousAggregator()
        compound = aggregator.aggregate(verdicts, specs)
        if compound.final_decision == "no_consensus":
            print("Sub-debates did not reach unanimous agreement")
        ```
    """

    @property
    def method(self) -> AggregationMethod:
        return AggregationMethod.UNANIMOUS

    def aggregate(
        self,
        verdicts: list[Verdict],
        specs: list[SubDebateSpec],
    ) -> CompoundVerdict:
        """Aggregate requiring unanimity."""
        if not verdicts:
            return CompoundVerdict(
                final_decision="no_winner",
                confidence=0.0,
                reasoning="No sub-debate verdicts to aggregate.",
                sub_verdicts=[],
            )

        decisions = set(v.decision.lower() for v in verdicts)

        if len(decisions) == 1:
            # Unanimous
            winner = decisions.pop()
            avg_confidence = sum(v.confidence for v in verdicts) / len(verdicts)
            reasoning = f"All {len(verdicts)} sub-debates agreed on: {winner}"
        else:
            # No unanimity
            winner = "no_consensus"
            avg_confidence = 0.0
            reasoning = (
                f"Sub-debates did not reach unanimous agreement. "
                f"Decisions: {', '.join(decisions)}"
            )

        return CompoundVerdict(
            final_decision=winner,
            confidence=avg_confidence,
            reasoning=reasoning,
            sub_verdicts=verdicts,
            sub_topics=[s.aspect for s in specs],
            aggregation_method=self.method.value,
            aggregation_weights={},
        )


class WeightedMajorityAggregator(VerdictAggregator):
    """Majority vote with weights from sub-debate specs.

    Each sub-debate's vote is weighted by its spec weight.

    Example:
        ```python
        aggregator = WeightedMajorityAggregator()
        compound = aggregator.aggregate(verdicts, specs)
        ```
    """

    @property
    def method(self) -> AggregationMethod:
        return AggregationMethod.WEIGHTED_MAJORITY

    def aggregate(
        self,
        verdicts: list[Verdict],
        specs: list[SubDebateSpec],
    ) -> CompoundVerdict:
        """Aggregate using weighted majority vote."""
        if not verdicts:
            return CompoundVerdict(
                final_decision="no_winner",
                confidence=0.0,
                reasoning="No sub-debate verdicts to aggregate.",
                sub_verdicts=[],
            )

        # Sum weighted votes
        weighted_votes: dict[str, float] = {}
        total_weight = 0.0

        for verdict, spec in zip(verdicts, specs):
            decision = verdict.decision.lower()
            weight = spec.weight if spec else 1.0
            total_weight += weight

            if decision not in weighted_votes:
                weighted_votes[decision] = 0.0
            weighted_votes[decision] += weight

        # Normalize
        if total_weight > 0:
            for decision in weighted_votes:
                weighted_votes[decision] /= total_weight

        # Determine winner
        winner = max(weighted_votes, key=weighted_votes.get)
        confidence = weighted_votes[winner]

        weights_dict = {spec.aspect: spec.weight for spec in specs}

        reasoning_parts = [
            f"Aggregated {len(verdicts)} sub-debate verdicts using weighted majority."
        ]
        for decision, weight in sorted(
            weighted_votes.items(), key=lambda x: x[1], reverse=True
        ):
            reasoning_parts.append(f"- {decision}: {weight:.2%} weighted votes")

        return CompoundVerdict(
            final_decision=winner,
            confidence=confidence,
            reasoning="\n".join(reasoning_parts),
            sub_verdicts=verdicts,
            sub_topics=[s.aspect for s in specs],
            aggregation_method=self.method.value,
            aggregation_weights=weights_dict,
        )


def create_aggregator(method: AggregationMethod) -> VerdictAggregator:
    """Factory function to create an aggregator by method.

    Args:
        method: The aggregation method.

    Returns:
        Appropriate VerdictAggregator instance.
    """
    aggregators = {
        AggregationMethod.WEIGHTED_AVERAGE: WeightedAverageAggregator,
        AggregationMethod.MAJORITY_VOTE: MajorityVoteAggregator,
        AggregationMethod.CONFIDENCE_WEIGHTED: ConfidenceWeightedAggregator,
        AggregationMethod.UNANIMOUS: UnanimousAggregator,
        AggregationMethod.WEIGHTED_MAJORITY: WeightedMajorityAggregator,
    }

    aggregator_class = aggregators.get(method)
    if not aggregator_class:
        raise ValueError(f"Unknown aggregation method: {method}")

    return aggregator_class()
