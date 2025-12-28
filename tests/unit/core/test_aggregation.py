"""Tests for verdict aggregation strategies."""

import pytest

from artemis.core.aggregation import (
    ConfidenceWeightedAggregator,
    MajorityVoteAggregator,
    UnanimousAggregator,
    VerdictAggregator,
    WeightedAverageAggregator,
    WeightedMajorityAggregator,
    create_aggregator,
)
from artemis.core.types import (
    AggregationMethod,
    SubDebateSpec,
    Verdict,
)


class TestVerdictAggregatorInterface:
    """Tests for VerdictAggregator ABC."""

    def test_cannot_instantiate_abc(self):
        """Should not be able to instantiate abstract class."""
        with pytest.raises(TypeError):
            VerdictAggregator()

    def test_subclass_requires_methods(self):
        """Subclass must implement required methods."""
        class IncompleteAggregator(VerdictAggregator):
            pass

        with pytest.raises(TypeError):
            IncompleteAggregator()


class TestWeightedAverageAggregator:
    """Tests for WeightedAverageAggregator."""

    @pytest.fixture
    def aggregator(self) -> WeightedAverageAggregator:
        return WeightedAverageAggregator()

    @pytest.fixture
    def verdicts(self) -> list[Verdict]:
        """Create test verdicts."""
        return [
            Verdict(decision="agent_a", confidence=0.8, reasoning="Strong argument"),
            Verdict(decision="agent_b", confidence=0.6, reasoning="Decent argument"),
            Verdict(decision="agent_a", confidence=0.7, reasoning="Good point"),
        ]

    @pytest.fixture
    def specs(self) -> list[SubDebateSpec]:
        """Create test specs."""
        return [
            SubDebateSpec(aspect="Aspect 1", weight=0.4),
            SubDebateSpec(aspect="Aspect 2", weight=0.3),
            SubDebateSpec(aspect="Aspect 3", weight=0.3),
        ]

    def test_method_property(self, aggregator: WeightedAverageAggregator):
        """Should return WEIGHTED_AVERAGE method."""
        assert aggregator.method == AggregationMethod.WEIGHTED_AVERAGE

    def test_aggregate_empty_verdicts(self, aggregator: WeightedAverageAggregator):
        """Should handle empty verdict list."""
        result = aggregator.aggregate([], [])
        assert result.final_decision == "no_winner"
        assert result.confidence == 0.0

    def test_aggregate_single_verdict(self, aggregator: WeightedAverageAggregator):
        """Should handle single verdict."""
        verdict = Verdict(decision="winner", confidence=0.9, reasoning="Only one")
        spec = SubDebateSpec(aspect="Only aspect", weight=1.0)

        result = aggregator.aggregate([verdict], [spec])
        assert result.final_decision == "winner"
        assert result.confidence == pytest.approx(0.9, rel=0.01)

    def test_aggregate_weighted_winner(
        self,
        aggregator: WeightedAverageAggregator,
        verdicts: list[Verdict],
        specs: list[SubDebateSpec],
    ):
        """Should determine winner based on weighted scores."""
        result = aggregator.aggregate(verdicts, specs)

        # agent_a has 2 verdicts, should win
        assert result.final_decision == "agent_a"
        assert result.confidence > 0
        assert len(result.sub_verdicts) == 3
        assert result.aggregation_method == "weighted_average"

    def test_aggregate_includes_weights(
        self,
        aggregator: WeightedAverageAggregator,
        verdicts: list[Verdict],
        specs: list[SubDebateSpec],
    ):
        """Should include aggregation weights in result."""
        result = aggregator.aggregate(verdicts, specs)

        assert result.aggregation_weights is not None
        assert "Aspect 1" in result.aggregation_weights
        assert result.aggregation_weights["Aspect 1"] == 0.4


class TestMajorityVoteAggregator:
    """Tests for MajorityVoteAggregator."""

    @pytest.fixture
    def aggregator(self) -> MajorityVoteAggregator:
        return MajorityVoteAggregator()

    def test_method_property(self, aggregator: MajorityVoteAggregator):
        """Should return MAJORITY_VOTE method."""
        assert aggregator.method == AggregationMethod.MAJORITY_VOTE

    def test_aggregate_empty_verdicts(self, aggregator: MajorityVoteAggregator):
        """Should handle empty verdict list."""
        result = aggregator.aggregate([], [])
        assert result.final_decision == "no_winner"

    def test_aggregate_clear_majority(self, aggregator: MajorityVoteAggregator):
        """Should determine winner by vote count."""
        verdicts = [
            Verdict(decision="A", confidence=0.7, reasoning=""),
            Verdict(decision="A", confidence=0.8, reasoning=""),
            Verdict(decision="B", confidence=0.9, reasoning=""),
        ]
        specs = [
            SubDebateSpec(aspect=f"Aspect {i}", weight=1.0)
            for i in range(3)
        ]

        result = aggregator.aggregate(verdicts, specs)
        assert result.final_decision == "a"  # lowercase
        assert result.confidence == pytest.approx(2/3, rel=0.01)

    def test_aggregate_tie_uses_confidence(self, aggregator: MajorityVoteAggregator):
        """Should break ties using confidence."""
        verdicts = [
            Verdict(decision="A", confidence=0.9, reasoning=""),  # High confidence
            Verdict(decision="B", confidence=0.5, reasoning=""),  # Low confidence
        ]
        specs = [
            SubDebateSpec(aspect=f"Aspect {i}", weight=1.0)
            for i in range(2)
        ]

        result = aggregator.aggregate(verdicts, specs)
        # A should win due to higher confidence
        assert result.final_decision == "a"


class TestConfidenceWeightedAggregator:
    """Tests for ConfidenceWeightedAggregator."""

    @pytest.fixture
    def aggregator(self) -> ConfidenceWeightedAggregator:
        return ConfidenceWeightedAggregator()

    def test_method_property(self, aggregator: ConfidenceWeightedAggregator):
        """Should return CONFIDENCE_WEIGHTED method."""
        assert aggregator.method == AggregationMethod.CONFIDENCE_WEIGHTED

    def test_aggregate_empty_verdicts(self, aggregator: ConfidenceWeightedAggregator):
        """Should handle empty verdict list."""
        result = aggregator.aggregate([], [])
        assert result.final_decision == "no_winner"

    def test_aggregate_high_confidence_wins(self, aggregator: ConfidenceWeightedAggregator):
        """Higher confidence verdicts should have more weight."""
        verdicts = [
            Verdict(decision="A", confidence=0.9, reasoning=""),  # High
            Verdict(decision="B", confidence=0.3, reasoning=""),  # Low
            Verdict(decision="B", confidence=0.3, reasoning=""),  # Low
        ]
        specs = [
            SubDebateSpec(aspect=f"Aspect {i}", weight=1.0)
            for i in range(3)
        ]

        result = aggregator.aggregate(verdicts, specs)
        # A should win due to higher confidence weight (0.9 vs 0.3+0.3)
        assert result.final_decision == "a"


class TestUnanimousAggregator:
    """Tests for UnanimousAggregator."""

    @pytest.fixture
    def aggregator(self) -> UnanimousAggregator:
        return UnanimousAggregator()

    def test_method_property(self, aggregator: UnanimousAggregator):
        """Should return UNANIMOUS method."""
        assert aggregator.method == AggregationMethod.UNANIMOUS

    def test_aggregate_empty_verdicts(self, aggregator: UnanimousAggregator):
        """Should handle empty verdict list."""
        result = aggregator.aggregate([], [])
        assert result.final_decision == "no_winner"

    def test_aggregate_unanimous_agreement(self, aggregator: UnanimousAggregator):
        """Should return verdict when all agree."""
        verdicts = [
            Verdict(decision="winner", confidence=0.8, reasoning=""),
            Verdict(decision="winner", confidence=0.9, reasoning=""),
            Verdict(decision="winner", confidence=0.7, reasoning=""),
        ]
        specs = [
            SubDebateSpec(aspect=f"Aspect {i}", weight=1.0)
            for i in range(3)
        ]

        result = aggregator.aggregate(verdicts, specs)
        assert result.final_decision == "winner"
        assert result.confidence == pytest.approx(0.8, rel=0.01)  # Average

    def test_aggregate_no_consensus(self, aggregator: UnanimousAggregator):
        """Should return no_consensus when verdicts differ."""
        verdicts = [
            Verdict(decision="A", confidence=0.8, reasoning=""),
            Verdict(decision="B", confidence=0.9, reasoning=""),
        ]
        specs = [
            SubDebateSpec(aspect=f"Aspect {i}", weight=1.0)
            for i in range(2)
        ]

        result = aggregator.aggregate(verdicts, specs)
        assert result.final_decision == "no_consensus"
        assert result.confidence == 0.0


class TestWeightedMajorityAggregator:
    """Tests for WeightedMajorityAggregator."""

    @pytest.fixture
    def aggregator(self) -> WeightedMajorityAggregator:
        return WeightedMajorityAggregator()

    def test_method_property(self, aggregator: WeightedMajorityAggregator):
        """Should return WEIGHTED_MAJORITY method."""
        assert aggregator.method == AggregationMethod.WEIGHTED_MAJORITY

    def test_aggregate_empty_verdicts(self, aggregator: WeightedMajorityAggregator):
        """Should handle empty verdict list."""
        result = aggregator.aggregate([], [])
        assert result.final_decision == "no_winner"

    def test_aggregate_respects_weights(self, aggregator: WeightedMajorityAggregator):
        """Spec weights should affect outcome."""
        verdicts = [
            Verdict(decision="A", confidence=0.8, reasoning=""),
            Verdict(decision="B", confidence=0.8, reasoning=""),
        ]
        # Give much more weight to first verdict
        specs = [
            SubDebateSpec(aspect="Aspect 1", weight=0.9),
            SubDebateSpec(aspect="Aspect 2", weight=0.1),
        ]

        result = aggregator.aggregate(verdicts, specs)
        assert result.final_decision == "a"
        assert result.confidence == pytest.approx(0.9, rel=0.01)


class TestCreateAggregator:
    """Tests for create_aggregator factory function."""

    def test_create_weighted_average(self):
        """Should create WeightedAverageAggregator."""
        agg = create_aggregator(AggregationMethod.WEIGHTED_AVERAGE)
        assert isinstance(agg, WeightedAverageAggregator)

    def test_create_majority_vote(self):
        """Should create MajorityVoteAggregator."""
        agg = create_aggregator(AggregationMethod.MAJORITY_VOTE)
        assert isinstance(agg, MajorityVoteAggregator)

    def test_create_confidence_weighted(self):
        """Should create ConfidenceWeightedAggregator."""
        agg = create_aggregator(AggregationMethod.CONFIDENCE_WEIGHTED)
        assert isinstance(agg, ConfidenceWeightedAggregator)

    def test_create_unanimous(self):
        """Should create UnanimousAggregator."""
        agg = create_aggregator(AggregationMethod.UNANIMOUS)
        assert isinstance(agg, UnanimousAggregator)

    def test_create_weighted_majority(self):
        """Should create WeightedMajorityAggregator."""
        agg = create_aggregator(AggregationMethod.WEIGHTED_MAJORITY)
        assert isinstance(agg, WeightedMajorityAggregator)

    def test_create_invalid_method(self):
        """Should raise for unknown method."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            create_aggregator("invalid_method")
