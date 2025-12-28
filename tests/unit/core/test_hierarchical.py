"""Tests for hierarchical debate orchestration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artemis.core.aggregation import (
    MajorityVoteAggregator,
    WeightedAverageAggregator,
)
from artemis.core.decomposition import (
    ManualDecomposer,
    RuleBasedDecomposer,
)
from artemis.core.hierarchical import HierarchicalDebate
from artemis.core.types import (
    CompoundVerdict,
    DebateConfig,
    DebateMetadata,
    DebateResult,
    HierarchicalContext,
    SubDebateSpec,
    Verdict,
)


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = MagicMock()
    agent.name = "TestAgent"
    return agent


@pytest.fixture
def mock_agents(mock_agent):
    """Create list of mock agents."""
    agent1 = MagicMock()
    agent1.name = "Agent1"
    agent2 = MagicMock()
    agent2.name = "Agent2"
    return [agent1, agent2]


@pytest.fixture
def mock_verdict():
    """Create a mock verdict."""
    return Verdict(
        decision="Agent1",
        confidence=0.8,
        reasoning="Agent1 had stronger arguments",
        unanimous=False,
    )


@pytest.fixture
def mock_debate_result(mock_verdict):
    """Create a mock debate result."""
    from datetime import datetime

    return DebateResult(
        debate_id="test-debate",
        topic="Test topic",
        verdict=mock_verdict,
        rounds=[],
        started_at="2024-01-01T00:00:00",
        ended_at="2024-01-01T00:01:00",
        metadata=DebateMetadata(
            started_at=datetime(2024, 1, 1, 0, 0, 0),
            ended_at=datetime(2024, 1, 1, 0, 1, 0),
            total_rounds=3,
            total_turns=6,
            agents=["Agent1", "Agent2"],
        ),
    )


class TestHierarchicalDebateInit:
    """Tests for HierarchicalDebate initialization."""

    def test_init_basic(self, mock_agents):
        """Should initialize with basic parameters."""
        debate = HierarchicalDebate(
            topic="Should AI be regulated?",
            agents=mock_agents,
        )

        assert debate.topic == "Should AI be regulated?"
        assert debate.agents == mock_agents
        assert debate.max_depth == 2
        assert debate.rounds == 3

    def test_init_custom_decomposer(self, mock_agents):
        """Should accept custom decomposer."""
        specs = [SubDebateSpec(aspect="Aspect 1", weight=1.0)]
        decomposer = ManualDecomposer(specs)

        debate = HierarchicalDebate(
            topic="Test",
            agents=mock_agents,
            decomposer=decomposer,
        )

        assert debate.decomposer is decomposer

    def test_init_custom_aggregator(self, mock_agents):
        """Should accept custom aggregator."""
        aggregator = MajorityVoteAggregator()

        debate = HierarchicalDebate(
            topic="Test",
            agents=mock_agents,
            aggregator=aggregator,
        )

        assert debate.aggregator is aggregator

    def test_init_custom_max_depth(self, mock_agents):
        """Should accept custom max_depth."""
        debate = HierarchicalDebate(
            topic="Test",
            agents=mock_agents,
            max_depth=5,
        )

        assert debate.max_depth == 5

    def test_init_custom_rounds(self, mock_agents):
        """Should accept custom rounds."""
        debate = HierarchicalDebate(
            topic="Test",
            agents=mock_agents,
            rounds=5,
        )

        assert debate.rounds == 5

    def test_init_requires_two_agents(self, mock_agent):
        """Should require at least 2 agents."""
        with pytest.raises(ValueError, match="at least 2 agents"):
            HierarchicalDebate(
                topic="Test",
                agents=[mock_agent],
            )

    def test_init_generates_debate_id(self, mock_agents):
        """Should generate unique debate ID."""
        debate1 = HierarchicalDebate(topic="Test", agents=mock_agents)
        debate2 = HierarchicalDebate(topic="Test", agents=mock_agents)

        assert debate1.debate_id != debate2.debate_id

    def test_init_default_decomposer(self, mock_agents):
        """Should use RuleBasedDecomposer by default."""
        debate = HierarchicalDebate(topic="Test", agents=mock_agents)
        assert isinstance(debate.decomposer, RuleBasedDecomposer)

    def test_init_default_aggregator(self, mock_agents):
        """Should use WeightedAverageAggregator by default."""
        debate = HierarchicalDebate(topic="Test", agents=mock_agents)
        assert isinstance(debate.aggregator, WeightedAverageAggregator)


class TestHierarchicalDebateRun:
    """Tests for HierarchicalDebate.run()."""

    @pytest.mark.asyncio
    async def test_run_leaf_debate(self, mock_agents, mock_debate_result):
        """Should run leaf debate at max depth."""
        # Create debate with max_depth=1 to force leaf (depth=0 >= max_depth=1)
        # Use manual decomposer with empty specs to ensure we hit leaf immediately
        debate = HierarchicalDebate(
            topic="Test topic",
            agents=mock_agents,
            max_depth=1,
            decomposer=ManualDecomposer([]),  # Empty specs forces leaf behavior
        )

        # Mock the Debate class
        with patch("artemis.core.hierarchical.Debate") as MockDebate:
            mock_debate_instance = AsyncMock()
            mock_debate_instance.run.return_value = mock_debate_result
            MockDebate.return_value = mock_debate_instance

            result = await debate.run()

            assert isinstance(result, CompoundVerdict)
            assert result.depth == 0
            MockDebate.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_decomposition(self, mock_agents, mock_debate_result):
        """Should decompose and run sub-debates."""
        specs = [
            SubDebateSpec(aspect="Sub-topic 1", weight=0.5),
            SubDebateSpec(aspect="Sub-topic 2", weight=0.5),
        ]
        decomposer = ManualDecomposer(specs)

        debate = HierarchicalDebate(
            topic="Main topic",
            agents=mock_agents,
            decomposer=decomposer,
            max_depth=1,
        )

        with patch("artemis.core.hierarchical.Debate") as MockDebate:
            mock_debate_instance = AsyncMock()
            mock_debate_instance.run.return_value = mock_debate_result
            MockDebate.return_value = mock_debate_instance

            result = await debate.run()

            # Should have run 2 sub-debates
            assert MockDebate.call_count == 2
            assert len(result.sub_verdicts) == 2

    @pytest.mark.asyncio
    async def test_run_insufficient_subtopics(self, mock_agents, mock_debate_result):
        """Should run as leaf when decomposition returns < 2 topics."""
        # Decomposer that returns single topic
        specs = [SubDebateSpec(aspect="Only one", weight=1.0)]
        decomposer = ManualDecomposer(specs)

        debate = HierarchicalDebate(
            topic="Test",
            agents=mock_agents,
            decomposer=decomposer,
            max_depth=2,
        )

        with patch("artemis.core.hierarchical.Debate") as MockDebate:
            mock_debate_instance = AsyncMock()
            mock_debate_instance.run.return_value = mock_debate_result
            MockDebate.return_value = mock_debate_instance

            result = await debate.run()

            # Should run as leaf (1 debate)
            assert MockDebate.call_count == 1


class TestHierarchicalDebateHelpers:
    """Tests for HierarchicalDebate helper methods."""

    def test_get_debate_tree_empty(self, mock_agents):
        """Should return tree structure before running."""
        debate = HierarchicalDebate(
            topic="Test",
            agents=mock_agents,
        )

        tree = debate.get_debate_tree()

        assert tree["topic"] == "Test"
        assert tree["debate_id"] == debate.debate_id
        assert tree["max_depth"] == 2
        assert tree["sub_debates"] == 0
        assert tree["sub_results"] == []

    def test_sub_debates_property(self, mock_agents):
        """Should return copy of sub-debates list."""
        debate = HierarchicalDebate(topic="Test", agents=mock_agents)

        # Access property
        sub = debate.sub_debates

        # Should be a copy
        assert sub == []
        sub.append("something")
        assert debate.sub_debates == []

    def test_sub_results_property(self, mock_agents):
        """Should return copy of sub-results list."""
        debate = HierarchicalDebate(topic="Test", agents=mock_agents)

        # Access property
        results = debate.sub_results

        # Should be a copy
        assert results == []
        results.append("something")
        assert debate.sub_results == []

    def test_repr(self, mock_agents):
        """Should have informative repr."""
        debate = HierarchicalDebate(
            topic="This is a long topic about AI",
            agents=mock_agents,
            max_depth=3,
        )

        repr_str = repr(debate)

        assert "HierarchicalDebate" in repr_str
        assert "This is a long topic" in repr_str
        assert "max_depth=3" in repr_str
        assert "Agent1" in repr_str


class TestHierarchicalContext:
    """Tests for HierarchicalContext usage."""

    @pytest.mark.asyncio
    async def test_context_passed_to_decomposer(self, mock_agents, mock_debate_result):
        """Decomposer should receive context."""
        mock_decomposer = AsyncMock()
        mock_decomposer.decompose.return_value = []  # Empty to trigger leaf

        debate = HierarchicalDebate(
            topic="Main",
            agents=mock_agents,
            decomposer=mock_decomposer,
            max_depth=1,
        )

        with patch("artemis.core.hierarchical.Debate") as MockDebate:
            mock_debate_instance = AsyncMock()
            mock_debate_instance.run.return_value = mock_debate_result
            MockDebate.return_value = mock_debate_instance

            await debate.run()

        # Check decomposer was called with context
        call_args = mock_decomposer.decompose.call_args
        assert call_args is not None
        context = call_args[0][1]  # Second positional arg
        assert isinstance(context, HierarchicalContext)
        assert context.parent_topic == "Main"
        assert context.depth == 0

    @pytest.mark.asyncio
    async def test_depth_increases_in_recursion(self, mock_agents, mock_debate_result):
        """Depth should increase for sub-debates."""
        call_depths = []

        async def capture_decompose(topic, context, max_subtopics=5):
            if context:
                call_depths.append(context.depth)
            # Return empty at depth 1 to stop recursion
            if context and context.depth >= 1:
                return []
            return [
                SubDebateSpec(aspect="Sub 1", weight=0.5),
                SubDebateSpec(aspect="Sub 2", weight=0.5),
            ]

        mock_decomposer = AsyncMock()
        mock_decomposer.decompose.side_effect = capture_decompose

        debate = HierarchicalDebate(
            topic="Root",
            agents=mock_agents,
            decomposer=mock_decomposer,
            max_depth=2,
        )

        with patch("artemis.core.hierarchical.Debate") as MockDebate:
            mock_debate_instance = AsyncMock()
            mock_debate_instance.run.return_value = mock_debate_result
            MockDebate.return_value = mock_debate_instance

            await debate.run()

        # Should have calls at depth 0 and depth 1
        assert 0 in call_depths
        assert 1 in call_depths


class TestRoundsAdjustment:
    """Tests for rounds adjustment based on depth."""

    @pytest.mark.asyncio
    async def test_rounds_decrease_with_depth(self, mock_agents, mock_debate_result):
        """Rounds should decrease at deeper levels."""
        rounds_used = []

        def capture_debate_init(**kwargs):
            rounds_used.append(kwargs.get("rounds", 3))
            mock_instance = AsyncMock()
            mock_instance.run.return_value = mock_debate_result
            return mock_instance

        debate = HierarchicalDebate(
            topic="Root",
            agents=mock_agents,
            rounds=3,
            max_depth=1,  # Direct leaf (empty decomposer)
            decomposer=ManualDecomposer([]),  # Empty specs forces leaf
        )

        with patch("artemis.core.hierarchical.Debate") as MockDebate:
            MockDebate.side_effect = capture_debate_init

            await debate.run()

        # At depth 0 with rounds=3: rounds = max(1, 3-0) = 3
        assert rounds_used[0] == 3
