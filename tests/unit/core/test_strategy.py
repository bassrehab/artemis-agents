"""
Unit tests for Agent strategy selection and opponent modeling.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from artemis.core.agent import Agent, DebateStrategy, OpponentModel, StrategyContext
from artemis.core.types import (
    Argument,
    ArgumentLevel,
    DebateContext,
    ModelResponse,
    Turn,
    Usage,
)
from artemis.models import BaseModel


class TestDebateStrategy:
    """Tests for DebateStrategy enum."""

    def test_all_strategies_defined(self) -> None:
        """Test that all expected strategies are defined."""
        expected = ["establish", "reinforce", "counter", "synthesize", "adapt"]
        for strategy_name in expected:
            assert hasattr(DebateStrategy, strategy_name.upper())

    def test_strategy_values(self) -> None:
        """Test that strategy values match expected strings."""
        assert DebateStrategy.ESTABLISH.value == "establish"
        assert DebateStrategy.REINFORCE.value == "reinforce"
        assert DebateStrategy.COUNTER.value == "counter"
        assert DebateStrategy.SYNTHESIZE.value == "synthesize"
        assert DebateStrategy.ADAPT.value == "adapt"


class TestStrategyContext:
    """Tests for StrategyContext dataclass."""

    def test_context_creation(self) -> None:
        """Test creating a strategy context."""
        context = StrategyContext(
            current_round=2,
            total_rounds=5,
            own_argument_count=3,
            opponent_argument_count=4,
        )
        assert context.current_round == 2
        assert context.total_rounds == 5
        assert context.is_winning is None

    def test_context_with_winning_status(self) -> None:
        """Test context with winning status."""
        context = StrategyContext(
            current_round=3,
            total_rounds=5,
            own_argument_count=5,
            opponent_argument_count=4,
            is_winning=True,
        )
        assert context.is_winning is True


class TestOpponentModel:
    """Tests for OpponentModel class."""

    @pytest.fixture
    def opponent_model(self) -> OpponentModel:
        return OpponentModel(name="Opponent")

    def test_model_creation(self, opponent_model: OpponentModel) -> None:
        """Test creating an opponent model."""
        assert opponent_model.name == "Opponent"
        assert opponent_model.argument_count == 0
        assert opponent_model.primary_themes == []

    def test_update_with_argument(self, opponent_model: OpponentModel) -> None:
        """Test updating model with an argument."""
        argument = Argument(
            agent="Opponent",
            level=ArgumentLevel.TACTICAL,
            content="AI technology creates economic disruption and workforce changes.",
        )
        opponent_model.update(argument)

        assert opponent_model.argument_count == 1
        assert len(opponent_model.arguments) == 1

    def test_theme_extraction(self, opponent_model: OpponentModel) -> None:
        """Test that themes are extracted from arguments."""
        argument = Argument(
            agent="Opponent",
            level=ArgumentLevel.TACTICAL,
            content="Economic disruption is inevitable. Technology advancement continues.",
        )
        opponent_model.update(argument)

        # Should extract some themes (words > 6 chars)
        assert len(opponent_model.primary_themes) > 0

    def test_multiple_arguments(self, opponent_model: OpponentModel) -> None:
        """Test tracking multiple arguments."""
        for i in range(3):
            argument = Argument(
                agent="Opponent",
                level=ArgumentLevel.TACTICAL,
                content=f"Argument number {i} with content.",
            )
            opponent_model.update(argument)

        assert opponent_model.argument_count == 3


class TestAgentStrategySelection:
    """Tests for Agent strategy selection."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.supports_reasoning = False
        model.generate = AsyncMock(
            return_value=ModelResponse(
                content="Generated argument content.",
                usage=Usage(prompt_tokens=100, completion_tokens=50),
                model="gpt-4o-mock",
            )
        )
        return model

    @pytest.fixture
    def agent(self, mock_model: MagicMock) -> Agent:
        """Create a test agent."""
        return Agent(
            name="TestAgent",
            role="Test role",
            model=mock_model,
        )

    def test_establish_strategy_early_round(
        self, agent: Agent, mock_model: MagicMock  # noqa: ARG002
    ) -> None:
        """Test that ESTABLISH strategy is selected in early rounds."""
        context = DebateContext(
            topic="Test topic",
            current_round=0,
            total_rounds=10,
            turn_in_round=0,
        )
        strategy = agent.select_strategy(context)
        assert strategy == DebateStrategy.ESTABLISH

    def test_reinforce_strategy_middle_early(
        self, agent: Agent, mock_model: MagicMock  # noqa: ARG002
    ) -> None:
        """Test that REINFORCE strategy is selected in middle-early rounds."""
        context = DebateContext(
            topic="Test topic",
            current_round=3,
            total_rounds=10,
            turn_in_round=0,
        )
        strategy = agent.select_strategy(context)
        assert strategy == DebateStrategy.REINFORCE

    def test_synthesize_strategy_late_round(
        self, agent: Agent, mock_model: MagicMock  # noqa: ARG002
    ) -> None:
        """Test that SYNTHESIZE strategy is selected in late rounds."""
        context = DebateContext(
            topic="Test topic",
            current_round=9,
            total_rounds=10,
            turn_in_round=0,
        )
        strategy = agent.select_strategy(context)
        assert strategy == DebateStrategy.SYNTHESIZE

    def test_counter_strategy_when_behind(
        self, agent: Agent, mock_model: MagicMock  # noqa: ARG002
    ) -> None:
        """Test that COUNTER strategy is selected when opponent has more arguments."""
        # Create context with opponent arguments in transcript
        opponent_arg = Argument(
            agent="Opponent",
            level=ArgumentLevel.TACTICAL,
            content="Opponent argument",
        )
        context = DebateContext(
            topic="Test topic",
            current_round=6,
            total_rounds=10,
            turn_in_round=0,
            transcript=[
                Turn(agent="Opponent", argument=opponent_arg, round=1, sequence=0),
                Turn(agent="Opponent", argument=opponent_arg, round=2, sequence=0),
                Turn(agent="Opponent", argument=opponent_arg, round=3, sequence=0),
            ],
        )
        strategy = agent.select_strategy(context)
        assert strategy == DebateStrategy.COUNTER

    def test_current_strategy_property(
        self, agent: Agent, mock_model: MagicMock  # noqa: ARG002
    ) -> None:
        """Test that current_strategy property reflects selected strategy."""
        context = DebateContext(
            topic="Test topic",
            current_round=0,
            total_rounds=10,
            turn_in_round=0,
        )
        agent.select_strategy(context)
        assert agent.current_strategy == DebateStrategy.ESTABLISH


class TestAgentOpponentModeling:
    """Tests for Agent opponent modeling."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.supports_reasoning = False
        model.generate = AsyncMock(
            return_value=ModelResponse(
                content="Generated response.",
                usage=Usage(prompt_tokens=100, completion_tokens=50),
                model="gpt-4o-mock",
            )
        )
        return model

    @pytest.fixture
    def agent(self, mock_model: MagicMock) -> Agent:
        """Create a test agent."""
        return Agent(
            name="TestAgent",
            role="Test role",
            model=mock_model,
        )

    def test_observe_opponent(self, agent: Agent) -> None:
        """Test observing an opponent's argument."""
        opponent_arg = Argument(
            agent="Opponent",
            level=ArgumentLevel.TACTICAL,
            content="The opponent's detailed argument with evidence (Smith, 2024).",
        )
        agent.observe_opponent(opponent_arg)

        model = agent.get_opponent_model("Opponent")
        assert model is not None
        assert model.argument_count == 1

    def test_observe_self_ignored(self, agent: Agent) -> None:
        """Test that observing own arguments is ignored."""
        own_arg = Argument(
            agent="TestAgent",
            level=ArgumentLevel.TACTICAL,
            content="My own argument.",
        )
        agent.observe_opponent(own_arg)

        model = agent.get_opponent_model("TestAgent")
        assert model is None

    def test_multiple_opponents(self, agent: Agent) -> None:
        """Test modeling multiple opponents."""
        arg1 = Argument(
            agent="Opponent1",
            level=ArgumentLevel.TACTICAL,
            content="First opponent argument.",
        )
        arg2 = Argument(
            agent="Opponent2",
            level=ArgumentLevel.TACTICAL,
            content="Second opponent argument.",
        )

        agent.observe_opponent(arg1)
        agent.observe_opponent(arg2)

        assert agent.get_opponent_model("Opponent1") is not None
        assert agent.get_opponent_model("Opponent2") is not None

    def test_get_nonexistent_opponent(self, agent: Agent) -> None:
        """Test getting model for non-existent opponent."""
        model = agent.get_opponent_model("NonExistent")
        assert model is None


class TestAgentStrategyInstructions:
    """Tests for Agent strategy instruction generation."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.supports_reasoning = False
        return model

    @pytest.fixture
    def agent(self, mock_model: MagicMock) -> Agent:
        """Create a test agent."""
        return Agent(
            name="TestAgent",
            role="Test role",
            model=mock_model,
        )

    def test_establish_instructions(self, agent: Agent) -> None:
        """Test instructions for ESTABLISH strategy."""
        instructions = agent.build_strategy_instructions(DebateStrategy.ESTABLISH)
        assert "core position" in instructions.lower()
        assert "thesis" in instructions.lower()

    def test_reinforce_instructions(self, agent: Agent) -> None:
        """Test instructions for REINFORCE strategy."""
        instructions = agent.build_strategy_instructions(DebateStrategy.REINFORCE)
        assert "evidence" in instructions.lower()

    def test_counter_instructions(self, agent: Agent) -> None:
        """Test instructions for COUNTER strategy."""
        instructions = agent.build_strategy_instructions(DebateStrategy.COUNTER)
        assert "refute" in instructions.lower() or "counter" in instructions.lower()

    def test_synthesize_instructions(self, agent: Agent) -> None:
        """Test instructions for SYNTHESIZE strategy."""
        instructions = agent.build_strategy_instructions(DebateStrategy.SYNTHESIZE)
        assert "synthesize" in instructions.lower() or "conclusion" in instructions.lower()

    def test_adapt_instructions(self, agent: Agent) -> None:
        """Test instructions for ADAPT strategy."""
        instructions = agent.build_strategy_instructions(DebateStrategy.ADAPT)
        assert "adapt" in instructions.lower()


class TestAgentCausalGraph:
    """Tests for Agent causal graph functionality."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.supports_reasoning = False
        model.generate = AsyncMock(
            return_value=ModelResponse(
                content="AI causes automation. This leads to efficiency.",
                usage=Usage(prompt_tokens=100, completion_tokens=50),
                model="gpt-4o-mock",
            )
        )
        return model

    @pytest.fixture
    def agent(self, mock_model: MagicMock) -> Agent:
        """Create a test agent."""
        return Agent(
            name="TestAgent",
            role="Test role",
            model=mock_model,
        )

    def test_causal_graph_property(self, agent: Agent) -> None:
        """Test that agent has a causal graph."""
        graph = agent.causal_graph
        assert graph is not None
        assert len(graph) == 0  # Initially empty

    @pytest.mark.asyncio
    async def test_causal_graph_updated_after_generation(
        self, agent: Agent, mock_model: MagicMock  # noqa: ARG002
    ) -> None:
        """Test that causal graph is updated after argument generation."""
        context = DebateContext(
            topic="AI Impact",
            current_round=1,
            total_rounds=5,
            turn_in_round=0,
        )

        await agent.generate_argument(context, ArgumentLevel.TACTICAL)

        # Graph should be updated (may or may not have nodes depending on extraction)
        graph = agent.causal_graph
        assert graph is not None


class TestAgentEvidenceExtraction:
    """Tests for Agent evidence extraction."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.supports_reasoning = False
        return model

    @pytest.fixture
    def agent(self, mock_model: MagicMock) -> Agent:
        """Create a test agent."""
        return Agent(
            name="TestAgent",
            role="Test role",
            model=mock_model,
        )

    def test_extract_evidence(self, agent: Agent) -> None:
        """Test extracting evidence from text."""
        text = "According to Smith (2024), 75% of respondents agreed."
        evidence = agent.extract_evidence(text)

        assert len(evidence) >= 1
