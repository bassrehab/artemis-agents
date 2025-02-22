"""
Unit tests for Agent class.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from artemis.core.agent import Agent
from artemis.core.argument import ArgumentBuilder, ArgumentHierarchy, ArgumentParser
from artemis.core.types import (
    Argument,
    ArgumentLevel,
    CausalLink,
    DebateContext,
    Evidence,
    ModelResponse,
    ReasoningConfig,
    ReasoningResponse,
    Usage,
)
from artemis.exceptions import ArgumentGenerationError
from artemis.models import BaseModel


class TestArgumentBuilder:
    """Tests for ArgumentBuilder class."""

    def test_build_basic_argument(self) -> None:
        """Test building a basic argument."""
        arg = (
            ArgumentBuilder(agent="Proponent", level=ArgumentLevel.TACTICAL)
            .set_content("This is my argument.")
            .build()
        )

        assert arg.agent == "Proponent"
        assert arg.level == ArgumentLevel.TACTICAL
        assert arg.content == "This is my argument."
        assert arg.evidence == []
        assert arg.causal_links == []

    def test_build_with_evidence(self) -> None:
        """Test building argument with evidence."""
        evidence = Evidence(type="study", content="Study shows...", source="Nature")

        arg = (
            ArgumentBuilder(agent="Expert", level=ArgumentLevel.OPERATIONAL)
            .set_content("Based on research...")
            .add_evidence(evidence)
            .build()
        )

        assert len(arg.evidence) == 1
        assert arg.evidence[0].source == "Nature"

    def test_build_with_causal_links(self) -> None:
        """Test building argument with causal links."""
        link = CausalLink(cause="Action A", effect="Result B", strength=0.8)

        arg = (
            ArgumentBuilder(agent="Analyst", level=ArgumentLevel.TACTICAL)
            .set_content("A causes B...")
            .add_causal_link(link)
            .build()
        )

        assert len(arg.causal_links) == 1
        assert arg.causal_links[0].strength == 0.8

    def test_build_with_rebuttal(self) -> None:
        """Test building a rebuttal argument."""
        arg = (
            ArgumentBuilder(agent="Opponent", level=ArgumentLevel.TACTICAL)
            .set_content("I disagree because...")
            .set_rebuts("arg-123")
            .build()
        )

        assert arg.rebuts == "arg-123"


class TestArgumentParser:
    """Tests for ArgumentParser class."""

    @pytest.fixture
    def parser(self) -> ArgumentParser:
        return ArgumentParser()

    def test_parse_basic_content(self, parser: ArgumentParser) -> None:
        """Test parsing basic content."""
        content = "This is a simple argument."
        arg = parser.parse(content, agent="Test", level=ArgumentLevel.TACTICAL)

        assert arg.content == content
        assert arg.agent == "Test"
        assert arg.level == ArgumentLevel.TACTICAL

    def test_parse_with_citation(self, parser: ArgumentParser) -> None:
        """Test parsing content with citations."""
        content = "According to Smith, this is true. (Author, 2024)"
        arg = parser.parse(content, agent="Test", level=ArgumentLevel.TACTICAL)

        # Should extract citations as evidence
        assert len(arg.evidence) >= 1


class TestArgumentHierarchy:
    """Tests for ArgumentHierarchy class."""

    @pytest.fixture
    def hierarchy(self) -> ArgumentHierarchy:
        return ArgumentHierarchy()

    @pytest.fixture
    def strategic_arg(self) -> Argument:
        return Argument(
            agent="Agent1",
            level=ArgumentLevel.STRATEGIC,
            content="Strategic position",
        )

    @pytest.fixture
    def tactical_arg(self) -> Argument:
        return Argument(
            agent="Agent1",
            level=ArgumentLevel.TACTICAL,
            content="Tactical support",
        )

    def test_add_root_argument(self, hierarchy: ArgumentHierarchy, strategic_arg: Argument) -> None:
        """Test adding a root argument."""
        hierarchy.add(strategic_arg)
        assert len(hierarchy) == 1
        assert hierarchy.get(strategic_arg.id) == strategic_arg

    def test_add_child_argument(
        self,
        hierarchy: ArgumentHierarchy,
        strategic_arg: Argument,
        tactical_arg: Argument,
    ) -> None:
        """Test adding a child argument."""
        hierarchy.add(strategic_arg)
        hierarchy.add(tactical_arg, parent_id=strategic_arg.id)

        assert len(hierarchy) == 2
        children = hierarchy.get_children(strategic_arg.id)
        assert len(children) == 1
        assert children[0].id == tactical_arg.id

    def test_invalid_hierarchy(self, hierarchy: ArgumentHierarchy, strategic_arg: Argument) -> None:
        """Test that invalid hierarchy raises error."""
        hierarchy.add(strategic_arg)

        # Try to add strategic as child of strategic
        another_strategic = Argument(
            agent="Agent2",
            level=ArgumentLevel.STRATEGIC,
            content="Another strategic",
        )

        with pytest.raises(ValueError, match="Cannot add"):
            hierarchy.add(another_strategic, parent_id=strategic_arg.id)

    def test_get_by_level(
        self,
        hierarchy: ArgumentHierarchy,
        strategic_arg: Argument,
        tactical_arg: Argument,
    ) -> None:
        """Test getting arguments by level."""
        hierarchy.add(strategic_arg)
        hierarchy.add(tactical_arg)

        strategic = hierarchy.get_strategic_arguments()
        tactical = hierarchy.get_tactical_arguments()

        assert len(strategic) == 1
        assert len(tactical) == 1


class TestAgent:
    """Tests for Agent class."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.supports_reasoning = False
        model.generate = AsyncMock(
            return_value=ModelResponse(
                content="This is a generated argument. Because X, therefore Y.",
                usage=Usage(prompt_tokens=100, completion_tokens=50),
                model="gpt-4o-mock",
            )
        )
        return model

    @pytest.fixture
    def context(self) -> DebateContext:
        """Create a test debate context."""
        return DebateContext(
            topic="Should AI be regulated?",
            current_round=0,
            total_rounds=3,
            turn_in_round=0,
            agent_positions={"Proponent": "Yes", "Opponent": "No"},
        )

    def test_agent_init(self, mock_model: MagicMock) -> None:
        """Test agent initialization."""
        agent = Agent(
            name="TestAgent",
            role="Argues for the proposition",
            model=mock_model,
        )

        assert agent.name == "TestAgent"
        assert agent.role == "Argues for the proposition"
        assert agent.model == mock_model

    def test_agent_repr(self, mock_model: MagicMock) -> None:
        """Test agent string representation."""
        agent = Agent(name="TestAgent", role="Test role", model=mock_model)
        repr_str = repr(agent)

        assert "TestAgent" in repr_str
        assert "gpt-4o-mock" in repr_str

    @pytest.mark.asyncio
    async def test_generate_argument(self, mock_model: MagicMock, context: DebateContext) -> None:
        """Test generating an argument."""
        agent = Agent(name="Proponent", role="Argues in favor", model=mock_model)

        argument = await agent.generate_argument(context, ArgumentLevel.TACTICAL)

        assert argument.agent == "Proponent"
        assert argument.level == ArgumentLevel.TACTICAL
        assert len(argument.content) > 0
        mock_model.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_opening(self, mock_model: MagicMock, context: DebateContext) -> None:
        """Test generating opening statement."""
        agent = Agent(name="Proponent", role="Argues in favor", model=mock_model)

        argument = await agent.generate_opening(context)

        assert argument.agent == "Proponent"
        assert argument.level == ArgumentLevel.STRATEGIC

    @pytest.mark.asyncio
    async def test_generate_closing(self, mock_model: MagicMock, context: DebateContext) -> None:
        """Test generating closing statement."""
        agent = Agent(name="Proponent", role="Argues in favor", model=mock_model)

        argument = await agent.generate_closing(context)

        assert argument.agent == "Proponent"
        assert argument.level == ArgumentLevel.STRATEGIC

    @pytest.mark.asyncio
    async def test_generate_rebuttal(self, mock_model: MagicMock, context: DebateContext) -> None:
        """Test generating a rebuttal."""
        agent = Agent(name="Opponent", role="Argues against", model=mock_model)

        target = Argument(
            agent="Proponent",
            level=ArgumentLevel.TACTICAL,
            content="Original argument",
        )

        rebuttal = await agent.generate_rebuttal(context, target)

        assert rebuttal.agent == "Opponent"
        assert rebuttal.rebuts == target.id

    @pytest.mark.asyncio
    async def test_argument_history(self, mock_model: MagicMock, context: DebateContext) -> None:
        """Test that arguments are tracked in history."""
        agent = Agent(name="Proponent", role="Argues in favor", model=mock_model)

        assert len(agent.argument_history) == 0

        await agent.generate_argument(context, ArgumentLevel.TACTICAL)
        assert len(agent.argument_history) == 1

        await agent.generate_argument(context, ArgumentLevel.OPERATIONAL)
        assert len(agent.argument_history) == 2

    @pytest.mark.asyncio
    async def test_generate_with_reasoning(self, context: DebateContext) -> None:
        """Test generating with reasoning model."""
        mock_model = MagicMock(spec=BaseModel)
        mock_model.model = "o1-mock"
        mock_model.supports_reasoning = True
        mock_model.generate_with_reasoning = AsyncMock(
            return_value=ReasoningResponse(
                content="Reasoned argument",
                usage=Usage(prompt_tokens=100, completion_tokens=200),
                thinking="I thought about this...",
                thinking_tokens=150,
            )
        )

        agent = Agent(
            name="Reasoner",
            role="Deep thinker",
            model=mock_model,
            reasoning=ReasoningConfig(enabled=True, thinking_budget=10000),
        )

        argument = await agent.generate_argument(context, ArgumentLevel.STRATEGIC)

        assert argument.content == "Reasoned argument"
        mock_model.generate_with_reasoning.assert_called_once()

    @pytest.mark.asyncio
    async def test_generation_error(self, context: DebateContext) -> None:
        """Test that generation errors are properly wrapped."""
        mock_model = MagicMock(spec=BaseModel)
        mock_model.model = "failing-model"
        mock_model.supports_reasoning = False
        mock_model.generate = AsyncMock(side_effect=Exception("API Error"))

        agent = Agent(name="Failing", role="Fails", model=mock_model)

        with pytest.raises(ArgumentGenerationError) as exc_info:
            await agent.generate_argument(context, ArgumentLevel.TACTICAL)

        assert "API Error" in str(exc_info.value)
        assert exc_info.value.agent_name == "Failing"
