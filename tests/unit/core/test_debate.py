"""
Unit tests for Debate orchestrator.
"""

from collections.abc import Generator
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artemis.core.agent import Agent
from artemis.core.debate import Debate, DebateError, DebateHaltedError
from artemis.core.types import (
    Argument,
    ArgumentEvaluation,
    ArgumentLevel,
    DebateConfig,
    DebateContext,
    DebateState,
    SafetyIndicator,
    SafetyIndicatorType,
    SafetyResult,
    Turn,
    Verdict,
)
from artemis.models import BaseModel


@pytest.fixture(autouse=True)
def mock_jury_panel() -> Generator[MagicMock, None, None]:
    """Mock JuryPanel for all tests to avoid API key requirements."""
    with patch("artemis.core.debate.JuryPanel") as MockJury:
        mock_jury = MagicMock()
        mock_jury.__len__ = MagicMock(return_value=3)
        mock_jury.deliberate = AsyncMock(
            return_value=Verdict(
                decision="Pro",
                confidence=0.8,
                reasoning="Test verdict",
                unanimous=True,
            )
        )
        MockJury.return_value = mock_jury
        yield MockJury


class TestDebateInit:
    """Tests for Debate initialization."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.supports_reasoning = False
        return model

    @pytest.fixture
    def mock_agents(self, mock_model: MagicMock) -> list[Agent]:
        """Create mock agents."""
        agent1 = Agent(name="Pro", role="Argues in favor", model=mock_model)
        agent2 = Agent(name="Con", role="Argues against", model=mock_model)
        return [agent1, agent2]

    def test_debate_creation(self, mock_agents: list[Agent]) -> None:
        """Test basic debate creation."""
        debate = Debate(
            topic="AI Ethics",
            agents=mock_agents,
            rounds=5,
        )
        assert debate.topic == "AI Ethics"
        assert len(debate.agents) == 2
        assert debate.total_rounds == 5
        assert debate.state == DebateState.SETUP

    def test_debate_requires_two_agents(self, mock_agents: list[Agent]) -> None:
        """Test that debate requires at least 2 agents."""
        with pytest.raises(ValueError, match="at least 2 agents"):
            Debate(
                topic="Test",
                agents=[mock_agents[0]],
                rounds=3,
            )

    def test_debate_with_custom_config(self, mock_agents: list[Agent]) -> None:
        """Test debate with custom configuration."""
        config = DebateConfig(
            turn_timeout=120,
            max_argument_tokens=2000,
            safety_mode="active",
            halt_on_safety_violation=True,
        )
        debate = Debate(
            topic="Test",
            agents=mock_agents,
            rounds=3,
            config=config,
        )
        assert debate.config.turn_timeout == 120
        assert debate.config.halt_on_safety_violation is True

    def test_debate_id_assigned(self, mock_agents: list[Agent]) -> None:
        """Test that unique debate ID is assigned."""
        debate1 = Debate(topic="Test1", agents=mock_agents, rounds=3)
        debate2 = Debate(topic="Test2", agents=mock_agents, rounds=3)
        assert debate1.debate_id != debate2.debate_id

    def test_debate_repr(self, mock_agents: list[Agent]) -> None:
        """Test string representation."""
        debate = Debate(topic="A very long topic name", agents=mock_agents, rounds=3)
        repr_str = repr(debate)
        assert "Debate" in repr_str
        assert "setup" in repr_str


class TestDebatePositions:
    """Tests for assigning debate positions."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.supports_reasoning = False
        return model

    @pytest.fixture
    def mock_agents(self, mock_model: MagicMock) -> list[Agent]:
        """Create mock agents."""
        agent1 = Agent(name="Pro", role="Argues in favor", model=mock_model)
        agent2 = Agent(name="Con", role="Argues against", model=mock_model)
        return [agent1, agent2]

    def test_assign_positions(self, mock_agents: list[Agent]) -> None:
        """Test assigning positions to agents."""
        debate = Debate(topic="Test", agents=mock_agents, rounds=3)
        debate.assign_positions({"Pro": "in favor", "Con": "against"})

        assert debate._agent_positions["Pro"] == "in favor"
        assert debate._agent_positions["Con"] == "against"

    def test_positions_set_on_agents(self, mock_agents: list[Agent]) -> None:
        """Test that positions are also set on agent objects."""
        debate = Debate(topic="Test", agents=mock_agents, rounds=3)
        debate.assign_positions({"Pro": "in favor", "Con": "against"})

        pro_agent = debate.get_agent("Pro")
        assert pro_agent is not None
        assert pro_agent.position == "in favor"


class TestDebateHelperMethods:
    """Tests for Debate helper methods."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.supports_reasoning = False
        return model

    @pytest.fixture
    def mock_agents(self, mock_model: MagicMock) -> list[Agent]:
        """Create mock agents."""
        agent1 = Agent(name="Pro", role="Argues in favor", model=mock_model)
        agent2 = Agent(name="Con", role="Argues against", model=mock_model)
        return [agent1, agent2]

    @pytest.fixture
    def debate(self, mock_agents: list[Agent]) -> Debate:
        """Create a debate."""
        return Debate(topic="Test Topic", agents=mock_agents, rounds=5)

    def test_get_agent(self, debate: Debate) -> None:
        """Test getting agent by name."""
        agent = debate.get_agent("Pro")
        assert agent is not None
        assert agent.name == "Pro"

    def test_get_agent_not_found(self, debate: Debate) -> None:
        """Test getting non-existent agent."""
        agent = debate.get_agent("NonExistent")
        assert agent is None

    def test_transcript_property(self, debate: Debate) -> None:
        """Test transcript property returns copy."""
        transcript = debate.transcript
        assert isinstance(transcript, list)
        assert len(transcript) == 0

    def test_current_round_property(self, debate: Debate) -> None:
        """Test current round property."""
        assert debate.current_round == 0

    def test_safety_alerts_property(self, debate: Debate) -> None:
        """Test safety alerts property."""
        alerts = debate.safety_alerts
        assert isinstance(alerts, list)
        assert len(alerts) == 0

    def test_get_round_level_early(self, debate: Debate) -> None:
        """Test argument level for early rounds.

        With adaptive selection and no transcript, defaults to strategic.
        """
        level = debate._get_round_level(1)
        assert level == ArgumentLevel.STRATEGIC

    def test_get_round_level_middle_adaptive(self, debate: Debate) -> None:
        """Test that adaptive level selection is active in BALANCED mode.

        Without transcript, fallback to round-based returns strategic for
        rounds 1-3 (progress <= 0.7 with strategic disagreement).
        """
        level = debate._get_round_level(3)
        # With empty transcript, disagreeement defaults to STRATEGIC
        # and recommend_level returns STRATEGIC for progress <= 0.7
        assert level == ArgumentLevel.STRATEGIC

    def test_get_round_level_late_adaptive(self, debate: Debate) -> None:
        """Test adaptive level selection for late rounds.

        Without transcript and strategic disagreement, returns TACTICAL
        for late rounds (progress > 0.7).
        """
        level = debate._get_round_level(5)
        # With empty transcript, disagreement defaults to STRATEGIC
        # and recommend_level returns TACTICAL for progress > 0.7
        assert level == ArgumentLevel.TACTICAL

    def test_mechanical_level_early(self, debate: Debate) -> None:
        """Test mechanical level selection for early rounds."""
        level = debate._mechanical_level(1)
        assert level == ArgumentLevel.STRATEGIC

    def test_mechanical_level_middle(self, debate: Debate) -> None:
        """Test mechanical level selection for middle rounds."""
        level = debate._mechanical_level(3)
        assert level == ArgumentLevel.TACTICAL

    def test_mechanical_level_late(self, debate: Debate) -> None:
        """Test mechanical level selection for late rounds."""
        level = debate._mechanical_level(5)
        assert level == ArgumentLevel.OPERATIONAL


class TestDebateContext:
    """Tests for debate context building."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.supports_reasoning = False
        return model

    @pytest.fixture
    def mock_agents(self, mock_model: MagicMock) -> list[Agent]:
        """Create mock agents."""
        agent1 = Agent(name="Pro", role="Argues in favor", model=mock_model)
        agent2 = Agent(name="Con", role="Argues against", model=mock_model)
        return [agent1, agent2]

    def test_build_context(self, mock_agents: list[Agent]) -> None:
        """Test building debate context."""
        debate = Debate(topic="Test Topic", agents=mock_agents, rounds=5)
        context = debate._build_context()

        assert isinstance(context, DebateContext)
        assert context.topic == "Test Topic"
        assert context.current_round == 0
        assert context.total_rounds == 5

    def test_context_includes_positions(self, mock_agents: list[Agent]) -> None:
        """Test that context includes agent positions."""
        debate = Debate(topic="Test", agents=mock_agents, rounds=3)
        debate.assign_positions({"Pro": "in favor", "Con": "against"})
        context = debate._build_context()

        assert context.agent_positions["Pro"] == "in favor"


class TestDebateSafetyMonitors:
    """Tests for safety monitor integration."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.supports_reasoning = False
        return model

    @pytest.fixture
    def mock_agents(self, mock_model: MagicMock) -> list[Agent]:
        """Create mock agents."""
        agent1 = Agent(name="Pro", role="Argues in favor", model=mock_model)
        agent2 = Agent(name="Con", role="Argues against", model=mock_model)
        return [agent1, agent2]

    def test_add_safety_monitor(self, mock_agents: list[Agent]) -> None:
        """Test adding a safety monitor."""
        debate = Debate(topic="Test", agents=mock_agents, rounds=3)

        def monitor(_turn: Turn, _context: DebateContext) -> SafetyResult:
            return SafetyResult(monitor="test", severity=0.0)

        debate.add_safety_monitor(monitor)
        assert len(debate._safety_monitors) == 1

    @pytest.mark.asyncio
    async def test_safety_monitor_called(self, mock_agents: list[Agent]) -> None:
        """Test that safety monitors are called during processing."""
        debate = Debate(topic="Test", agents=mock_agents, rounds=3)

        monitor_called = False

        def monitor(_turn: Turn, _context: DebateContext) -> SafetyResult:
            nonlocal monitor_called
            monitor_called = True
            return SafetyResult(monitor="test", severity=0.0)

        debate.add_safety_monitor(monitor)

        # Create a turn to process
        turn = Turn(
            round=1,
            sequence=0,
            agent="Pro",
            argument=Argument(
                agent="Pro",
                level=ArgumentLevel.STRATEGIC,
                content="Test argument",
            ),
        )

        context = debate._build_context()

        # Mock the evaluator
        with patch.object(debate._evaluator, "evaluate_argument") as mock_eval:
            mock_eval.return_value = ArgumentEvaluation(
                argument_id="arg_0",
                scores={"logic": 0.8},
                weights={"logic": 1.0},
                causal_score=0.8,
                total_score=0.8,
            )

            await debate._process_turn(turn, context)

        assert monitor_called

    @pytest.mark.asyncio
    async def test_safety_alert_raised(self, mock_agents: list[Agent]) -> None:
        """Test that safety alerts are collected."""
        debate = Debate(topic="Test", agents=mock_agents, rounds=3)

        def monitor(_turn: Turn, _context: DebateContext) -> SafetyResult:
            return SafetyResult(
                monitor="test_monitor",
                severity=0.8,
                should_alert=True,
                indicators=[
                    SafetyIndicator(
                        type=SafetyIndicatorType.FACTUAL_INCONSISTENCY,
                        severity=0.8,
                        evidence="Found inconsistency",
                    )
                ],
            )

        debate.add_safety_monitor(monitor)

        turn = Turn(
            round=1,
            sequence=0,
            agent="Pro",
            argument=Argument(
                agent="Pro",
                level=ArgumentLevel.STRATEGIC,
                content="Test",
            ),
        )

        context = debate._build_context()

        with patch.object(debate._evaluator, "evaluate_argument") as mock_eval:
            mock_eval.return_value = ArgumentEvaluation(
                argument_id="arg_0",
                scores={},
                weights={},
                causal_score=0.5,
                total_score=0.5,
            )

            await debate._process_turn(turn, context)

        assert len(debate.safety_alerts) == 1
        assert debate.safety_alerts[0].monitor == "test_monitor"

    @pytest.mark.asyncio
    async def test_safety_halt(self, mock_agents: list[Agent]) -> None:
        """Test that debate halts on safety violation when configured."""
        config = DebateConfig(halt_on_safety_violation=True)
        debate = Debate(
            topic="Test",
            agents=mock_agents,
            rounds=3,
            config=config,
        )

        def monitor(_turn: Turn, _context: DebateContext) -> SafetyResult:
            return SafetyResult(
                monitor="critical_monitor",
                severity=1.0,
                should_halt=True,
            )

        debate.add_safety_monitor(monitor)

        turn = Turn(
            round=1,
            sequence=0,
            agent="Pro",
            argument=Argument(
                agent="Pro",
                level=ArgumentLevel.STRATEGIC,
                content="Test",
            ),
        )

        context = debate._build_context()

        with patch.object(debate._evaluator, "evaluate_argument") as mock_eval:
            mock_eval.return_value = ArgumentEvaluation(
                argument_id="arg_0",
                scores={},
                weights={},
                causal_score=0.5,
                total_score=0.5,
            )

            with pytest.raises(DebateHaltedError) as exc_info:
                await debate._process_turn(turn, context)

            assert exc_info.value.alert.monitor == "critical_monitor"


class TestDebateScoring:
    """Tests for debate scoring."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.supports_reasoning = False
        return model

    @pytest.fixture
    def mock_agents(self, mock_model: MagicMock) -> list[Agent]:
        """Create mock agents."""
        agent1 = Agent(name="Pro", role="Argues in favor", model=mock_model)
        agent2 = Agent(name="Con", role="Argues against", model=mock_model)
        return [agent1, agent2]

    def test_get_scores_empty(self, mock_agents: list[Agent]) -> None:
        """Test getting scores with no turns."""
        debate = Debate(topic="Test", agents=mock_agents, rounds=3)
        scores = debate.get_scores()

        assert "Pro" in scores
        assert "Con" in scores
        assert scores["Pro"] == 0.0
        assert scores["Con"] == 0.0

    def test_get_scores_with_turns(self, mock_agents: list[Agent]) -> None:
        """Test getting scores with recorded turns."""
        debate = Debate(topic="Test", agents=mock_agents, rounds=3)

        # Manually add turns to transcript
        turn1 = Turn(
            round=1,
            sequence=0,
            agent="Pro",
            argument=Argument(
                agent="Pro",
                level=ArgumentLevel.STRATEGIC,
                content="Test",
            ),
            evaluation=ArgumentEvaluation(
                argument_id="arg_0",
                scores={"logic": 0.9},
                weights={"logic": 1.0},
                causal_score=0.9,
                total_score=0.9,
            ),
        )
        turn2 = Turn(
            round=1,
            sequence=1,
            agent="Con",
            argument=Argument(
                agent="Con",
                level=ArgumentLevel.STRATEGIC,
                content="Test",
            ),
            evaluation=ArgumentEvaluation(
                argument_id="arg_1",
                scores={"logic": 0.7},
                weights={"logic": 1.0},
                causal_score=0.7,
                total_score=0.7,
            ),
        )

        debate._transcript.append(turn1)
        debate._transcript.append(turn2)

        scores = debate.get_scores()

        assert scores["Pro"] == 0.9
        assert scores["Con"] == 0.7


class TestDebateRun:
    """Tests for debate execution."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.supports_reasoning = False
        return model

    @pytest.fixture
    def mock_agents(self, mock_model: MagicMock) -> list[Agent]:
        """Create mock agents with mocked generate."""
        agent1 = Agent(name="Pro", role="Argues in favor", model=mock_model)
        agent2 = Agent(name="Con", role="Argues against", model=mock_model)
        return [agent1, agent2]

    @pytest.mark.asyncio
    async def test_run_complete_debate(self, mock_agents: list[Agent]) -> None:
        """Test running a complete debate."""
        with patch("artemis.core.debate.JuryPanel") as MockJury:
            # Mock jury deliberation
            mock_jury = MagicMock()
            mock_jury.deliberate = AsyncMock(
                return_value=Verdict(
                    decision="Pro",
                    confidence=0.85,
                    reasoning="Pro presented stronger arguments.",
                    unanimous=True,
                )
            )
            mock_jury.__len__ = MagicMock(return_value=3)
            MockJury.return_value = mock_jury

            debate = Debate(
                topic="Test Topic",
                agents=mock_agents,
                rounds=2,
            )

            # Mock agent argument generation
            for agent in mock_agents:
                agent.generate_argument = AsyncMock(
                    return_value=Argument(
                        agent=agent.name,
                        level=ArgumentLevel.STRATEGIC,
                        content=f"{agent.name}'s argument",
                    )
                )

            # Mock evaluator
            with patch.object(debate._evaluator, "evaluate_argument") as mock_eval:
                mock_eval.return_value = ArgumentEvaluation(
                    argument_id="arg_0",
                    scores={"logic": 0.8},
                    weights={"logic": 1.0},
                    causal_score=0.8,
                    total_score=0.8,
                )

                result = await debate.run()

            assert result.verdict.decision == "Pro"
            assert result.final_state == DebateState.COMPLETE
            assert len(result.transcript) > 0
            assert result.metadata.total_rounds == 2

    @pytest.mark.asyncio
    async def test_run_halted_debate(self, mock_agents: list[Agent]) -> None:
        """Test debate that gets halted."""
        config = DebateConfig(halt_on_safety_violation=True)

        with patch("artemis.core.debate.JuryPanel") as MockJury:
            mock_jury = MagicMock()
            mock_jury.__len__ = MagicMock(return_value=3)
            MockJury.return_value = mock_jury

            debate = Debate(
                topic="Test",
                agents=mock_agents,
                rounds=2,
                config=config,
            )

            # Add halting safety monitor
            def halt_monitor(_turn: Turn, _context: DebateContext) -> SafetyResult:
                return SafetyResult(
                    monitor="halt",
                    severity=1.0,
                    should_halt=True,
                )

            debate.add_safety_monitor(halt_monitor)

            # Mock agent argument generation
            for agent in mock_agents:
                agent.generate_argument = AsyncMock(
                    return_value=Argument(
                        agent=agent.name,
                        level=ArgumentLevel.STRATEGIC,
                        content="Test",
                    )
                )

            with patch.object(debate._evaluator, "evaluate_argument") as mock_eval:
                mock_eval.return_value = ArgumentEvaluation(
                    argument_id="arg_0",
                    scores={},
                    weights={},
                    causal_score=0.5,
                    total_score=0.5,
                )

                result = await debate.run()

            assert result.final_state == DebateState.HALTED
            assert result.verdict.decision == "halted"


class TestDebatePhases:
    """Tests for individual debate phases."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.supports_reasoning = False
        return model

    @pytest.fixture
    def mock_agents(self, mock_model: MagicMock) -> list[Agent]:
        """Create mock agents."""
        agent1 = Agent(name="Pro", role="Argues in favor", model=mock_model)
        agent2 = Agent(name="Con", role="Argues against", model=mock_model)
        return [agent1, agent2]

    @pytest.mark.asyncio
    async def test_opening_phase(self, mock_agents: list[Agent]) -> None:
        """Test opening statements phase."""
        with patch("artemis.core.debate.JuryPanel"):
            debate = Debate(topic="Test", agents=mock_agents, rounds=3)

            for agent in mock_agents:
                agent.generate_argument = AsyncMock(
                    return_value=Argument(
                        agent=agent.name,
                        level=ArgumentLevel.STRATEGIC,
                        content="Opening statement",
                    )
                )

            with patch.object(debate._evaluator, "evaluate_argument") as mock_eval:
                mock_eval.return_value = ArgumentEvaluation(
                    argument_id="arg_0",
                    scores={},
                    weights={},
                    causal_score=0.5,
                    total_score=0.5,
                )

                await debate._run_opening()

            assert debate.state == DebateState.DEBATE
            assert len(debate.transcript) == 2  # One per agent

    @pytest.mark.asyncio
    async def test_execute_round(self, mock_agents: list[Agent]) -> None:
        """Test executing a single round."""
        with patch("artemis.core.debate.JuryPanel"):
            debate = Debate(topic="Test", agents=mock_agents, rounds=3)

            for agent in mock_agents:
                agent.generate_argument = AsyncMock(
                    return_value=Argument(
                        agent=agent.name,
                        level=ArgumentLevel.TACTICAL,
                        content="Round argument",
                    )
                )

            with patch.object(debate._evaluator, "evaluate_argument") as mock_eval:
                mock_eval.return_value = ArgumentEvaluation(
                    argument_id="arg_0",
                    scores={},
                    weights={},
                    causal_score=0.5,
                    total_score=0.5,
                )

                turns = await debate._execute_round(1)

            assert len(turns) == 2

    @pytest.mark.asyncio
    async def test_closing_phase(self, mock_agents: list[Agent]) -> None:
        """Test closing arguments phase."""
        with patch("artemis.core.debate.JuryPanel"):
            debate = Debate(topic="Test", agents=mock_agents, rounds=3)

            for agent in mock_agents:
                agent.generate_argument = AsyncMock(
                    return_value=Argument(
                        agent=agent.name,
                        level=ArgumentLevel.STRATEGIC,
                        content="Closing statement",
                    )
                )

            with patch.object(debate._evaluator, "evaluate_argument") as mock_eval:
                mock_eval.return_value = ArgumentEvaluation(
                    argument_id="arg_0",
                    scores={},
                    weights={},
                    causal_score=0.5,
                    total_score=0.5,
                )

                await debate._run_closing()

            assert debate.state == DebateState.CLOSING
            # Closing turns have round=total_rounds+1
            closing_round = debate.total_rounds + 1
            closing_turns = [t for t in debate.transcript if t.round == closing_round]
            assert len(closing_turns) == 2

    @pytest.mark.asyncio
    async def test_deliberation_phase(self, mock_agents: list[Agent]) -> None:
        """Test jury deliberation phase."""
        with patch("artemis.core.debate.JuryPanel") as MockJury:
            mock_jury = MagicMock()
            mock_jury.deliberate = AsyncMock(
                return_value=Verdict(
                    decision="Pro",
                    confidence=0.9,
                    reasoning="Strong arguments",
                    unanimous=True,
                )
            )
            MockJury.return_value = mock_jury

            debate = Debate(topic="Test", agents=mock_agents, rounds=3)

            verdict = await debate._run_deliberation()

            assert verdict.decision == "Pro"
            assert debate.state == DebateState.DELIBERATION


class TestDebateResult:
    """Tests for DebateResult building."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.supports_reasoning = False
        return model

    @pytest.fixture
    def mock_agents(self, mock_model: MagicMock) -> list[Agent]:
        """Create mock agents."""
        agent1 = Agent(name="Pro", role="Argues in favor", model=mock_model)
        agent2 = Agent(name="Con", role="Argues against", model=mock_model)
        return [agent1, agent2]

    def test_build_result(self, mock_agents: list[Agent]) -> None:
        """Test building debate result."""
        with patch("artemis.core.debate.JuryPanel") as MockJury:
            mock_jury = MagicMock()
            mock_jury.__len__ = MagicMock(return_value=3)
            MockJury.return_value = mock_jury

            debate = Debate(topic="Test", agents=mock_agents, rounds=3)
            debate._started_at = datetime.utcnow()
            debate._ended_at = datetime.utcnow()

            verdict = Verdict(
                decision="Pro",
                confidence=0.8,
                reasoning="Good arguments",
                unanimous=False,
            )

            result = debate._build_result(verdict)

            assert result.topic == "Test"
            assert result.verdict.decision == "Pro"
            assert result.metadata.total_rounds == 3
            assert result.metadata.agents == ["Pro", "Con"]
            assert result.metadata.jury_size == 3


class TestDebateRunSingleRound:
    """Tests for step-by-step debate execution."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.supports_reasoning = False
        return model

    @pytest.fixture
    def mock_agents(self, mock_model: MagicMock) -> list[Agent]:
        """Create mock agents."""
        agent1 = Agent(name="Pro", role="Argues in favor", model=mock_model)
        agent2 = Agent(name="Con", role="Argues against", model=mock_model)
        return [agent1, agent2]

    @pytest.mark.asyncio
    async def test_run_single_round_from_setup(
        self, mock_agents: list[Agent]
    ) -> None:
        """Test running single round from setup runs opening."""
        with patch("artemis.core.debate.JuryPanel"):
            debate = Debate(topic="Test", agents=mock_agents, rounds=3)

            for agent in mock_agents:
                agent.generate_argument = AsyncMock(
                    return_value=Argument(
                        agent=agent.name,
                        level=ArgumentLevel.STRATEGIC,
                        content="Opening",
                    )
                )

            with patch.object(debate._evaluator, "evaluate_argument") as mock_eval:
                mock_eval.return_value = ArgumentEvaluation(
                    argument_id="arg_0",
                    scores={},
                    weights={},
                    causal_score=0.5,
                    total_score=0.5,
                )

                turns = await debate.run_single_round()

            assert len(turns) == 2
            assert all(t.round == 0 for t in turns)

    @pytest.mark.asyncio
    async def test_run_single_round_after_complete(
        self, mock_agents: list[Agent]
    ) -> None:
        """Test that running round after complete raises error."""
        with patch("artemis.core.debate.JuryPanel"):
            debate = Debate(topic="Test", agents=mock_agents, rounds=3)
            debate._state = DebateState.COMPLETE

            with pytest.raises(DebateError, match="already complete"):
                await debate.run_single_round()
