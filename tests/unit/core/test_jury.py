"""
Unit tests for Jury Panel evaluation mechanism.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artemis.core.jury import (
    ConsensusResult,
    JurorEvaluation,
    JuryConfig,
    JuryMember,
    JuryPanel,
)
from artemis.core.types import (
    Argument,
    ArgumentEvaluation,
    ArgumentLevel,
    DebateContext,
    JuryPerspective,
    ModelResponse,
    Turn,
    Usage,
    Verdict,
)
from artemis.models import BaseModel


class TestJurorEvaluation:
    """Tests for JurorEvaluation dataclass."""

    def test_evaluation_creation(self) -> None:
        """Test creating a juror evaluation."""
        evaluation = JurorEvaluation(
            juror_id="juror_0",
            perspective=JuryPerspective.ANALYTICAL,
            agent_scores={"Agent1": 0.8, "Agent2": 0.6},
            criterion_scores={
                "Agent1": {"logic": 0.9, "evidence": 0.7},
                "Agent2": {"logic": 0.5, "evidence": 0.7},
            },
            winner="Agent1",
            confidence=0.85,
            reasoning="Agent1 demonstrated stronger logical reasoning.",
        )
        assert evaluation.juror_id == "juror_0"
        assert evaluation.perspective == JuryPerspective.ANALYTICAL
        assert evaluation.winner == "Agent1"
        assert evaluation.confidence == 0.85

    def test_evaluation_with_empty_scores(self) -> None:
        """Test evaluation with empty score dicts."""
        evaluation = JurorEvaluation(
            juror_id="juror_1",
            perspective=JuryPerspective.ETHICAL,
            agent_scores={},
            criterion_scores={},
            winner="",
            confidence=0.0,
            reasoning="No arguments to evaluate.",
        )
        assert evaluation.agent_scores == {}
        assert evaluation.winner == ""


class TestConsensusResult:
    """Tests for ConsensusResult dataclass."""

    def test_consensus_creation(self) -> None:
        """Test creating a consensus result."""
        consensus = ConsensusResult(
            decision="Agent1",
            agreement_score=0.85,
            supporting_jurors=["juror_0", "juror_1", "juror_2"],
            dissenting_jurors=[],
            reasoning="All jurors support Agent1.",
        )
        assert consensus.decision == "Agent1"
        assert consensus.agreement_score == 0.85
        assert len(consensus.supporting_jurors) == 3
        assert len(consensus.dissenting_jurors) == 0

    def test_consensus_with_dissent(self) -> None:
        """Test consensus with dissenting jurors."""
        consensus = ConsensusResult(
            decision="Agent1",
            agreement_score=0.67,
            supporting_jurors=["juror_0", "juror_1"],
            dissenting_jurors=["juror_2"],
            reasoning="Majority supports Agent1.",
        )
        assert len(consensus.dissenting_jurors) == 1

    def test_consensus_draw(self) -> None:
        """Test draw consensus."""
        consensus = ConsensusResult(
            decision="draw",
            agreement_score=0.5,
            supporting_jurors=[],
            dissenting_jurors=[],
            reasoning="Evenly split decision.",
        )
        assert consensus.decision == "draw"


class TestJuryConfig:
    """Tests for JuryConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = JuryConfig()
        assert config.evaluators == 3
        assert config.model == "gpt-4o"
        assert config.consensus_threshold == 0.7
        assert len(config.criteria) > 0
        assert config.require_reasoning is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = JuryConfig(
            evaluators=5,
            model="claude-3-opus",
            consensus_threshold=0.8,
            criteria=["logic", "evidence"],
            require_reasoning=False,
        )
        assert config.evaluators == 5
        assert config.model == "claude-3-opus"
        assert config.consensus_threshold == 0.8
        assert config.criteria == ["logic", "evidence"]
        assert config.require_reasoning is False


class TestJuryMember:
    """Tests for JuryMember class."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.generate = AsyncMock(
            return_value=ModelResponse(
                content="Agent1 presented stronger logical arguments.",
                usage=Usage(prompt_tokens=100, completion_tokens=50),
                model="gpt-4o-mock",
            )
        )
        return model

    @pytest.fixture
    def sample_transcript(self) -> list[Turn]:
        """Create a sample debate transcript."""
        return [
            Turn(
                agent="Agent1",
                argument=Argument(
                    agent="Agent1",
                    level=ArgumentLevel.STRATEGIC,
                    content="AI will fundamentally transform education.",
                ),
                round=1,
                sequence=0,
            ),
            Turn(
                agent="Agent2",
                argument=Argument(
                    agent="Agent2",
                    level=ArgumentLevel.STRATEGIC,
                    content="Human teachers remain irreplaceable.",
                ),
                round=1,
                sequence=1,
            ),
            Turn(
                agent="Agent1",
                argument=Argument(
                    agent="Agent1",
                    level=ArgumentLevel.TACTICAL,
                    content="Studies show 30% improvement with AI tutoring.",
                ),
                round=2,
                sequence=0,
            ),
        ]

    @pytest.fixture
    def sample_context(self) -> DebateContext:
        """Create a sample debate context."""
        return DebateContext(
            topic="AI in Education",
            current_round=3,
            total_rounds=5,
            turn_in_round=0,
        )

    def test_juror_creation_with_model_instance(self, mock_model: MagicMock) -> None:
        """Test creating a juror with model instance."""
        juror = JuryMember(
            juror_id="juror_0",
            perspective=JuryPerspective.ANALYTICAL,
            model=mock_model,
        )
        assert juror.juror_id == "juror_0"
        assert juror.perspective == JuryPerspective.ANALYTICAL
        assert juror._model == mock_model

    @patch("artemis.core.jury.create_model")
    def test_juror_creation_with_model_string(
        self, mock_create_model: MagicMock, mock_model: MagicMock
    ) -> None:
        """Test creating a juror with model string."""
        mock_create_model.return_value = mock_model
        juror = JuryMember(
            juror_id="juror_1",
            perspective=JuryPerspective.ETHICAL,
            model="gpt-4o",
        )
        assert juror.juror_id == "juror_1"
        mock_create_model.assert_called_once()

    def test_perspective_prompts_defined(self) -> None:
        """Test that all perspectives have prompts."""
        for perspective in JuryPerspective:
            assert perspective in JuryMember.PERSPECTIVE_PROMPTS
            assert len(JuryMember.PERSPECTIVE_PROMPTS[perspective]) > 0

    @pytest.mark.asyncio
    async def test_evaluate_transcript(
        self,
        mock_model: MagicMock,
        sample_transcript: list[Turn],
        sample_context: DebateContext,
    ) -> None:
        """Test evaluating a debate transcript."""
        juror = JuryMember(
            juror_id="juror_0",
            perspective=JuryPerspective.ANALYTICAL,
            model=mock_model,
        )

        # Mock the evaluator
        with patch.object(juror._evaluator, "evaluate_argument") as mock_eval:
            mock_eval.return_value = ArgumentEvaluation(
                argument_id="arg_0",
                scores={"logic": 0.8, "evidence": 0.7},
                weights={"logic": 0.5, "evidence": 0.5},
                causal_score=0.7,
                total_score=0.75,
            )

            evaluation = await juror.evaluate(sample_transcript, sample_context)

            assert evaluation.juror_id == "juror_0"
            assert evaluation.perspective == JuryPerspective.ANALYTICAL
            assert "Agent1" in evaluation.agent_scores
            assert "Agent2" in evaluation.agent_scores
            assert evaluation.winner in ["Agent1", "Agent2"]
            assert 0 <= evaluation.confidence <= 1

    @pytest.mark.asyncio
    async def test_evaluate_empty_transcript(
        self,
        mock_model: MagicMock,
        sample_context: DebateContext,
    ) -> None:
        """Test evaluating an empty transcript."""
        juror = JuryMember(
            juror_id="juror_0",
            perspective=JuryPerspective.ANALYTICAL,
            model=mock_model,
        )

        evaluation = await juror.evaluate([], sample_context)

        assert evaluation.agent_scores == {}

    def test_repr(self, mock_model: MagicMock) -> None:
        """Test string representation."""
        juror = JuryMember(
            juror_id="juror_0",
            perspective=JuryPerspective.PRACTICAL,
            model=mock_model,
        )
        repr_str = repr(juror)
        assert "juror_0" in repr_str
        assert "practical" in repr_str


class TestJuryMemberScoring:
    """Tests for JuryMember scoring methods."""

    @pytest.fixture
    def juror(self) -> JuryMember:
        """Create a juror with mock model."""
        mock_model = MagicMock(spec=BaseModel)
        mock_model.model = "gpt-4o-mock"
        return JuryMember(
            juror_id="juror_0",
            perspective=JuryPerspective.ANALYTICAL,
            model=mock_model,
        )

    def test_compute_agent_scores(self, juror: JuryMember) -> None:
        """Test computing agent scores from evaluations."""
        evaluations = {
            "Agent1": [
                ArgumentEvaluation(
                    argument_id="arg_0",
                    scores={"logic": 0.8},
                    weights={"logic": 1.0},
                    causal_score=0.8,
                    total_score=0.8,
                ),
                ArgumentEvaluation(
                    argument_id="arg_1",
                    scores={"logic": 0.9},
                    weights={"logic": 1.0},
                    causal_score=0.9,
                    total_score=0.9,
                ),
            ],
            "Agent2": [
                ArgumentEvaluation(
                    argument_id="arg_2",
                    scores={"logic": 0.6},
                    weights={"logic": 1.0},
                    causal_score=0.6,
                    total_score=0.6,
                ),
            ],
        }

        scores = juror._compute_agent_scores(evaluations)

        assert abs(scores["Agent1"] - 0.85) < 0.001  # (0.8 + 0.9) / 2
        assert abs(scores["Agent2"] - 0.6) < 0.001

    def test_compute_agent_scores_empty(self, juror: JuryMember) -> None:
        """Test computing scores with empty evaluations."""
        evaluations: dict[str, list[ArgumentEvaluation]] = {"Agent1": []}

        scores = juror._compute_agent_scores(evaluations)

        assert scores["Agent1"] == 0.0

    def test_compute_criterion_scores(self, juror: JuryMember) -> None:
        """Test computing criterion-level scores."""
        evaluations = {
            "Agent1": [
                ArgumentEvaluation(
                    argument_id="arg_0",
                    scores={"logic": 0.8, "evidence": 0.7},
                    weights={"logic": 0.5, "evidence": 0.5},
                    causal_score=0.75,
                    total_score=0.75,
                ),
                ArgumentEvaluation(
                    argument_id="arg_1",
                    scores={"logic": 0.9, "evidence": 0.8},
                    weights={"logic": 0.5, "evidence": 0.5},
                    causal_score=0.85,
                    total_score=0.85,
                ),
            ],
        }

        criterion_scores = juror._compute_criterion_scores(evaluations)

        assert "Agent1" in criterion_scores
        assert abs(criterion_scores["Agent1"]["logic"] - 0.85) < 0.001  # (0.8 + 0.9) / 2
        assert abs(criterion_scores["Agent1"]["evidence"] - 0.75) < 0.001  # (0.7 + 0.8) / 2

    def test_apply_perspective_weighting(self, juror: JuryMember) -> None:
        """Test perspective-based score weighting."""
        scores = {"Agent1": 0.8, "Agent2": 0.6}

        weighted = juror._apply_perspective_weighting(scores)

        # Currently returns unweighted, but should still work
        assert weighted == scores


class TestJuryPanel:
    """Tests for JuryPanel class."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model."""
        model = MagicMock(spec=BaseModel)
        model.model = "gpt-4o-mock"
        model.generate = AsyncMock(
            return_value=ModelResponse(
                content="Agent1 won with better arguments.",
                usage=Usage(prompt_tokens=100, completion_tokens=50),
                model="gpt-4o-mock",
            )
        )
        return model

    @pytest.fixture
    def sample_transcript(self) -> list[Turn]:
        """Create a sample debate transcript."""
        return [
            Turn(
                agent="Agent1",
                argument=Argument(
                    agent="Agent1",
                    level=ArgumentLevel.STRATEGIC,
                    content="AI will transform healthcare fundamentally.",
                ),
                round=1,
                sequence=0,
            ),
            Turn(
                agent="Agent2",
                argument=Argument(
                    agent="Agent2",
                    level=ArgumentLevel.STRATEGIC,
                    content="Human doctors are irreplaceable.",
                ),
                round=1,
                sequence=1,
            ),
        ]

    @pytest.fixture
    def sample_context(self) -> DebateContext:
        """Create a sample debate context."""
        return DebateContext(
            topic="AI in Healthcare",
            current_round=3,
            total_rounds=5,
            turn_in_round=0,
        )

    @patch("artemis.core.jury.create_model")
    def test_panel_creation(self, mock_create_model: MagicMock) -> None:
        """Test creating a jury panel."""
        mock_model = MagicMock(spec=BaseModel)
        mock_model.model = "gpt-4o-mock"
        mock_create_model.return_value = mock_model

        panel = JuryPanel(evaluators=3, model="gpt-4o")

        assert len(panel.jurors) == 3
        assert panel.consensus_threshold == 0.7

    @patch("artemis.core.jury.create_model")
    def test_panel_diverse_perspectives(self, mock_create_model: MagicMock) -> None:
        """Test that panel has diverse perspectives."""
        mock_model = MagicMock(spec=BaseModel)
        mock_model.model = "gpt-4o-mock"
        mock_create_model.return_value = mock_model

        panel = JuryPanel(evaluators=5)

        perspectives = panel.get_perspectives()
        assert len(perspectives) == 5
        # Should cycle through perspectives
        assert JuryPerspective.ANALYTICAL in perspectives
        assert JuryPerspective.ETHICAL in perspectives

    @patch("artemis.core.jury.create_model")
    def test_assign_perspective(self, mock_create_model: MagicMock) -> None:
        """Test perspective assignment logic."""
        mock_model = MagicMock(spec=BaseModel)
        mock_model.model = "gpt-4o-mock"
        mock_create_model.return_value = mock_model

        panel = JuryPanel(evaluators=1)

        # Test cycling
        assert panel._assign_perspective(0) == JuryPerspective.ANALYTICAL
        assert panel._assign_perspective(1) == JuryPerspective.ETHICAL
        assert panel._assign_perspective(5) == JuryPerspective.ANALYTICAL  # Cycle

    @patch("artemis.core.jury.create_model")
    def test_get_juror(self, mock_create_model: MagicMock) -> None:
        """Test getting a specific juror."""
        mock_model = MagicMock(spec=BaseModel)
        mock_model.model = "gpt-4o-mock"
        mock_create_model.return_value = mock_model

        panel = JuryPanel(evaluators=3)

        juror = panel.get_juror("juror_0")
        assert juror is not None
        assert juror.juror_id == "juror_0"

        missing = panel.get_juror("juror_99")
        assert missing is None

    @patch("artemis.core.jury.create_model")
    def test_panel_len(self, mock_create_model: MagicMock) -> None:
        """Test panel length."""
        mock_model = MagicMock(spec=BaseModel)
        mock_model.model = "gpt-4o-mock"
        mock_create_model.return_value = mock_model

        panel = JuryPanel(evaluators=5)
        assert len(panel) == 5

    @patch("artemis.core.jury.create_model")
    def test_panel_repr(self, mock_create_model: MagicMock) -> None:
        """Test panel string representation."""
        mock_model = MagicMock(spec=BaseModel)
        mock_model.model = "gpt-4o-mock"
        mock_create_model.return_value = mock_model

        panel = JuryPanel(evaluators=3, consensus_threshold=0.8)
        repr_str = repr(panel)
        assert "3" in repr_str
        assert "0.8" in repr_str


class TestJuryPanelConsensus:
    """Tests for JuryPanel consensus building."""

    def test_build_consensus_unanimous(self) -> None:
        """Test building consensus with unanimous agreement."""
        evaluations = [
            JurorEvaluation(
                juror_id="juror_0",
                perspective=JuryPerspective.ANALYTICAL,
                agent_scores={"Agent1": 0.9, "Agent2": 0.6},
                criterion_scores={},
                winner="Agent1",
                confidence=0.9,
                reasoning="Agent1 was better.",
            ),
            JurorEvaluation(
                juror_id="juror_1",
                perspective=JuryPerspective.ETHICAL,
                agent_scores={"Agent1": 0.85, "Agent2": 0.65},
                criterion_scores={},
                winner="Agent1",
                confidence=0.85,
                reasoning="Agent1 showed better ethics.",
            ),
            JurorEvaluation(
                juror_id="juror_2",
                perspective=JuryPerspective.PRACTICAL,
                agent_scores={"Agent1": 0.8, "Agent2": 0.7},
                criterion_scores={},
                winner="Agent1",
                confidence=0.8,
                reasoning="Agent1 more practical.",
            ),
        ]

        with patch("artemis.core.jury.create_model"):
            panel = JuryPanel.__new__(JuryPanel)
            panel.consensus_threshold = 0.7

        consensus = panel._build_consensus(evaluations)

        assert consensus.decision == "Agent1"
        assert consensus.agreement_score == 1.0
        assert len(consensus.supporting_jurors) == 3
        assert len(consensus.dissenting_jurors) == 0

    def test_build_consensus_majority(self) -> None:
        """Test building consensus with majority agreement."""
        evaluations = [
            JurorEvaluation(
                juror_id="juror_0",
                perspective=JuryPerspective.ANALYTICAL,
                agent_scores={"Agent1": 0.9, "Agent2": 0.6},
                criterion_scores={},
                winner="Agent1",
                confidence=0.9,
                reasoning="Agent1 was better.",
            ),
            JurorEvaluation(
                juror_id="juror_1",
                perspective=JuryPerspective.ETHICAL,
                agent_scores={"Agent1": 0.85, "Agent2": 0.65},
                criterion_scores={},
                winner="Agent1",
                confidence=0.85,
                reasoning="Agent1 showed better ethics.",
            ),
            JurorEvaluation(
                juror_id="juror_2",
                perspective=JuryPerspective.PRACTICAL,
                agent_scores={"Agent1": 0.5, "Agent2": 0.8},
                criterion_scores={},
                winner="Agent2",
                confidence=0.75,
                reasoning="Agent2 more practical.",
            ),
        ]

        with patch("artemis.core.jury.create_model"):
            panel = JuryPanel.__new__(JuryPanel)
            panel.consensus_threshold = 0.7

        consensus = panel._build_consensus(evaluations)

        assert consensus.decision == "Agent1"
        assert len(consensus.supporting_jurors) == 2
        assert len(consensus.dissenting_jurors) == 1

    def test_build_consensus_empty(self) -> None:
        """Test building consensus with no evaluations."""
        with patch("artemis.core.jury.create_model"):
            panel = JuryPanel.__new__(JuryPanel)
            panel.consensus_threshold = 0.7

        consensus = panel._build_consensus([])

        assert consensus.decision == "draw"
        assert consensus.agreement_score == 0.0

    def test_calculate_confidence(self) -> None:
        """Test confidence calculation."""
        evaluations = [
            JurorEvaluation(
                juror_id="juror_0",
                perspective=JuryPerspective.ANALYTICAL,
                agent_scores={"Agent1": 0.9},
                criterion_scores={},
                winner="Agent1",
                confidence=0.9,
                reasoning="",
            ),
            JurorEvaluation(
                juror_id="juror_1",
                perspective=JuryPerspective.ETHICAL,
                agent_scores={"Agent1": 0.8},
                criterion_scores={},
                winner="Agent1",
                confidence=0.8,
                reasoning="",
            ),
        ]

        consensus = ConsensusResult(
            decision="Agent1",
            agreement_score=1.0,
            supporting_jurors=["juror_0", "juror_1"],
            dissenting_jurors=[],
            reasoning="",
        )

        with patch("artemis.core.jury.create_model"):
            panel = JuryPanel.__new__(JuryPanel)

        confidence = panel._calculate_confidence(evaluations, consensus)

        # 60% agreement (1.0) + 40% avg confidence (0.85) = 0.94
        assert 0.9 <= confidence <= 1.0

    def test_calculate_confidence_empty(self) -> None:
        """Test confidence with no evaluations."""
        with patch("artemis.core.jury.create_model"):
            panel = JuryPanel.__new__(JuryPanel)

        consensus = ConsensusResult(
            decision="draw",
            agreement_score=0.0,
            supporting_jurors=[],
            dissenting_jurors=[],
            reasoning="",
        )

        confidence = panel._calculate_confidence([], consensus)
        assert confidence == 0.0


class TestJuryPanelDissent:
    """Tests for collecting dissenting opinions."""

    def test_collect_dissents(self) -> None:
        """Test collecting dissenting opinions."""
        evaluations = [
            JurorEvaluation(
                juror_id="juror_0",
                perspective=JuryPerspective.ANALYTICAL,
                agent_scores={"Agent1": 0.9, "Agent2": 0.6},
                criterion_scores={},
                winner="Agent1",
                confidence=0.9,
                reasoning="Agent1 was better.",
            ),
            JurorEvaluation(
                juror_id="juror_1",
                perspective=JuryPerspective.ETHICAL,
                agent_scores={"Agent1": 0.5, "Agent2": 0.8},
                criterion_scores={},
                winner="Agent2",
                confidence=0.8,
                reasoning="Agent2 showed better ethics.",
            ),
        ]

        consensus = ConsensusResult(
            decision="Agent1",
            agreement_score=0.53,
            supporting_jurors=["juror_0"],
            dissenting_jurors=["juror_1"],
            reasoning="",
        )

        with patch("artemis.core.jury.create_model"):
            panel = JuryPanel.__new__(JuryPanel)

        dissents = panel._collect_dissents(evaluations, consensus)

        assert len(dissents) == 1
        assert dissents[0].juror_id == "juror_1"
        assert dissents[0].perspective == JuryPerspective.ETHICAL
        assert dissents[0].position == "Agent2"

    def test_collect_dissents_none(self) -> None:
        """Test when there are no dissents."""
        evaluations = [
            JurorEvaluation(
                juror_id="juror_0",
                perspective=JuryPerspective.ANALYTICAL,
                agent_scores={"Agent1": 0.9},
                criterion_scores={},
                winner="Agent1",
                confidence=0.9,
                reasoning="",
            ),
        ]

        consensus = ConsensusResult(
            decision="Agent1",
            agreement_score=1.0,
            supporting_jurors=["juror_0"],
            dissenting_jurors=[],
            reasoning="",
        )

        with patch("artemis.core.jury.create_model"):
            panel = JuryPanel.__new__(JuryPanel)

        dissents = panel._collect_dissents(evaluations, consensus)
        assert len(dissents) == 0


class TestJuryPanelScoreAggregation:
    """Tests for score aggregation."""

    def test_aggregate_scores(self) -> None:
        """Test aggregating scores across jurors."""
        evaluations = [
            JurorEvaluation(
                juror_id="juror_0",
                perspective=JuryPerspective.ANALYTICAL,
                agent_scores={"Agent1": 0.9, "Agent2": 0.6},
                criterion_scores={},
                winner="Agent1",
                confidence=0.9,
                reasoning="",
            ),
            JurorEvaluation(
                juror_id="juror_1",
                perspective=JuryPerspective.ETHICAL,
                agent_scores={"Agent1": 0.8, "Agent2": 0.7},
                criterion_scores={},
                winner="Agent1",
                confidence=0.8,
                reasoning="",
            ),
        ]

        with patch("artemis.core.jury.create_model"):
            panel = JuryPanel.__new__(JuryPanel)

        aggregated = panel._aggregate_scores(evaluations)

        assert abs(aggregated["Agent1"] - 0.85) < 0.001  # (0.9 + 0.8) / 2
        assert abs(aggregated["Agent2"] - 0.65) < 0.001  # (0.6 + 0.7) / 2

    def test_aggregate_scores_empty(self) -> None:
        """Test aggregating with no evaluations."""
        with patch("artemis.core.jury.create_model"):
            panel = JuryPanel.__new__(JuryPanel)

        aggregated = panel._aggregate_scores([])
        assert aggregated == {}


class TestJuryPanelVerdictReasoning:
    """Tests for verdict reasoning generation."""

    def test_generate_verdict_reasoning_winner(self) -> None:
        """Test generating reasoning for a winner verdict."""
        evaluations = [
            JurorEvaluation(
                juror_id="juror_0",
                perspective=JuryPerspective.ANALYTICAL,
                agent_scores={"Agent1": 0.9},
                criterion_scores={},
                winner="Agent1",
                confidence=0.9,
                reasoning="",
            ),
        ]

        consensus = ConsensusResult(
            decision="Agent1",
            agreement_score=1.0,
            supporting_jurors=["juror_0"],
            dissenting_jurors=[],
            reasoning="",
        )

        scores = {"Agent1": 0.9, "Agent2": 0.6}

        with patch("artemis.core.jury.create_model"):
            panel = JuryPanel.__new__(JuryPanel)

        reasoning = panel._generate_verdict_reasoning(evaluations, consensus, scores)

        assert "Agent1" in reasoning
        assert "favor" in reasoning.lower() or "decided" in reasoning.lower()

    def test_generate_verdict_reasoning_draw(self) -> None:
        """Test generating reasoning for a draw verdict."""
        consensus = ConsensusResult(
            decision="draw",
            agreement_score=0.5,
            supporting_jurors=[],
            dissenting_jurors=[],
            reasoning="",
        )

        with patch("artemis.core.jury.create_model"):
            panel = JuryPanel.__new__(JuryPanel)

        reasoning = panel._generate_verdict_reasoning([], consensus, {})

        assert "draw" in reasoning.lower()


class TestJuryPanelDeliberation:
    """Tests for full deliberation process."""

    @pytest.fixture
    def sample_transcript(self) -> list[Turn]:
        """Create a sample transcript."""
        return [
            Turn(
                agent="Agent1",
                argument=Argument(
                    agent="Agent1",
                    level=ArgumentLevel.STRATEGIC,
                    content="AI will revolutionize medicine.",
                ),
                round=1,
                sequence=0,
            ),
            Turn(
                agent="Agent2",
                argument=Argument(
                    agent="Agent2",
                    level=ArgumentLevel.STRATEGIC,
                    content="Traditional medicine is proven.",
                ),
                round=1,
                sequence=1,
            ),
        ]

    @pytest.fixture
    def sample_context(self) -> DebateContext:
        """Create sample context."""
        return DebateContext(
            topic="AI in Medicine",
            current_round=2,
            total_rounds=5,
            turn_in_round=0,
        )

    @pytest.mark.asyncio
    async def test_deliberate_returns_verdict(
        self,
        sample_transcript: list[Turn],
        sample_context: DebateContext,
    ) -> None:
        """Test that deliberation returns a valid verdict."""
        mock_model = MagicMock(spec=BaseModel)
        mock_model.model = "gpt-4o-mock"
        mock_model.generate = AsyncMock(
            return_value=ModelResponse(
                content="Agent1 presented compelling arguments.",
                usage=Usage(prompt_tokens=100, completion_tokens=50),
                model="gpt-4o-mock",
            )
        )

        with patch("artemis.core.jury.create_model") as mock_create:
            mock_create.return_value = mock_model
            panel = JuryPanel(evaluators=3)

            # Mock each juror's evaluator
            for juror in panel.jurors:
                juror._evaluator.evaluate_argument = AsyncMock(
                    return_value=ArgumentEvaluation(
                        argument_id="arg_0",
                        scores={"logic": 0.8},
                        weights={"logic": 1.0},
                        causal_score=0.8,
                        total_score=0.8,
                    )
                )

            verdict = await panel.deliberate(sample_transcript, sample_context)

            assert isinstance(verdict, Verdict)
            assert verdict.decision in ["Agent1", "Agent2", "draw"]
            assert 0 <= verdict.confidence <= 1
            assert verdict.reasoning is not None
            assert isinstance(verdict.dissenting_opinions, list)
            assert isinstance(verdict.score_breakdown, dict)

    @pytest.mark.asyncio
    async def test_deliberate_with_unanimous_verdict(
        self,
        sample_transcript: list[Turn],
        sample_context: DebateContext,
    ) -> None:
        """Test unanimous verdict detection."""
        mock_model = MagicMock(spec=BaseModel)
        mock_model.model = "gpt-4o-mock"
        mock_model.generate = AsyncMock(
            return_value=ModelResponse(
                content="Agent1 clearly won.",
                usage=Usage(prompt_tokens=100, completion_tokens=50),
                model="gpt-4o-mock",
            )
        )

        with patch("artemis.core.jury.create_model") as mock_create:
            mock_create.return_value = mock_model
            panel = JuryPanel(evaluators=3)

            # Make all jurors give Agent1 higher scores
            for juror in panel.jurors:
                juror._evaluator.evaluate_argument = AsyncMock(
                    side_effect=lambda arg, _ctx: ArgumentEvaluation(
                        argument_id="arg_0",
                        scores={"logic": 0.9 if arg.agent == "Agent1" else 0.5},
                        weights={"logic": 1.0},
                        causal_score=0.9 if arg.agent == "Agent1" else 0.5,
                        total_score=0.9 if arg.agent == "Agent1" else 0.5,
                    )
                )

            verdict = await panel.deliberate(sample_transcript, sample_context)

            assert verdict.unanimous is True
            assert len(verdict.dissenting_opinions) == 0
