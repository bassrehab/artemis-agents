"""
Unit tests for L-AE-CR Adaptive Evaluation module.
"""

import pytest

from artemis.core.causal import CausalGraph
from artemis.core.evaluation import (
    AdaptationConfig,
    AdaptiveEvaluator,
    CriterionEvaluator,
    EvaluationDimension,
    RoundEvaluator,
    TopicAnalysis,
)
from artemis.core.types import (
    Argument,
    ArgumentLevel,
    CausalLink,
    DebateContext,
    EvaluationCriteria,
    Evidence,
)


class TestEvaluationDimension:
    """Tests for EvaluationDimension enum."""

    def test_all_dimensions_defined(self) -> None:
        """Test that all expected dimensions are defined."""
        expected = [
            "logical_coherence",
            "evidence_quality",
            "causal_reasoning",
            "ethical_alignment",
            "persuasiveness",
        ]
        for dim in expected:
            assert hasattr(EvaluationDimension, dim.upper())

    def test_dimension_values(self) -> None:
        """Test dimension string values."""
        assert EvaluationDimension.LOGICAL_COHERENCE.value == "logical_coherence"
        assert EvaluationDimension.EVIDENCE_QUALITY.value == "evidence_quality"


class TestAdaptationConfig:
    """Tests for AdaptationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AdaptationConfig()
        assert config.adaptation_rate == 0.1
        assert config.sensitivity_threshold == 0.7
        assert config.complexity_threshold == 0.7
        assert config.ethical_boost == 1.5

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = AdaptationConfig(
            adaptation_rate=0.2,
            ethical_boost=2.0,
        )
        assert config.adaptation_rate == 0.2
        assert config.ethical_boost == 2.0


class TestTopicAnalysis:
    """Tests for TopicAnalysis class."""

    def test_basic_analysis(self) -> None:
        """Test basic topic analysis."""
        analysis = TopicAnalysis.analyze("Should we regulate AI?")
        assert 0.0 <= analysis.sensitivity <= 1.0
        assert 0.0 <= analysis.complexity <= 1.0
        assert 0.0 <= analysis.controversy <= 1.0

    def test_sensitive_topic(self) -> None:
        """Test analysis of ethically sensitive topic."""
        analysis = TopicAnalysis.analyze(
            "Should AI have rights to make life and death decisions?"
        )
        assert analysis.sensitivity > 0.0

    def test_complex_topic(self) -> None:
        """Test analysis of complex/technical topic."""
        analysis = TopicAnalysis.analyze(
            "Should scientific algorithms be used for economic policy regulation?"
        )
        assert analysis.complexity > 0.0

    def test_controversial_topic(self) -> None:
        """Test analysis of controversial topic."""
        analysis = TopicAnalysis.analyze(
            "This controversial debate is highly polarizing and divisive."
        )
        assert analysis.controversy > 0.0

    def test_neutral_topic(self) -> None:
        """Test analysis of neutral topic."""
        analysis = TopicAnalysis.analyze("What color should we paint the wall?")
        # Should have lower scores
        assert analysis.sensitivity < 0.5
        assert analysis.complexity < 0.5


class TestCriterionEvaluator:
    """Tests for CriterionEvaluator class."""

    @pytest.fixture
    def evaluator(self) -> CriterionEvaluator:
        return CriterionEvaluator()

    @pytest.fixture
    def basic_argument(self) -> Argument:
        return Argument(
            agent="Test",
            level=ArgumentLevel.TACTICAL,
            content="This is a basic argument.",
        )

    @pytest.fixture
    def well_structured_argument(self) -> Argument:
        return Argument(
            agent="Test",
            level=ArgumentLevel.TACTICAL,
            content=(
                "I argue that AI should be regulated. Therefore, we must act now. "
                "First, AI poses risks. Second, regulation is needed. "
                "Furthermore, this is supported by evidence. "
                "Consequently, we should implement safeguards."
            ),
        )

    def test_evaluate_logical_coherence_basic(
        self, evaluator: CriterionEvaluator, basic_argument: Argument
    ) -> None:
        """Test basic logical coherence evaluation."""
        score = evaluator.evaluate_logical_coherence(basic_argument)
        assert 0.0 <= score <= 1.0

    def test_evaluate_logical_coherence_well_structured(
        self, evaluator: CriterionEvaluator, well_structured_argument: Argument
    ) -> None:
        """Test logical coherence for well-structured argument."""
        score = evaluator.evaluate_logical_coherence(well_structured_argument)
        assert score > 0.5  # Should score higher

    def test_evaluate_evidence_quality_no_evidence(
        self, evaluator: CriterionEvaluator, basic_argument: Argument
    ) -> None:
        """Test evidence quality with no evidence."""
        score = evaluator.evaluate_evidence_quality(basic_argument)
        assert 0.0 <= score <= 1.0
        assert score < 0.6  # Lower score without evidence

    def test_evaluate_evidence_quality_with_evidence(
        self, evaluator: CriterionEvaluator
    ) -> None:
        """Test evidence quality with evidence."""
        argument = Argument(
            agent="Test",
            level=ArgumentLevel.TACTICAL,
            content="According to Smith (2024), this is supported by research.",
            evidence=[
                Evidence(
                    type="study",
                    content="Smith study results",
                    source="Smith (2024)",
                    verified=True,
                ),
            ],
        )
        score = evaluator.evaluate_evidence_quality(argument)
        assert score > 0.4

    def test_evaluate_causal_reasoning_no_links(
        self, evaluator: CriterionEvaluator, basic_argument: Argument
    ) -> None:
        """Test causal reasoning with no causal links."""
        score = evaluator.evaluate_causal_reasoning(basic_argument)
        assert 0.0 <= score <= 1.0

    def test_evaluate_causal_reasoning_with_links(
        self, evaluator: CriterionEvaluator
    ) -> None:
        """Test causal reasoning with causal links."""
        argument = Argument(
            agent="Test",
            level=ArgumentLevel.TACTICAL,
            content="AI causes job displacement because automation replaces workers.",
            causal_links=[
                CausalLink(cause="AI", effect="Job Displacement", strength=0.8),
            ],
        )
        score = evaluator.evaluate_causal_reasoning(argument)
        assert score > 0.4

    def test_evaluate_causal_reasoning_with_graph(
        self, evaluator: CriterionEvaluator
    ) -> None:
        """Test causal reasoning with existing causal graph."""
        argument = Argument(
            agent="Test",
            level=ArgumentLevel.TACTICAL,
            content="Technology leads to efficiency.",
            causal_links=[
                CausalLink(cause="Technology", effect="Efficiency", strength=0.7),
            ],
        )

        # Create graph with related nodes
        graph = CausalGraph()
        graph.add_link(
            CausalLink(cause="Technology", effect="Automation", strength=0.8)
        )

        score = evaluator.evaluate_causal_reasoning(argument, graph)
        assert score > 0.4

    def test_evaluate_ethical_alignment_neutral(
        self, evaluator: CriterionEvaluator, basic_argument: Argument
    ) -> None:
        """Test ethical alignment for neutral argument."""
        score = evaluator.evaluate_ethical_alignment(basic_argument)
        assert 0.4 <= score <= 0.6  # Should be near neutral

    def test_evaluate_ethical_alignment_positive(
        self, evaluator: CriterionEvaluator
    ) -> None:
        """Test ethical alignment for ethically aware argument."""
        argument = Argument(
            agent="Test",
            level=ArgumentLevel.TACTICAL,
            content=(
                "We must consider the ethical implications and protect stakeholder "
                "welfare. This approach is fair, just, and responsible."
            ),
        )
        score = evaluator.evaluate_ethical_alignment(argument)
        assert score > 0.5

    def test_evaluate_persuasiveness_basic(
        self, evaluator: CriterionEvaluator, basic_argument: Argument
    ) -> None:
        """Test persuasiveness for basic argument."""
        score = evaluator.evaluate_persuasiveness(basic_argument)
        assert 0.0 <= score <= 1.0

    def test_evaluate_persuasiveness_rhetorical(
        self, evaluator: CriterionEvaluator
    ) -> None:
        """Test persuasiveness for rhetorical argument."""
        argument = Argument(
            agent="Test",
            level=ArgumentLevel.STRATEGIC,
            content=(
                "Consider this: isn't it essential that we act now? "
                "We must take crucial steps. Imagine a future where this is solved. "
                "Our collective action is clearly the most important factor."
            ),
        )
        score = evaluator.evaluate_persuasiveness(argument)
        assert score > 0.5


class TestAdaptiveEvaluator:
    """Tests for AdaptiveEvaluator class."""

    @pytest.fixture
    def evaluator(self) -> AdaptiveEvaluator:
        return AdaptiveEvaluator()

    @pytest.fixture
    def context(self) -> DebateContext:
        return DebateContext(
            topic="Should AI be regulated?",
            current_round=1,
            total_rounds=3,
            turn_in_round=0,
        )

    @pytest.fixture
    def argument(self) -> Argument:
        return Argument(
            agent="Proponent",
            level=ArgumentLevel.TACTICAL,
            content=(
                "AI regulation is necessary because it causes potential harm. "
                "Therefore, we must implement ethical guidelines. "
                "According to experts, this is the responsible approach."
            ),
            causal_links=[
                CausalLink(cause="AI", effect="Potential Harm", strength=0.7),
            ],
        )

    def test_evaluator_initialization(self, evaluator: AdaptiveEvaluator) -> None:
        """Test evaluator initialization."""
        assert evaluator.criteria is not None
        assert evaluator.config is not None
        assert len(evaluator.causal_graph) == 0

    def test_analyze_topic(self, evaluator: AdaptiveEvaluator) -> None:
        """Test topic analysis."""
        analysis = evaluator.analyze_topic("AI ethics and moral implications")
        assert analysis.sensitivity > 0.0
        assert isinstance(analysis, TopicAnalysis)

    def test_adapt_criteria_basic(
        self, evaluator: AdaptiveEvaluator, context: DebateContext
    ) -> None:
        """Test basic criteria adaptation."""
        adapted = evaluator.adapt_criteria(context)

        # Should sum to approximately 1.0
        assert abs(sum(adapted.values()) - 1.0) < 0.01

        # Should have all criteria
        assert "logical_coherence" in adapted
        assert "evidence_quality" in adapted
        assert "causal_reasoning" in adapted
        assert "ethical_alignment" in adapted
        assert "persuasiveness" in adapted

    def test_adapt_criteria_later_rounds(
        self, evaluator: AdaptiveEvaluator
    ) -> None:
        """Test that criteria adapt in later rounds."""
        early_context = DebateContext(
            topic="Test topic",
            current_round=0,
            total_rounds=5,
            turn_in_round=0,
        )
        late_context = DebateContext(
            topic="Test topic",
            current_round=4,
            total_rounds=5,
            turn_in_round=0,
        )

        early_adapted = evaluator.adapt_criteria(early_context)
        late_adapted = evaluator.adapt_criteria(late_context)

        # Evidence weight should increase in later rounds
        assert late_adapted["evidence_quality"] >= early_adapted["evidence_quality"]

    def test_adapt_criteria_sensitive_topic(
        self, evaluator: AdaptiveEvaluator
    ) -> None:
        """Test criteria adaptation for sensitive topics."""
        sensitive_context = DebateContext(
            topic="Life and death ethical decisions in healthcare",
            current_round=1,
            total_rounds=3,
            turn_in_round=0,
        )

        # First analyze the sensitive topic
        evaluator.analyze_topic(sensitive_context.topic)
        adapted = evaluator.adapt_criteria(sensitive_context)

        # Ethical alignment should be boosted
        base_ethical = 0.15  # Default weight
        assert adapted["ethical_alignment"] > base_ethical * 0.8  # Should be higher

    @pytest.mark.asyncio
    async def test_evaluate_argument(
        self,
        evaluator: AdaptiveEvaluator,
        context: DebateContext,
        argument: Argument,
    ) -> None:
        """Test full argument evaluation."""
        evaluation = await evaluator.evaluate_argument(argument, context)

        assert evaluation.argument_id == argument.id
        assert 0.0 <= evaluation.total_score <= 1.0
        assert len(evaluation.scores) == 5
        assert len(evaluation.criterion_details) == 5

        # Check all criteria are scored
        for criterion in [
            "logical_coherence",
            "evidence_quality",
            "causal_reasoning",
            "ethical_alignment",
            "persuasiveness",
        ]:
            assert criterion in evaluation.scores
            assert 0.0 <= evaluation.scores[criterion] <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_updates_causal_graph(
        self,
        evaluator: AdaptiveEvaluator,
        context: DebateContext,
        argument: Argument,
    ) -> None:
        """Test that evaluation updates the causal graph."""
        assert len(evaluator.causal_graph) == 0

        await evaluator.evaluate_argument(argument, context)

        # Should have added nodes from the causal link
        assert len(evaluator.causal_graph) > 0

    @pytest.mark.asyncio
    async def test_evaluate_causal_graph_update(
        self,
        evaluator: AdaptiveEvaluator,
        context: DebateContext,
        argument: Argument,
    ) -> None:
        """Test that evaluation returns causal graph update."""
        evaluation = await evaluator.evaluate_argument(argument, context)

        assert evaluation.causal_graph_update is not None
        assert len(evaluation.causal_graph_update.added_links) > 0

    def test_get_criteria_weights(self, evaluator: AdaptiveEvaluator) -> None:
        """Test getting criteria weights."""
        weights = evaluator.get_criteria_weights()
        assert isinstance(weights, dict)
        assert len(weights) == 5

    def test_set_criteria_weights(self, evaluator: AdaptiveEvaluator) -> None:
        """Test setting custom criteria weights."""
        custom_weights = {
            "logical_coherence": 0.3,
            "evidence_quality": 0.3,
            "causal_reasoning": 0.2,
            "ethical_alignment": 0.1,
            "persuasiveness": 0.1,
        }
        evaluator.set_criteria_weights(custom_weights)

        weights = evaluator.get_criteria_weights()
        assert weights["logical_coherence"] == 0.3

    def test_reset_causal_graph(self, evaluator: AdaptiveEvaluator) -> None:
        """Test resetting the causal graph."""
        # Add something to the graph
        evaluator.causal_graph.add_link(
            CausalLink(cause="A", effect="B", strength=0.5)
        )
        assert len(evaluator.causal_graph) > 0

        evaluator.reset_causal_graph()
        assert len(evaluator.causal_graph) == 0

    def test_custom_criteria_initialization(self) -> None:
        """Test initialization with custom criteria."""
        custom_criteria = EvaluationCriteria(
            logical_coherence=0.4,
            evidence_quality=0.2,
            causal_reasoning=0.2,
            ethical_alignment=0.1,
            persuasiveness=0.1,
        )
        evaluator = AdaptiveEvaluator(criteria=custom_criteria)

        weights = evaluator.get_criteria_weights()
        assert weights["logical_coherence"] == 0.4


class TestRoundEvaluator:
    """Tests for RoundEvaluator class."""

    @pytest.fixture
    def round_evaluator(self) -> RoundEvaluator:
        return RoundEvaluator()

    @pytest.fixture
    def context(self) -> DebateContext:
        return DebateContext(
            topic="Test debate topic",
            current_round=1,
            total_rounds=3,
            turn_in_round=0,
        )

    @pytest.fixture
    def arguments(self) -> list[Argument]:
        return [
            Argument(
                agent="Agent1",
                level=ArgumentLevel.TACTICAL,
                content="First agent's argument with some reasoning.",
            ),
            Argument(
                agent="Agent2",
                level=ArgumentLevel.TACTICAL,
                content="Second agent's counter-argument. Therefore, I disagree.",
            ),
        ]

    @pytest.mark.asyncio
    async def test_evaluate_round(
        self,
        round_evaluator: RoundEvaluator,
        context: DebateContext,
        arguments: list[Argument],
    ) -> None:
        """Test evaluating a full round."""
        evaluations = await round_evaluator.evaluate_round(arguments, context)

        assert len(evaluations) == 2
        for evaluation in evaluations:
            assert 0.0 <= evaluation.total_score <= 1.0

    @pytest.mark.asyncio
    async def test_get_round_summary(
        self,
        round_evaluator: RoundEvaluator,
        context: DebateContext,
        arguments: list[Argument],
    ) -> None:
        """Test getting round summary."""
        evaluations = await round_evaluator.evaluate_round(arguments, context)
        summary = round_evaluator.get_round_summary(evaluations)

        assert "average_score" in summary
        assert "max_score" in summary
        assert "min_score" in summary
        assert summary["max_score"] >= summary["min_score"]

    def test_get_round_summary_empty(
        self, round_evaluator: RoundEvaluator
    ) -> None:
        """Test round summary with no evaluations."""
        summary = round_evaluator.get_round_summary([])
        assert summary["average_score"] == 0.0

    @pytest.mark.asyncio
    async def test_compare_agents(
        self,
        round_evaluator: RoundEvaluator,
        context: DebateContext,
        arguments: list[Argument],
    ) -> None:
        """Test comparing agents' scores."""
        evaluations = await round_evaluator.evaluate_round(arguments, context)
        comparison = round_evaluator.compare_agents(evaluations, arguments)

        assert "Agent1" in comparison
        assert "Agent2" in comparison
        assert "average" in comparison["Agent1"]
        assert "total_arguments" in comparison["Agent1"]
        assert comparison["Agent1"]["total_arguments"] == 1


class TestIntegration:
    """Integration tests for evaluation system."""

    @pytest.mark.asyncio
    async def test_full_evaluation_flow(self) -> None:
        """Test full evaluation flow with multiple arguments."""
        evaluator = AdaptiveEvaluator()

        context = DebateContext(
            topic="Should artificial intelligence be regulated?",
            current_round=1,
            total_rounds=3,
            turn_in_round=0,
        )

        # Analyze topic first
        evaluator.analyze_topic(context.topic)

        # Create arguments
        arg1 = Argument(
            agent="Proponent",
            level=ArgumentLevel.STRATEGIC,
            content=(
                "I argue that AI must be regulated. First, AI poses significant risks. "
                "Second, we need ethical guidelines. Third, society must be protected. "
                "Therefore, comprehensive regulation is essential."
            ),
        )

        arg2 = Argument(
            agent="Opponent",
            level=ArgumentLevel.STRATEGIC,
            content=(
                "However, over-regulation stifles innovation. "
                "Studies show that regulatory burden reduces technology advancement. "
                "Consequently, we should be careful not to over-regulate."
            ),
            causal_links=[
                CausalLink(
                    cause="Over-regulation",
                    effect="Reduced Innovation",
                    strength=0.7,
                ),
            ],
        )

        # Evaluate both
        eval1 = await evaluator.evaluate_argument(arg1, context)
        eval2 = await evaluator.evaluate_argument(arg2, context)

        # Both should have valid scores
        assert 0.0 <= eval1.total_score <= 1.0
        assert 0.0 <= eval2.total_score <= 1.0

        # Causal graph should have been updated
        assert len(evaluator.causal_graph) > 0

    @pytest.mark.asyncio
    async def test_criteria_adaptation_across_rounds(self) -> None:
        """Test that criteria adapt across debate rounds."""
        evaluator = AdaptiveEvaluator()

        argument = Argument(
            agent="Test",
            level=ArgumentLevel.TACTICAL,
            content="This is a test argument with some content.",
        )

        # Evaluate in early round
        early_context = DebateContext(
            topic="Test topic",
            current_round=0,
            total_rounds=5,
            turn_in_round=0,
        )
        early_eval = await evaluator.evaluate_argument(argument, early_context)

        # Reset graph for fair comparison
        evaluator.reset_causal_graph()

        # Evaluate in late round
        late_context = DebateContext(
            topic="Test topic",
            current_round=4,
            total_rounds=5,
            turn_in_round=0,
        )
        late_eval = await evaluator.evaluate_argument(argument, late_context)

        # Weights should be different
        assert early_eval.weights != late_eval.weights
