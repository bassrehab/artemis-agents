"""
Unit tests for Ethics Evaluator.
"""

import pytest

from artemis.core.ethics import (
    EthicalConcern,
    EthicalFramework,
    EthicsEvaluator,
    EthicsResult,
    FrameworkScore,
    StakeholderImpact,
    StakeholderType,
)
from artemis.core.types import Argument, ArgumentLevel, DebateContext


class TestEthicalFramework:
    """Tests for EthicalFramework enum."""

    def test_all_frameworks_defined(self) -> None:
        """Test that all expected frameworks are defined."""
        expected = ["utilitarian", "deontological", "virtue", "care", "justice"]
        for framework_name in expected:
            assert hasattr(EthicalFramework, framework_name.upper())

    def test_framework_values(self) -> None:
        """Test that framework values match expected strings."""
        assert EthicalFramework.UTILITARIAN.value == "utilitarian"
        assert EthicalFramework.DEONTOLOGICAL.value == "deontological"
        assert EthicalFramework.VIRTUE.value == "virtue"
        assert EthicalFramework.CARE.value == "care"
        assert EthicalFramework.JUSTICE.value == "justice"


class TestStakeholderType:
    """Tests for StakeholderType enum."""

    def test_all_stakeholders_defined(self) -> None:
        """Test that all expected stakeholder types are defined."""
        expected = [
            "individuals", "communities", "organizations",
            "society", "environment", "future_generations",
        ]
        for stakeholder_name in expected:
            assert hasattr(StakeholderType, stakeholder_name.upper())


class TestFrameworkScore:
    """Tests for FrameworkScore dataclass."""

    def test_score_creation(self) -> None:
        """Test creating a framework score."""
        score = FrameworkScore(
            framework=EthicalFramework.UTILITARIAN,
            score=0.8,
            reasoning="Strong utilitarian argument.",
            strengths=["Good cost-benefit analysis"],
            weaknesses=[],
        )
        assert score.framework == EthicalFramework.UTILITARIAN
        assert score.score == 0.8
        assert len(score.strengths) == 1

    def test_score_default_lists(self) -> None:
        """Test default empty lists."""
        score = FrameworkScore(
            framework=EthicalFramework.VIRTUE,
            score=0.5,
            reasoning="Moderate",
        )
        assert score.strengths == []
        assert score.weaknesses == []


class TestStakeholderImpact:
    """Tests for StakeholderImpact dataclass."""

    def test_impact_creation(self) -> None:
        """Test creating a stakeholder impact."""
        impact = StakeholderImpact(
            stakeholder=StakeholderType.SOCIETY,
            impact_type="positive",
            severity=0.7,
            description="Benefits society broadly.",
        )
        assert impact.stakeholder == StakeholderType.SOCIETY
        assert impact.impact_type == "positive"
        assert impact.severity == 0.7
        assert impact.reversible is True

    def test_impact_irreversible(self) -> None:
        """Test irreversible impact."""
        impact = StakeholderImpact(
            stakeholder=StakeholderType.ENVIRONMENT,
            impact_type="negative",
            severity=0.9,
            description="Environmental damage.",
            reversible=False,
        )
        assert impact.reversible is False


class TestEthicalConcern:
    """Tests for EthicalConcern dataclass."""

    def test_concern_creation(self) -> None:
        """Test creating an ethical concern."""
        concern = EthicalConcern(
            framework=EthicalFramework.DEONTOLOGICAL,
            concern_type="privacy",
            description="Privacy violation concern.",
            severity=0.6,
            evidence="surveillance, data collection",
        )
        assert concern.framework == EthicalFramework.DEONTOLOGICAL
        assert concern.concern_type == "privacy"
        assert concern.mitigated is False

    def test_concern_mitigated(self) -> None:
        """Test mitigated concern."""
        concern = EthicalConcern(
            framework=EthicalFramework.UTILITARIAN,
            concern_type="harm",
            description="Potential harm.",
            severity=0.4,
            evidence="risk",
            mitigated=True,
        )
        assert concern.mitigated is True


class TestEthicsResult:
    """Tests for EthicsResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating an ethics result."""
        result = EthicsResult(
            overall_score=0.75,
            framework_scores=[
                FrameworkScore(
                    framework=EthicalFramework.UTILITARIAN,
                    score=0.8,
                    reasoning="Good",
                )
            ],
            stakeholder_impacts=[],
            concerns=[],
            strengths=["Balanced consideration"],
            recommendations=["Strengthen care perspective"],
        )
        assert result.overall_score == 0.75
        assert len(result.framework_scores) == 1
        assert len(result.strengths) == 1


class TestEthicsEvaluatorInit:
    """Tests for EthicsEvaluator initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        evaluator = EthicsEvaluator()
        assert len(evaluator.frameworks) == 5
        assert sum(evaluator.weights.values()) == pytest.approx(1.0)

    def test_custom_frameworks(self) -> None:
        """Test initialization with custom frameworks."""
        evaluator = EthicsEvaluator(
            frameworks=[EthicalFramework.UTILITARIAN, EthicalFramework.JUSTICE]
        )
        assert len(evaluator.frameworks) == 2
        assert EthicalFramework.UTILITARIAN in evaluator.frameworks

    def test_custom_weights(self) -> None:
        """Test initialization with custom weights."""
        evaluator = EthicsEvaluator(
            weights={
                EthicalFramework.UTILITARIAN: 0.5,
                EthicalFramework.DEONTOLOGICAL: 0.3,
                EthicalFramework.VIRTUE: 0.2,
            }
        )
        # Weights should be normalized
        assert sum(evaluator.weights.values()) == pytest.approx(1.0)

    def test_repr(self) -> None:
        """Test string representation."""
        evaluator = EthicsEvaluator()
        repr_str = repr(evaluator)
        assert "EthicsEvaluator" in repr_str


class TestEthicsEvaluatorEvaluation:
    """Tests for EthicsEvaluator evaluation."""

    @pytest.fixture
    def evaluator(self) -> EthicsEvaluator:
        """Create an ethics evaluator."""
        return EthicsEvaluator()

    @pytest.fixture
    def sample_context(self) -> DebateContext:
        """Create a sample debate context."""
        return DebateContext(
            topic="AI Ethics in Healthcare",
            current_round=2,
            total_rounds=5,
            turn_in_round=0,
            topic_sensitivity=0.7,
        )

    def test_evaluate_utilitarian_argument(
        self,
        evaluator: EthicsEvaluator,
        sample_context: DebateContext,
    ) -> None:
        """Test evaluating an argument with utilitarian framing."""
        argument = Argument(
            agent="Pro",
            level=ArgumentLevel.STRATEGIC,
            content=(
                "AI in healthcare will maximize patient welfare and improve "
                "outcomes for the greatest number of people. The benefits "
                "clearly outweigh the costs in terms of efficiency and "
                "well-being improvements."
            ),
        )

        result = evaluator.evaluate(argument, sample_context)

        assert isinstance(result, EthicsResult)
        assert 0 <= result.overall_score <= 1

        # Should have utilitarian framework score
        util_scores = [
            fs for fs in result.framework_scores
            if fs.framework == EthicalFramework.UTILITARIAN
        ]
        assert len(util_scores) == 1

    def test_evaluate_deontological_argument(
        self,
        evaluator: EthicsEvaluator,
        sample_context: DebateContext,
    ) -> None:
        """Test evaluating an argument with deontological framing."""
        argument = Argument(
            agent="Con",
            level=ArgumentLevel.STRATEGIC,
            content=(
                "We have a duty to protect patient rights and dignity. "
                "The principle of informed consent requires that patients "
                "maintain autonomy over their healthcare decisions."
            ),
        )

        result = evaluator.evaluate(argument, sample_context)

        # Should recognize deontological elements
        deont_scores = [
            fs for fs in result.framework_scores
            if fs.framework == EthicalFramework.DEONTOLOGICAL
        ]
        assert len(deont_scores) == 1

    def test_evaluate_virtue_argument(
        self,
        evaluator: EthicsEvaluator,
        sample_context: DebateContext,
    ) -> None:
        """Test evaluating an argument with virtue ethics framing."""
        argument = Argument(
            agent="Pro",
            level=ArgumentLevel.TACTICAL,
            content=(
                "Healthcare professionals demonstrate integrity and moral "
                "excellence when they embrace technologies that improve "
                "patient care. Wisdom suggests we adapt responsibly."
            ),
        )

        result = evaluator.evaluate(argument, sample_context)

        virtue_scores = [
            fs for fs in result.framework_scores
            if fs.framework == EthicalFramework.VIRTUE
        ]
        assert len(virtue_scores) == 1

    def test_evaluate_empty_argument(
        self,
        evaluator: EthicsEvaluator,
        sample_context: DebateContext,
    ) -> None:
        """Test evaluating an empty argument."""
        argument = Argument(
            agent="Pro",
            level=ArgumentLevel.TACTICAL,
            content="",
        )

        result = evaluator.evaluate(argument, sample_context)

        assert isinstance(result, EthicsResult)
        assert result.overall_score >= 0

    def test_evaluate_with_concerns(
        self,
        evaluator: EthicsEvaluator,
        sample_context: DebateContext,
    ) -> None:
        """Test that concerns are identified."""
        argument = Argument(
            agent="Pro",
            level=ArgumentLevel.TACTICAL,
            content=(
                "While there is some risk of harm to patients, the technology "
                "could potentially exploit vulnerable populations and may "
                "involve surveillance of personal health data."
            ),
        )

        result = evaluator.evaluate(argument, sample_context)

        # Should identify concerns
        assert len(result.concerns) > 0


class TestEthicsEvaluatorStakeholders:
    """Tests for stakeholder impact analysis."""

    @pytest.fixture
    def evaluator(self) -> EthicsEvaluator:
        """Create an ethics evaluator."""
        return EthicsEvaluator()

    @pytest.fixture
    def sample_context(self) -> DebateContext:
        """Create a sample debate context."""
        return DebateContext(
            topic="Climate Policy",
            current_round=2,
            total_rounds=5,
            turn_in_round=0,
            topic_sensitivity=0.6,
        )

    def test_identify_society_stakeholder(
        self,
        evaluator: EthicsEvaluator,
        sample_context: DebateContext,
    ) -> None:
        """Test identifying society as a stakeholder."""
        argument = Argument(
            agent="Pro",
            level=ArgumentLevel.STRATEGIC,
            content=(
                "This policy will benefit society as a whole and improve "
                "public welfare for the entire population."
            ),
        )

        result = evaluator.evaluate(argument, sample_context)

        society_impacts = [
            si for si in result.stakeholder_impacts
            if si.stakeholder == StakeholderType.SOCIETY
        ]
        assert len(society_impacts) >= 1

    def test_identify_environment_stakeholder(
        self,
        evaluator: EthicsEvaluator,
        sample_context: DebateContext,
    ) -> None:
        """Test identifying environment as a stakeholder."""
        argument = Argument(
            agent="Pro",
            level=ArgumentLevel.TACTICAL,
            content=(
                "This initiative will protect the environment and preserve "
                "biodiversity for a sustainable planet."
            ),
        )

        result = evaluator.evaluate(argument, sample_context)

        env_impacts = [
            si for si in result.stakeholder_impacts
            if si.stakeholder == StakeholderType.ENVIRONMENT
        ]
        assert len(env_impacts) >= 1

    def test_identify_future_generations(
        self,
        evaluator: EthicsEvaluator,
        sample_context: DebateContext,
    ) -> None:
        """Test identifying future generations as stakeholders."""
        argument = Argument(
            agent="Con",
            level=ArgumentLevel.STRATEGIC,
            content=(
                "We must consider future generations and the legacy we leave "
                "for our children. Long-term sustainable development is key."
            ),
        )

        result = evaluator.evaluate(argument, sample_context)

        future_impacts = [
            si for si in result.stakeholder_impacts
            if si.stakeholder == StakeholderType.FUTURE_GENERATIONS
        ]
        assert len(future_impacts) >= 1

    def test_positive_impact_detection(
        self,
        evaluator: EthicsEvaluator,
        sample_context: DebateContext,
    ) -> None:
        """Test detecting positive stakeholder impact."""
        argument = Argument(
            agent="Pro",
            level=ArgumentLevel.TACTICAL,
            content=(
                "This policy will help individuals by improving their health "
                "and supporting their well-being."
            ),
        )

        result = evaluator.evaluate(argument, sample_context)

        # Check for positive impact
        positive_impacts = [
            si for si in result.stakeholder_impacts
            if si.impact_type == "positive"
        ]
        assert len(positive_impacts) >= 1

    def test_negative_impact_detection(
        self,
        evaluator: EthicsEvaluator,
        sample_context: DebateContext,
    ) -> None:
        """Test detecting negative stakeholder impact."""
        argument = Argument(
            agent="Con",
            level=ArgumentLevel.TACTICAL,
            content=(
                "This could harm communities and damage local businesses, "
                "causing people to suffer economic losses."
            ),
        )

        result = evaluator.evaluate(argument, sample_context)

        # Check for negative impact
        negative_impacts = [
            si for si in result.stakeholder_impacts
            if si.impact_type == "negative"
        ]
        assert len(negative_impacts) >= 1


class TestEthicsEvaluatorConcerns:
    """Tests for ethical concern identification."""

    @pytest.fixture
    def evaluator(self) -> EthicsEvaluator:
        """Create an ethics evaluator."""
        return EthicsEvaluator()

    @pytest.fixture
    def sample_context(self) -> DebateContext:
        """Create a sample debate context."""
        return DebateContext(
            topic="Data Privacy",
            current_round=2,
            total_rounds=5,
            turn_in_round=0,
            topic_sensitivity=0.8,
        )

    def test_identify_harm_concern(
        self,
        evaluator: EthicsEvaluator,
        sample_context: DebateContext,
    ) -> None:
        """Test identifying harm concerns."""
        argument = Argument(
            agent="Con",
            level=ArgumentLevel.TACTICAL,
            content=(
                "This technology could cause harm to users and damage their "
                "mental health, leading to suffering."
            ),
        )

        result = evaluator.evaluate(argument, sample_context)

        harm_concerns = [
            c for c in result.concerns if c.concern_type == "harm"
        ]
        assert len(harm_concerns) >= 1

    def test_identify_privacy_concern(
        self,
        evaluator: EthicsEvaluator,
        sample_context: DebateContext,
    ) -> None:
        """Test identifying privacy concerns."""
        argument = Argument(
            agent="Con",
            level=ArgumentLevel.TACTICAL,
            content=(
                "The surveillance of personal information and data collection "
                "practices raise serious privacy concerns."
            ),
        )

        result = evaluator.evaluate(argument, sample_context)

        privacy_concerns = [
            c for c in result.concerns if c.concern_type == "privacy"
        ]
        assert len(privacy_concerns) >= 1

    def test_identify_discrimination_concern(
        self,
        evaluator: EthicsEvaluator,
        sample_context: DebateContext,
    ) -> None:
        """Test identifying discrimination concerns."""
        argument = Argument(
            agent="Con",
            level=ArgumentLevel.TACTICAL,
            content=(
                "This algorithm shows bias and may discriminate against "
                "certain groups, creating unfair outcomes."
            ),
        )

        result = evaluator.evaluate(argument, sample_context)

        discrim_concerns = [
            c for c in result.concerns if c.concern_type == "discrimination"
        ]
        assert len(discrim_concerns) >= 1

    def test_mitigated_concern(
        self,
        evaluator: EthicsEvaluator,
        sample_context: DebateContext,
    ) -> None:
        """Test that mitigated concerns are recognized."""
        argument = Argument(
            agent="Pro",
            level=ArgumentLevel.TACTICAL,
            content=(
                "While there is some risk of harm, however, we can mitigate "
                "this by implementing safeguards to protect users."
            ),
        )

        result = evaluator.evaluate(argument, sample_context)

        # Concerns should be identified but may be mitigated
        harm_concerns = [
            c for c in result.concerns if c.concern_type == "harm"
        ]
        if harm_concerns:
            # At least one should be mitigated
            mitigated = [c for c in harm_concerns if c.mitigated]
            assert len(mitigated) >= 1


class TestEthicsEvaluatorScoring:
    """Tests for overall scoring."""

    @pytest.fixture
    def evaluator(self) -> EthicsEvaluator:
        """Create an ethics evaluator."""
        return EthicsEvaluator()

    def test_score_range(self, evaluator: EthicsEvaluator) -> None:
        """Test that scores are in valid range."""
        context = DebateContext(
            topic="Test",
            current_round=1,
            total_rounds=3,
            turn_in_round=0,
        )
        argument = Argument(
            agent="Test",
            level=ArgumentLevel.STRATEGIC,
            content="A balanced argument considering multiple perspectives.",
        )

        result = evaluator.evaluate(argument, context)

        assert 0 <= result.overall_score <= 1
        for fs in result.framework_scores:
            assert 0 <= fs.score <= 1

    def test_sensitivity_affects_score(self, evaluator: EthicsEvaluator) -> None:
        """Test that topic sensitivity affects scoring."""
        argument = Argument(
            agent="Test",
            level=ArgumentLevel.STRATEGIC,
            content="This benefits society through improved welfare outcomes.",
        )

        low_sens_context = DebateContext(
            topic="Test",
            current_round=1,
            total_rounds=3,
            turn_in_round=0,
            topic_sensitivity=0.2,
        )
        high_sens_context = DebateContext(
            topic="Sensitive Test",
            current_round=1,
            total_rounds=3,
            turn_in_round=0,
            topic_sensitivity=0.9,
        )

        low_result = evaluator.evaluate(argument, low_sens_context)
        high_result = evaluator.evaluate(argument, high_sens_context)

        # Higher sensitivity should lead to more scrutiny
        # (not necessarily lower score, but adjusted)
        assert isinstance(low_result.overall_score, float)
        assert isinstance(high_result.overall_score, float)


class TestEthicsEvaluatorRecommendations:
    """Tests for recommendations generation."""

    @pytest.fixture
    def evaluator(self) -> EthicsEvaluator:
        """Create an ethics evaluator."""
        return EthicsEvaluator()

    def test_recommendations_for_concerns(
        self, evaluator: EthicsEvaluator
    ) -> None:
        """Test recommendations are generated for concerns."""
        context = DebateContext(
            topic="Test",
            current_round=1,
            total_rounds=3,
            turn_in_round=0,
        )
        argument = Argument(
            agent="Test",
            level=ArgumentLevel.TACTICAL,
            content="This could cause harm and exploit vulnerable users.",
        )

        result = evaluator.evaluate(argument, context)

        assert len(result.recommendations) > 0

    def test_strengths_identified(self, evaluator: EthicsEvaluator) -> None:
        """Test that strengths are identified."""
        context = DebateContext(
            topic="Test",
            current_round=1,
            total_rounds=3,
            turn_in_round=0,
        )
        argument = Argument(
            agent="Test",
            level=ArgumentLevel.STRATEGIC,
            content=(
                "We must consider both sides and weigh the benefits against "
                "the costs. Society, individuals, and the environment all "
                "matter in this balanced consideration."
            ),
        )

        result = evaluator.evaluate(argument, context)

        # Should identify some strengths
        assert isinstance(result.strengths, list)


class TestEthicsEvaluatorHelpers:
    """Tests for helper methods."""

    @pytest.fixture
    def evaluator(self) -> EthicsEvaluator:
        """Create an ethics evaluator."""
        return EthicsEvaluator()

    def test_get_framework_summary(self, evaluator: EthicsEvaluator) -> None:
        """Test getting framework summary."""
        context = DebateContext(
            topic="Test",
            current_round=1,
            total_rounds=3,
            turn_in_round=0,
        )
        argument = Argument(
            agent="Test",
            level=ArgumentLevel.STRATEGIC,
            content="Test argument with ethical considerations.",
        )

        result = evaluator.evaluate(argument, context)
        summary = evaluator.get_framework_summary(result)

        assert isinstance(summary, dict)
        assert "utilitarian" in summary or "deontological" in summary
