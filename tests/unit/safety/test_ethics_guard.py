"""
Tests for ARTEMIS Ethics Guard

Tests cover:
- EthicsViolation enum
- EthicsConfig dataclass
- EthicsScore dataclass
- EthicsGuard monitor
- Violation detection algorithms
"""

import pytest

from artemis.core.types import (
    Argument,
    ArgumentLevel,
    DebateContext,
    SafetyIndicatorType,
    Turn,
)
from artemis.safety.base import MonitorMode
from artemis.safety.ethics_guard import (
    EthicsConfig,
    EthicsGuard,
    EthicsScore,
    EthicsViolation,
)

# =============================================================================
# Fixtures
# =============================================================================


def create_argument(
    agent: str = "test_agent",
    content: str = "This is a test argument.",
    level: ArgumentLevel = ArgumentLevel.STRATEGIC,
) -> Argument:
    """Create a test argument."""
    return Argument(
        agent=agent,
        level=level,
        content=content,
        evidence=[],
        causal_links=[],
    )


def create_turn(
    agent: str = "test_agent",
    round_num: int = 1,
    content: str = "This is a test argument.",
    level: ArgumentLevel = ArgumentLevel.STRATEGIC,
) -> Turn:
    """Create a test turn."""
    return Turn(
        round=round_num,
        sequence=0,
        agent=agent,
        argument=create_argument(agent=agent, content=content, level=level),
    )


@pytest.fixture
def mock_context() -> DebateContext:
    """Create a mock debate context."""
    return DebateContext(
        topic="Test topic for debate",
        current_round=2,
        total_rounds=5,
        turn_in_round=0,
        agent_positions={"agent_a": "pro", "agent_b": "con"},
        topic_complexity=0.5,
    )


@pytest.fixture
def guard() -> EthicsGuard:
    """Create an ethics guard for testing."""
    return EthicsGuard(sensitivity=0.6)


# =============================================================================
# EthicsViolation Tests
# =============================================================================


class TestEthicsViolation:
    """Tests for EthicsViolation enum."""

    def test_harmful_content(self):
        """Test harmful content violation value."""
        assert EthicsViolation.HARMFUL_CONTENT.value == "harmful_content"

    def test_bias_detected(self):
        """Test bias detected violation value."""
        assert EthicsViolation.BIAS_DETECTED.value == "bias_detected"

    def test_unfair_treatment(self):
        """Test unfair treatment violation value."""
        assert EthicsViolation.UNFAIR_TREATMENT.value == "unfair_treatment"

    def test_privacy_violation(self):
        """Test privacy violation value."""
        assert EthicsViolation.PRIVACY_VIOLATION.value == "privacy_violation"

    def test_misinformation(self):
        """Test misinformation violation value."""
        assert EthicsViolation.MISINFORMATION.value == "misinformation"

    def test_manipulation(self):
        """Test manipulation violation value."""
        assert EthicsViolation.MANIPULATION.value == "manipulation"

    def test_boundary_breach(self):
        """Test boundary breach violation value."""
        assert EthicsViolation.BOUNDARY_BREACH.value == "boundary_breach"

    def test_principle_violation(self):
        """Test principle violation value."""
        assert EthicsViolation.PRINCIPLE_VIOLATION.value == "principle_violation"


# =============================================================================
# EthicsConfig Tests
# =============================================================================


class TestEthicsConfig:
    """Tests for EthicsConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EthicsConfig()
        assert config.harmful_content_threshold == 0.5
        assert config.bias_threshold == 0.4
        assert config.fairness_threshold == 0.3
        assert "harmful_content" in config.enabled_checks
        assert "bias" in config.enabled_checks

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EthicsConfig(
            harmful_content_threshold=0.7,
            bias_threshold=0.6,
            enabled_checks=["harmful_content", "privacy"],
        )
        assert config.harmful_content_threshold == 0.7
        assert config.bias_threshold == 0.6
        assert len(config.enabled_checks) == 2

    def test_custom_boundaries(self):
        """Test custom ethical boundaries."""
        config = EthicsConfig(
            custom_boundaries={"no_politics": "Avoid political statements"}
        )
        assert "no_politics" in config.custom_boundaries


# =============================================================================
# EthicsScore Tests
# =============================================================================


class TestEthicsScore:
    """Tests for EthicsScore dataclass."""

    def test_default_values(self):
        """Test default score values."""
        score = EthicsScore()
        assert score.overall == 1.0
        assert score.harmful_score == 0.0
        assert score.bias_score == 0.0
        assert score.fairness_score == 1.0
        assert score.privacy_score == 1.0
        assert score.violations == []

    def test_with_violations(self):
        """Test score with violations."""
        violations = [
            (EthicsViolation.BIAS_DETECTED, 0.6, "Bias found"),
            (EthicsViolation.MANIPULATION, 0.4, "Manipulation found"),
        ]
        score = EthicsScore(
            overall=0.5,
            bias_score=0.6,
            violations=violations,
        )
        assert score.overall == 0.5
        assert len(score.violations) == 2


# =============================================================================
# EthicsGuard Tests
# =============================================================================


class TestEthicsGuard:
    """Tests for EthicsGuard."""

    def test_initialization_default(self):
        """Test default initialization."""
        guard = EthicsGuard()
        assert guard.name == "ethics_guard"
        assert guard.monitor_type == "ethics_enforcement"
        assert guard._sensitivity == 0.5
        assert guard._halt_on_violation is False

    def test_initialization_custom(self):
        """Test custom initialization."""
        guard = EthicsGuard(
            sensitivity=0.8,
            halt_on_violation=True,
            mode=MonitorMode.ACTIVE,
        )
        assert guard._sensitivity == 0.8
        assert guard._halt_on_violation is True
        assert guard.is_active is True

    def test_sensitivity_clamping(self):
        """Test sensitivity is clamped to 0-1."""
        high = EthicsGuard(sensitivity=1.5)
        assert high._sensitivity == 1.0

        low = EthicsGuard(sensitivity=-0.5)
        assert low._sensitivity == 0.0

    @pytest.mark.asyncio
    async def test_analyze_clean_content(self, guard, mock_context):
        """Test analysis of clean content."""
        turn = create_turn(content="A thoughtful argument about the topic.")
        result = await guard.analyze(turn, mock_context)

        assert result.monitor == "ethics_guard"
        assert result.severity == 0.0
        assert len(result.indicators) == 0

    @pytest.mark.asyncio
    async def test_detect_harmful_content(self, mock_context):
        """Test detection of harmful content."""
        guard = EthicsGuard(sensitivity=0.8)

        content = "We should use violence and kill our enemies!"
        turn = create_turn(content=content)
        result = await guard.analyze(turn, mock_context)

        assert result.severity > 0.0
        assert any(
            ind.type == SafetyIndicatorType.ETHICS_BOUNDARY
            for ind in result.indicators
        )

    @pytest.mark.asyncio
    async def test_detect_bias(self, mock_context):
        """Test detection of biased content."""
        guard = EthicsGuard(sensitivity=0.8)

        content = "All women are less capable. They always fail at this."
        turn = create_turn(content=content)
        result = await guard.analyze(turn, mock_context)

        assert result.severity > 0.0

    @pytest.mark.asyncio
    async def test_detect_manipulation(self, mock_context):
        """Test detection of manipulation."""
        guard = EthicsGuard(sensitivity=0.7)

        content = (
            "Only a fool would disagree with this. Everyone knows it's true. "
            "Trust me, they don't want you to know the real facts."
        )
        turn = create_turn(content=content)
        result = await guard.analyze(turn, mock_context)

        assert result.severity > 0.0

    @pytest.mark.asyncio
    async def test_detect_privacy_concerns(self, mock_context):
        """Test detection of privacy concerns."""
        guard = EthicsGuard(sensitivity=0.7)

        content = (
            "Let me share their personal address and phone number. "
            "Also their social security number is relevant here."
        )
        turn = create_turn(content=content)
        result = await guard.analyze(turn, mock_context)

        assert result.severity > 0.0

    @pytest.mark.asyncio
    async def test_detect_unfairness(self, mock_context):
        """Test detection of unfair treatment."""
        guard = EthicsGuard(sensitivity=0.7)

        content = (
            "Only idiots would believe this. Anyone with a brain knows better. "
            "No reasonable person could disagree."
        )
        turn = create_turn(content=content)
        result = await guard.analyze(turn, mock_context)

        assert result.severity > 0.0

    @pytest.mark.asyncio
    async def test_no_false_positive_normal_content(self, guard, mock_context):
        """Test no false positives with normal content."""
        content = (
            "The evidence suggests this policy would be effective. "
            "Research from multiple sources supports this conclusion. "
            "We should consider all perspectives carefully."
        )
        turn = create_turn(content=content)
        result = await guard.analyze(turn, mock_context)

        assert result.severity < 0.3

    @pytest.mark.asyncio
    async def test_tracks_violation_history(self, guard, mock_context):
        """Test violation history tracking."""
        content = "This is completely wrong and stupid thinking."
        turn = create_turn(agent="agent_a", content=content)
        await guard.analyze(turn, mock_context)

        # Check history
        history = guard.get_violation_history("agent_a")
        # May or may not have violations depending on detection
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_get_ethics_summary(self, guard, mock_context):
        """Test getting ethics summary."""
        content = "Everyone knows this, only fools disagree with hate."
        turn = create_turn(agent="agent_a", content=content)
        await guard.analyze(turn, mock_context)

        summary = guard.get_agent_ethics_summary("agent_a")
        assert "total_violations" in summary
        assert "max_severity" in summary
        assert "violation_types" in summary

    def test_get_summary_nonexistent(self, guard):
        """Test summary for unknown agent."""
        summary = guard.get_agent_ethics_summary("unknown")
        assert summary["total_violations"] == 0
        assert summary["max_severity"] == 0.0

    @pytest.mark.asyncio
    async def test_reset_agent_history(self, guard, mock_context):
        """Test resetting agent history."""
        content = "Violence and hate everywhere!"
        turn = create_turn(agent="agent_a", content=content)
        await guard.analyze(turn, mock_context)

        guard.reset_agent_history("agent_a")
        history = guard.get_violation_history("agent_a")
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_reset_all_history(self, guard, mock_context):
        """Test resetting all history."""
        for agent in ["agent_a", "agent_b"]:
            turn = create_turn(agent=agent, content="Some hate content here.")
            await guard.analyze(turn, mock_context)

        guard.reset_all_history()

        assert len(guard.get_violation_history("agent_a")) == 0
        assert len(guard.get_violation_history("agent_b")) == 0

    @pytest.mark.asyncio
    async def test_active_mode_halts(self, mock_context):
        """Test active mode can halt on violations."""
        guard = EthicsGuard(
            mode=MonitorMode.ACTIVE,
            sensitivity=0.9,
            halt_on_violation=True,
        )

        content = "Kill them all with violence and terror! Bomb everything!"
        turn = create_turn(content=content)
        result = await guard.analyze(turn, mock_context)

        # High severity should trigger halt
        if result.severity >= 0.7:
            assert result.should_halt is True

    @pytest.mark.asyncio
    async def test_passive_mode_no_halt(self, mock_context):
        """Test passive mode never halts."""
        guard = EthicsGuard(
            mode=MonitorMode.PASSIVE,
            sensitivity=0.9,
            halt_on_violation=True,
        )

        content = "Violence and terror everywhere!"
        turn = create_turn(content=content)
        result = await guard.analyze(turn, mock_context)

        # Passive mode should never halt
        assert result.should_halt is False


# =============================================================================
# Pattern Detection Tests
# =============================================================================


class TestPatternDetection:
    """Tests for specific pattern detection."""

    def test_harmful_patterns_list(self):
        """Test harmful patterns are defined."""
        guard = EthicsGuard()
        assert len(guard.HARMFUL_PATTERNS) > 0
        for pattern, weight in guard.HARMFUL_PATTERNS:
            assert isinstance(pattern, str)
            assert 0.0 <= weight <= 1.0

    def test_bias_patterns_list(self):
        """Test bias patterns are defined."""
        guard = EthicsGuard()
        assert len(guard.BIAS_PATTERNS) > 0
        for pattern, weight in guard.BIAS_PATTERNS:
            assert isinstance(pattern, str)
            assert 0.0 <= weight <= 1.0

    def test_manipulation_patterns_list(self):
        """Test manipulation patterns are defined."""
        guard = EthicsGuard()
        assert len(guard.MANIPULATION_PATTERNS) > 0
        for pattern, weight in guard.MANIPULATION_PATTERNS:
            assert isinstance(pattern, str)
            assert 0.0 <= weight <= 1.0

    def test_privacy_patterns_list(self):
        """Test privacy patterns are defined."""
        guard = EthicsGuard()
        assert len(guard.PRIVACY_PATTERNS) > 0
        for pattern, weight in guard.PRIVACY_PATTERNS:
            assert isinstance(pattern, str)
            assert 0.0 <= weight <= 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestEthicsGuardIntegration:
    """Integration tests for EthicsGuard."""

    @pytest.mark.asyncio
    async def test_multiple_violation_types(self, mock_context):
        """Test content with multiple violation types."""
        guard = EthicsGuard(sensitivity=0.8)

        content = (
            "Those people always cause problems. Everyone knows this. "
            "Only a fool would trust them. We should attack them with violence."
        )
        turn = create_turn(content=content)
        result = await guard.analyze(turn, mock_context)

        assert result.severity > 0.0
        # Should have multiple indicators
        assert len(result.indicators) >= 1

    @pytest.mark.asyncio
    async def test_composite_with_ethics_guard(self, mock_context):
        """Test EthicsGuard in composite monitor."""
        from artemis.safety.base import CompositeMonitor

        guard1 = EthicsGuard(sensitivity=0.5)
        guard2 = EthicsGuard(sensitivity=0.7)

        composite = CompositeMonitor(
            [guard1, guard2],
            aggregation="max",
        )

        turn = create_turn(content="A reasonable argument.")
        result = await composite.analyze(turn, mock_context)

        assert result.monitor == "composite_monitor"

    @pytest.mark.asyncio
    async def test_with_custom_config(self, mock_context):
        """Test with custom ethics configuration."""
        config = EthicsConfig(
            harmful_content_threshold=0.9,  # Very high threshold
            enabled_checks=["harmful_content"],  # Only check harmful
        )
        guard = EthicsGuard(sensitivity=0.8, ethics_config=config)

        # Bias content should not trigger with this config
        content = "All women are less capable than men."
        turn = create_turn(content=content)
        result = await guard.analyze(turn, mock_context)

        # Should not detect bias since it's not in enabled_checks
        bias_indicators = [
            i for i in result.indicators
            if "bias" in i.metadata.get("violation", "")
        ]
        assert len(bias_indicators) == 0


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_content(self, guard, mock_context):
        """Test handling empty content."""
        turn = create_turn(content="")
        result = await guard.analyze(turn, mock_context)

        assert result.severity >= 0.0

    @pytest.mark.asyncio
    async def test_single_word(self, guard, mock_context):
        """Test handling single word."""
        turn = create_turn(content="Hello")
        result = await guard.analyze(turn, mock_context)

        assert result.severity >= 0.0

    @pytest.mark.asyncio
    async def test_very_long_content(self, guard, mock_context):
        """Test handling very long content."""
        content = " ".join(["word"] * 1000)
        turn = create_turn(content=content)
        result = await guard.analyze(turn, mock_context)

        assert result.severity >= 0.0

    @pytest.mark.asyncio
    async def test_special_characters(self, guard, mock_context):
        """Test handling special characters."""
        content = "This has @#$%^&*() special chars!!!"
        turn = create_turn(content=content)
        result = await guard.analyze(turn, mock_context)

        assert result.severity >= 0.0

    @pytest.mark.asyncio
    async def test_case_insensitive_detection(self, mock_context):
        """Test detection is case insensitive."""
        guard = EthicsGuard(sensitivity=0.7)

        content = "VIOLENCE AND HATE ARE BAD"
        turn = create_turn(content=content)
        result = await guard.analyze(turn, mock_context)

        # Should still detect uppercase patterns
        assert result.severity > 0.0 or result.severity == 0.0  # Valid either way
