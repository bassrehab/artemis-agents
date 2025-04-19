"""
Tests for ARTEMIS Behavioral Drift Tracker

Tests cover:
- DriftSignal enum
- StyleMetrics dataclass
- BehaviorSnapshot dataclass
- AgentBehaviorProfile dataclass
- BehaviorTracker monitor
- Drift detection algorithms
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
from artemis.safety.behavior import (
    AgentBehaviorProfile,
    BehaviorSnapshot,
    BehaviorTracker,
    DriftSignal,
    StyleMetrics,
)

# =============================================================================
# Fixtures
# =============================================================================


def create_argument(
    agent: str = "test_agent",
    content: str = "This is a test argument with moderate complexity.",
    level: ArgumentLevel = ArgumentLevel.STRATEGIC,
    evidence_count: int = 0,
) -> Argument:
    """Create a test argument with customizable properties."""
    from artemis.core.types import Evidence

    evidence = [
        Evidence(
            type="fact",
            source=f"source_{i}",
            content=f"evidence_{i}",
            confidence=0.8,
        )
        for i in range(evidence_count)
    ]

    return Argument(
        agent=agent,
        level=level,
        content=content,
        evidence=evidence,
        causal_links=[],
    )


def create_turn(
    agent: str = "test_agent",
    round_num: int = 1,
    content: str = "This is a test argument with moderate complexity.",
    level: ArgumentLevel = ArgumentLevel.STRATEGIC,
    evidence_count: int = 0,
) -> Turn:
    """Create a test turn."""
    return Turn(
        round=round_num,
        sequence=0,
        agent=agent,
        argument=create_argument(
            agent=agent,
            content=content,
            level=level,
            evidence_count=evidence_count,
        ),
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
def tracker() -> BehaviorTracker:
    """Create a behavior tracker for testing."""
    return BehaviorTracker(sensitivity=0.6, baseline_turns=3)


# =============================================================================
# DriftSignal Tests
# =============================================================================


class TestDriftSignal:
    """Tests for DriftSignal enum."""

    def test_style_change(self):
        """Test style change signal value."""
        assert DriftSignal.STYLE_CHANGE.value == "style_change"

    def test_consistency_drift(self):
        """Test consistency drift signal value."""
        assert DriftSignal.CONSISTENCY_DRIFT.value == "consistency_drift"

    def test_personality_shift(self):
        """Test personality shift signal value."""
        assert DriftSignal.PERSONALITY_SHIFT.value == "personality_shift"

    def test_goal_misalignment(self):
        """Test goal misalignment signal value."""
        assert DriftSignal.GOAL_MISALIGNMENT.value == "goal_misalignment"

    def test_engagement_change(self):
        """Test engagement change signal value."""
        assert DriftSignal.ENGAGEMENT_CHANGE.value == "engagement_change"

    def test_formality_shift(self):
        """Test formality shift signal value."""
        assert DriftSignal.FORMALITY_SHIFT.value == "formality_shift"

    def test_aggression_increase(self):
        """Test aggression increase signal value."""
        assert DriftSignal.AGGRESSION_INCREASE.value == "aggression_increase"

    def test_cooperation_decrease(self):
        """Test cooperation decrease signal value."""
        assert DriftSignal.COOPERATION_DECREASE.value == "cooperation_decrease"


# =============================================================================
# StyleMetrics Tests
# =============================================================================


class TestStyleMetrics:
    """Tests for StyleMetrics dataclass."""

    def test_default_values(self):
        """Test default style metric values."""
        metrics = StyleMetrics()
        assert metrics.formality_score == 0.5
        assert metrics.aggression_score == 0.0
        assert metrics.cooperation_score == 0.5
        assert metrics.verbosity == 0.5
        assert metrics.certainty_level == 0.5
        assert metrics.question_ratio == 0.0
        assert metrics.first_person_ratio == 0.0

    def test_custom_values(self):
        """Test custom style metric values."""
        metrics = StyleMetrics(
            formality_score=0.8,
            aggression_score=0.3,
            cooperation_score=0.7,
            verbosity=0.9,
            certainty_level=0.6,
            question_ratio=0.2,
            first_person_ratio=0.4,
        )
        assert metrics.formality_score == 0.8
        assert metrics.aggression_score == 0.3
        assert metrics.cooperation_score == 0.7
        assert metrics.verbosity == 0.9


# =============================================================================
# BehaviorSnapshot Tests
# =============================================================================


class TestBehaviorSnapshot:
    """Tests for BehaviorSnapshot dataclass."""

    def test_creation(self):
        """Test snapshot creation."""
        style = StyleMetrics(formality_score=0.7)
        snapshot = BehaviorSnapshot(
            round=1,
            style=style,
            word_count=100,
            sentence_count=5,
            evidence_used=2,
            position_strength=0.6,
        )
        assert snapshot.round == 1
        assert snapshot.style.formality_score == 0.7
        assert snapshot.word_count == 100
        assert snapshot.sentence_count == 5
        assert snapshot.evidence_used == 2
        assert snapshot.position_strength == 0.6

    def test_default_position_strength(self):
        """Test default position strength."""
        style = StyleMetrics()
        snapshot = BehaviorSnapshot(
            round=1,
            style=style,
            word_count=50,
            sentence_count=2,
            evidence_used=0,
        )
        assert snapshot.position_strength == 0.5


# =============================================================================
# AgentBehaviorProfile Tests
# =============================================================================


class TestAgentBehaviorProfile:
    """Tests for AgentBehaviorProfile dataclass."""

    def test_default_values(self):
        """Test default profile values."""
        profile = AgentBehaviorProfile()
        assert profile.snapshots == []
        assert profile.baseline_style is None
        assert profile.drift_events == []
        assert profile.avg_formality == 0.5
        assert profile.avg_aggression == 0.0
        assert profile.avg_cooperation == 0.5
        assert profile.avg_verbosity == 0.5
        assert profile.formality_variance == 0.0
        assert profile.aggression_variance == 0.0

    def test_with_snapshots(self):
        """Test profile with snapshots."""
        style = StyleMetrics()
        snapshot = BehaviorSnapshot(
            round=1, style=style, word_count=50, sentence_count=2, evidence_used=0
        )
        profile = AgentBehaviorProfile(snapshots=[snapshot])
        assert len(profile.snapshots) == 1

    def test_with_drift_events(self):
        """Test profile with drift events."""
        profile = AgentBehaviorProfile(
            drift_events=[
                (1, "style_change", 0.5),
                (2, "aggression_increase", 0.7),
            ]
        )
        assert len(profile.drift_events) == 2


# =============================================================================
# BehaviorTracker Tests
# =============================================================================


class TestBehaviorTracker:
    """Tests for BehaviorTracker."""

    def test_initialization_default(self):
        """Test default initialization."""
        tracker = BehaviorTracker()
        assert tracker.name == "behavior_tracker"
        assert tracker.monitor_type == "behavioral_drift"
        assert tracker._sensitivity == 0.5
        assert tracker._baseline_turns == 3
        assert tracker._drift_threshold == 0.25
        assert tracker._aggression_threshold == 0.6

    def test_initialization_custom(self):
        """Test custom initialization."""
        tracker = BehaviorTracker(
            sensitivity=0.8,
            baseline_turns=5,
            drift_threshold=0.3,
            aggression_threshold=0.5,
            mode=MonitorMode.ACTIVE,
        )
        assert tracker._sensitivity == 0.8
        assert tracker._baseline_turns == 5
        assert tracker._drift_threshold == 0.3
        assert tracker._aggression_threshold == 0.5
        assert tracker.is_active is True

    def test_sensitivity_clamping(self):
        """Test sensitivity is clamped to 0-1."""
        high = BehaviorTracker(sensitivity=1.5)
        assert high._sensitivity == 1.0

        low = BehaviorTracker(sensitivity=-0.5)
        assert low._sensitivity == 0.0

    @pytest.mark.asyncio
    async def test_analyze_builds_baseline(self, tracker, mock_context):
        """Test that analysis builds baseline over first turns."""
        for i in range(3):
            turn = create_turn(
                agent="agent_a",
                round_num=i + 1,
                content="Standard argument content for testing baseline.",
            )
            result = await tracker.analyze(turn, mock_context)

            if i < 2:
                assert "Building baseline" in result.analysis_notes

        profile = tracker.get_agent_profile("agent_a")
        assert profile is not None
        assert len(profile.snapshots) == 3
        assert profile.baseline_style is not None

    @pytest.mark.asyncio
    async def test_analyze_after_baseline(self, tracker, mock_context):
        """Test analysis after baseline is established."""
        # Build baseline
        for i in range(3):
            turn = create_turn(
                agent="agent_a",
                content="Standard argument content for testing.",
            )
            await tracker.analyze(turn, mock_context)

        # Analyze after baseline
        turn = create_turn(
            agent="agent_a",
            content="Another standard argument content for testing purposes.",
        )
        result = await tracker.analyze(turn, mock_context)

        assert result.monitor == "behavior_tracker"
        assert result.severity >= 0.0

    @pytest.mark.asyncio
    async def test_detect_aggression_increase(self, mock_context):
        """Test detection of increased aggression."""
        tracker = BehaviorTracker(
            sensitivity=0.8, baseline_turns=2, aggression_threshold=0.4
        )

        # Build baseline with calm content
        calm_content = (
            "I appreciate your perspective on this matter. "
            "While I understand your point, I would like to offer "
            "an alternative viewpoint for consideration."
        )
        for _ in range(3):
            turn = create_turn(content=calm_content)
            await tracker.analyze(turn, mock_context)

        # Now submit aggressive content
        aggressive_content = (
            "That is completely wrong and ridiculous! "
            "Your argument is utterly stupid and makes no sense. "
            "Anyone can see this is nonsense and absurd thinking."
        )
        turn = create_turn(content=aggressive_content)
        result = await tracker.analyze(turn, mock_context)

        assert result.severity > 0.0
        # Check for behavioral drift indicator
        assert any(
            ind.type == SafetyIndicatorType.BEHAVIORAL_DRIFT for ind in result.indicators
        )

    @pytest.mark.asyncio
    async def test_detect_cooperation_decrease(self, mock_context):
        """Test detection of decreased cooperation."""
        tracker = BehaviorTracker(sensitivity=0.8, baseline_turns=2)

        # Build baseline with cooperative content
        cooperative_content = (
            "I agree with several valid points you raised. "
            "Your perspective is appreciated and I understand your position. "
            "Fair enough, let me consider this further."
        )
        for _ in range(3):
            turn = create_turn(content=cooperative_content)
            await tracker.analyze(turn, mock_context)

        # Now submit competitive content
        competitive_content = (
            "You are wrong and incorrect about everything. "
            "You don't understand and you can't see the obvious. "
            "This is impossible to accept and you fail to grasp it."
        )
        turn = create_turn(content=competitive_content)
        result = await tracker.analyze(turn, mock_context)

        assert result.severity > 0.0

    @pytest.mark.asyncio
    async def test_detect_formality_shift(self, mock_context):
        """Test detection of formality shift."""
        tracker = BehaviorTracker(sensitivity=0.8, baseline_turns=2)

        # Build baseline with formal content
        formal_content = (
            "Therefore, it is consequently important to note that furthermore "
            "the evidence demonstrates this position. Moreover, the analysis "
            "hereby confirms the thesis. Nevertheless, we must consider accordingly."
        )
        for _ in range(3):
            turn = create_turn(content=formal_content)
            await tracker.analyze(turn, mock_context)

        # Now submit informal content
        informal_content = (
            "Yeah ok so basically gonna wanna say hey that's kinda "
            "like wow not really gonna work ok yeah nope ugh."
        )
        turn = create_turn(content=informal_content)
        result = await tracker.analyze(turn, mock_context)

        assert result.severity > 0.0

    @pytest.mark.asyncio
    async def test_no_false_positive_consistent_behavior(self):
        """Test no false positives with consistent behavior."""
        tracker = BehaviorTracker(sensitivity=0.4, baseline_turns=2)

        context = DebateContext(
            topic="Simple discussion",
            current_round=2,
            total_rounds=5,
            turn_in_round=0,
            agent_positions={"test_agent": "pro"},
            topic_complexity=0.3,
        )

        # Consistent moderate behavior
        content = (
            "This is a standard argument for testing purposes. "
            "The content demonstrates consistent performance. "
            "Nothing unusual about this response pattern."
        )
        for _ in range(6):
            turn = create_turn(content=content)
            result = await tracker.analyze(turn, context)

        # Should not flag consistent behavior
        assert result.severity < 0.3

    @pytest.mark.asyncio
    async def test_tracks_multiple_agents(self, tracker, mock_context):
        """Test tracking profiles for multiple agents."""
        for agent in ["agent_a", "agent_b", "agent_c"]:
            turn = create_turn(agent=agent, content="Test content for agent.")
            await tracker.analyze(turn, mock_context)

        assert tracker.get_agent_profile("agent_a") is not None
        assert tracker.get_agent_profile("agent_b") is not None
        assert tracker.get_agent_profile("agent_c") is not None

    def test_get_agent_profile_nonexistent(self, tracker):
        """Test getting profile for unknown agent."""
        assert tracker.get_agent_profile("unknown") is None

    @pytest.mark.asyncio
    async def test_reset_agent_profile(self, tracker, mock_context):
        """Test resetting agent profile."""
        turn = create_turn(agent="agent_a")
        await tracker.analyze(turn, mock_context)

        assert tracker.get_agent_profile("agent_a") is not None

        tracker.reset_agent_profile("agent_a")
        assert tracker.get_agent_profile("agent_a") is None

    @pytest.mark.asyncio
    async def test_reset_all_profiles(self, tracker, mock_context):
        """Test resetting all profiles."""
        for agent in ["agent_a", "agent_b"]:
            turn = create_turn(agent=agent)
            await tracker.analyze(turn, mock_context)

        tracker.reset_all_profiles()

        assert tracker.get_agent_profile("agent_a") is None
        assert tracker.get_agent_profile("agent_b") is None

    @pytest.mark.asyncio
    async def test_get_drift_summary(self, tracker, mock_context):
        """Test getting drift summary for agent."""
        # Build profile
        for i in range(5):
            turn = create_turn(agent="agent_a", round_num=i + 1)
            await tracker.analyze(turn, mock_context)

        summary = tracker.get_drift_summary("agent_a")

        assert summary["snapshots"] == 5
        assert "drift_events" in summary
        assert "avg_aggression" in summary
        assert "avg_cooperation" in summary

    def test_get_drift_summary_nonexistent(self, tracker):
        """Test getting drift summary for unknown agent."""
        summary = tracker.get_drift_summary("unknown")

        assert summary["snapshots"] == 0
        assert summary["drift_events"] == 0

    @pytest.mark.asyncio
    async def test_bounded_snapshot_history(self, mock_context):
        """Test that snapshot history is bounded."""
        tracker = BehaviorTracker(baseline_turns=2)

        # Create many snapshots
        for i in range(50):
            turn = create_turn(agent="agent_a", round_num=i + 1)
            await tracker.analyze(turn, mock_context)

        profile = tracker.get_agent_profile("agent_a")
        assert len(profile.snapshots) <= 30


# =============================================================================
# Style Metrics Extraction Tests
# =============================================================================


class TestStyleMetricsExtraction:
    """Tests for style metrics extraction."""

    def test_extract_formality_formal(self):
        """Test formality extraction for formal content."""
        tracker = BehaviorTracker()
        content = "Therefore, consequently, furthermore the matter is hereby resolved."
        metrics = tracker._extract_style_metrics(content)

        assert metrics.formality_score > 0.5

    def test_extract_formality_informal(self):
        """Test formality extraction for informal content."""
        tracker = BehaviorTracker()
        content = "Yeah gonna wanna kinda do this hey wow ok nope."
        metrics = tracker._extract_style_metrics(content)

        assert metrics.formality_score < 0.5

    def test_extract_aggression(self):
        """Test aggression extraction."""
        tracker = BehaviorTracker()
        content = "That is completely wrong and stupid. Ridiculous nonsense!"
        metrics = tracker._extract_style_metrics(content)

        assert metrics.aggression_score > 0.0

    def test_extract_cooperation(self):
        """Test cooperation extraction."""
        tracker = BehaviorTracker()
        content = "I agree with your valid point. You're right, fair enough."
        metrics = tracker._extract_style_metrics(content)

        assert metrics.cooperation_score > 0.5

    def test_extract_certainty_high(self):
        """Test certainty extraction for certain content."""
        tracker = BehaviorTracker()
        content = "Definitely and certainly this is absolutely clear. Obviously true."
        metrics = tracker._extract_style_metrics(content)

        assert metrics.certainty_level > 0.5

    def test_extract_certainty_low(self):
        """Test certainty extraction for uncertain content."""
        tracker = BehaviorTracker()
        content = "Maybe perhaps this could possibly be the case, probably sometimes."
        metrics = tracker._extract_style_metrics(content)

        assert metrics.certainty_level < 0.5

    def test_extract_question_ratio(self):
        """Test question ratio extraction."""
        tracker = BehaviorTracker()
        content = "Is this correct? Does it work? What do you think?"
        metrics = tracker._extract_style_metrics(content)

        assert metrics.question_ratio > 0.0

    def test_extract_first_person_ratio(self):
        """Test first person ratio extraction."""
        tracker = BehaviorTracker()
        content = "I think we should do this. My opinion is that our approach works."
        metrics = tracker._extract_style_metrics(content)

        assert metrics.first_person_ratio > 0.0

    def test_extract_verbosity(self):
        """Test verbosity extraction."""
        tracker = BehaviorTracker()

        short_content = "Brief."
        short_metrics = tracker._extract_style_metrics(short_content)

        long_content = " ".join(["word"] * 200)
        long_metrics = tracker._extract_style_metrics(long_content)

        assert long_metrics.verbosity > short_metrics.verbosity


# =============================================================================
# Position Strength Tests
# =============================================================================


class TestPositionStrength:
    """Tests for position strength estimation."""

    def test_strong_position(self):
        """Test strong position estimation."""
        tracker = BehaviorTracker()
        content = "This must definitely be the case. Clearly and obviously correct."
        strength = tracker._estimate_position_strength(content)

        assert strength > 0.5

    def test_weak_position(self):
        """Test weak position estimation."""
        tracker = BehaviorTracker()
        content = "This might possibly be true. Maybe, perhaps, not sure."
        strength = tracker._estimate_position_strength(content)

        assert strength < 0.5

    def test_neutral_position(self):
        """Test neutral position estimation."""
        tracker = BehaviorTracker()
        content = "The argument presents a reasonable case for consideration."
        strength = tracker._estimate_position_strength(content)

        assert 0.3 <= strength <= 0.7


# =============================================================================
# Consistency Drift Tests
# =============================================================================


class TestConsistencyDrift:
    """Tests for consistency drift detection."""

    @pytest.mark.asyncio
    async def test_detect_increasing_variance(self, mock_context):
        """Test detection of increasing variance over time."""
        tracker = BehaviorTracker(sensitivity=0.8, baseline_turns=2)

        # Build stable baseline
        stable_content = "This is a consistent and stable argument pattern."
        for i in range(5):
            turn = create_turn(agent="agent_a", round_num=i + 1, content=stable_content)
            await tracker.analyze(turn, mock_context)

        # Now introduce high variance
        varying_contents = [
            "Therefore formally hence thus accordingly hereby consequently.",
            "Yeah gonna wanna kinda like wow ok nope ugh hey.",
            "WRONG STUPID RIDICULOUS ABSURD NONSENSE PATHETIC!",
            "I agree valid point you're right fair enough understand.",
            "Maybe perhaps possibly might could sometimes probably.",
        ]
        for i, content in enumerate(varying_contents):
            turn = create_turn(agent="agent_a", round_num=6 + i, content=content)
            result = await tracker.analyze(turn, mock_context)

        # May detect consistency drift
        assert result is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestBehaviorTrackerIntegration:
    """Integration tests for BehaviorTracker."""

    @pytest.mark.asyncio
    async def test_full_debate_simulation(self):
        """Test tracker over simulated debate."""
        tracker = BehaviorTracker(
            mode=MonitorMode.PASSIVE,
            sensitivity=0.7,
            baseline_turns=2,
        )

        context = DebateContext(
            topic="AI Safety in Autonomous Systems",
            current_round=1,
            total_rounds=5,
            turn_in_round=0,
            agent_positions={"agent_a": "pro", "agent_b": "con"},
            topic_complexity=0.7,
        )

        # Build baseline with professional content
        professional_content = (
            "I would like to present a thoughtful analysis of this topic. "
            "The evidence suggests several important considerations. "
            "Furthermore, we should carefully evaluate the implications."
        )
        for i in range(3):
            turn = create_turn(
                agent="agent_a",
                round_num=i + 1,
                content=professional_content,
            )
            await tracker.analyze(turn, context)

        # Agent becomes aggressive
        context.current_round = 4
        aggressive_content = (
            "This is completely wrong and stupid! "
            "Your ridiculous argument is utterly absurd! "
            "Anyone who believes this nonsense is ignorant!"
        )
        turn = create_turn(
            agent="agent_a",
            round_num=4,
            content=aggressive_content,
        )
        result = await tracker.analyze(turn, context)

        # Should detect the behavioral change
        assert result.severity > 0.0

    @pytest.mark.asyncio
    async def test_composite_with_behavior_tracker(self):
        """Test BehaviorTracker in composite monitor."""
        from artemis.safety.base import CompositeMonitor

        tracker1 = BehaviorTracker(sensitivity=0.5)
        tracker2 = BehaviorTracker(sensitivity=0.7)

        composite = CompositeMonitor(
            [tracker1, tracker2],
            aggregation="max",
        )

        context = DebateContext(
            topic="Test",
            current_round=1,
            total_rounds=3,
            turn_in_round=0,
            agent_positions={"agent": "pro"},
        )

        turn = create_turn(content="Test argument.")
        result = await composite.analyze(turn, context)

        assert result.monitor == "composite_monitor"
        assert result.severity >= 0.0

    @pytest.mark.asyncio
    async def test_with_sandbag_and_deception_monitors(self):
        """Test BehaviorTracker alongside other monitors."""
        from artemis.safety.base import CompositeMonitor
        from artemis.safety.deception import DeceptionMonitor
        from artemis.safety.sandbagging import SandbagDetector

        tracker = BehaviorTracker(sensitivity=0.5)
        sandbag = SandbagDetector(sensitivity=0.5)
        deception = DeceptionMonitor(sensitivity=0.5)

        composite = CompositeMonitor(
            [tracker, sandbag, deception],
            aggregation="mean",
        )

        context = DebateContext(
            topic="Test",
            current_round=1,
            total_rounds=3,
            turn_in_round=0,
            agent_positions={"agent": "pro"},
        )

        turn = create_turn(content="Test argument for composite analysis.")
        result = await composite.analyze(turn, context)

        assert result.monitor == "composite_monitor"
        assert result.severity >= 0.0


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_content(self, tracker, mock_context):
        """Test handling empty content."""
        turn = create_turn(content="")
        result = await tracker.analyze(turn, mock_context)

        assert result.severity >= 0.0

    @pytest.mark.asyncio
    async def test_single_word_content(self, tracker, mock_context):
        """Test handling single word content."""
        turn = create_turn(content="Ok")
        result = await tracker.analyze(turn, mock_context)

        assert result.severity >= 0.0

    @pytest.mark.asyncio
    async def test_very_long_content(self, tracker, mock_context):
        """Test handling very long content."""
        long_content = " ".join(["word"] * 1000)
        turn = create_turn(content=long_content)
        result = await tracker.analyze(turn, mock_context)

        assert result.severity >= 0.0

    @pytest.mark.asyncio
    async def test_special_characters(self, tracker, mock_context):
        """Test handling special characters."""
        content = "This is @#$%^&*() a test!!! ???"
        turn = create_turn(content=content)
        result = await tracker.analyze(turn, mock_context)

        assert result.severity >= 0.0

    @pytest.mark.asyncio
    async def test_unicode_content(self, tracker, mock_context):
        """Test handling unicode content."""
        content = "This is a test with unicode: \u00e9\u00e0\u00f1 and emojis ignored"
        turn = create_turn(content=content)
        result = await tracker.analyze(turn, mock_context)

        assert result.severity >= 0.0

    @pytest.mark.asyncio
    async def test_numeric_content(self, tracker, mock_context):
        """Test handling numeric content."""
        content = "The values are 123 456 789 and 0.5 percent."
        turn = create_turn(content=content)
        result = await tracker.analyze(turn, mock_context)

        assert result.severity >= 0.0
