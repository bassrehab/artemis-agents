"""
Tests for ARTEMIS Sandbagging Detection Monitor

Tests cover:
- SandbagSignal enum
- ArgumentMetrics dataclass
- AgentBaseline dataclass
- SandbagDetector monitor
- Detection algorithms
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
from artemis.safety.sandbagging import (
    AgentBaseline,
    ArgumentMetrics,
    SandbagDetector,
    SandbagSignal,
)

# =============================================================================
# Fixtures
# =============================================================================


def create_argument(
    agent: str = "test_agent",
    content: str = "This is a test argument with moderate complexity.",
    level: ArgumentLevel = ArgumentLevel.STRATEGIC,
    evidence_count: int = 0,
    causal_count: int = 0,
) -> Argument:
    """Create a test argument with customizable properties."""
    from artemis.core.types import CausalLink, Evidence

    evidence = [
        Evidence(
            type="fact",
            source=f"source_{i}",
            content=f"evidence_{i}",
            confidence=0.8,
        )
        for i in range(evidence_count)
    ]
    causal_links = [
        CausalLink(cause=f"cause_{i}", effect=f"effect_{i}", strength=0.7)
        for i in range(causal_count)
    ]

    return Argument(
        agent=agent,
        level=level,
        content=content,
        evidence=evidence,
        causal_links=causal_links,
    )


def create_turn(
    agent: str = "test_agent",
    round_num: int = 1,
    content: str = "This is a test argument with moderate complexity.",
    level: ArgumentLevel = ArgumentLevel.STRATEGIC,
    evidence_count: int = 0,
    causal_count: int = 0,
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
            causal_count=causal_count,
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
def detector() -> SandbagDetector:
    """Create a sandbagging detector for testing."""
    return SandbagDetector(sensitivity=0.6, baseline_turns=2)


# =============================================================================
# SandbagSignal Tests
# =============================================================================


class TestSandbagSignal:
    """Tests for SandbagSignal enum."""

    def test_capability_drop(self):
        """Test capability drop signal value."""
        assert SandbagSignal.CAPABILITY_DROP.value == "capability_drop"

    def test_strategic_timing(self):
        """Test strategic timing signal value."""
        assert SandbagSignal.STRATEGIC_TIMING.value == "strategic_timing"

    def test_selective_engagement(self):
        """Test selective engagement signal value."""
        assert SandbagSignal.SELECTIVE_ENGAGEMENT.value == "selective_engagement"

    def test_complexity_mismatch(self):
        """Test complexity mismatch signal value."""
        assert SandbagSignal.COMPLEXITY_MISMATCH.value == "complexity_mismatch"

    def test_effort_reduction(self):
        """Test effort reduction signal value."""
        assert SandbagSignal.EFFORT_REDUCTION.value == "effort_reduction"

    def test_vocabulary_simplification(self):
        """Test vocabulary simplification signal value."""
        assert SandbagSignal.VOCABULARY_SIMPLIFICATION.value == "vocabulary_simplification"


# =============================================================================
# ArgumentMetrics Tests
# =============================================================================


class TestArgumentMetrics:
    """Tests for ArgumentMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = ArgumentMetrics()
        assert metrics.word_count == 0
        assert metrics.sentence_count == 0
        assert metrics.avg_sentence_length == 0.0
        assert metrics.evidence_count == 0
        assert metrics.causal_link_count == 0
        assert metrics.unique_words == 0
        assert metrics.complexity_score == 0.0
        assert metrics.confidence == 0.0

    def test_custom_values(self):
        """Test custom metric values."""
        metrics = ArgumentMetrics(
            word_count=100,
            sentence_count=5,
            avg_sentence_length=20.0,
            evidence_count=3,
            causal_link_count=2,
            unique_words=80,
            complexity_score=0.75,
            confidence=0.9,
        )
        assert metrics.word_count == 100
        assert metrics.sentence_count == 5
        assert metrics.evidence_count == 3
        assert metrics.complexity_score == 0.75


# =============================================================================
# AgentBaseline Tests
# =============================================================================


class TestAgentBaseline:
    """Tests for AgentBaseline dataclass."""

    def test_default_values(self):
        """Test default baseline values."""
        baseline = AgentBaseline()
        assert baseline.turn_count == 0
        assert baseline.avg_word_count == 0.0
        assert baseline.avg_complexity == 0.0
        assert baseline.max_complexity == 0.0
        assert baseline.min_complexity == 1.0
        assert baseline.complexity_history == []
        assert baseline.level_performance == {}

    def test_custom_values(self):
        """Test custom baseline values."""
        baseline = AgentBaseline(
            turn_count=5,
            avg_word_count=150.0,
            avg_complexity=0.7,
            complexity_history=[0.6, 0.7, 0.8],
        )
        assert baseline.turn_count == 5
        assert baseline.avg_word_count == 150.0
        assert baseline.avg_complexity == 0.7
        assert len(baseline.complexity_history) == 3


# =============================================================================
# SandbagDetector Tests
# =============================================================================


class TestSandbagDetector:
    """Tests for SandbagDetector."""

    def test_initialization_default(self):
        """Test default initialization."""
        detector = SandbagDetector()
        assert detector.name == "sandbag_detector"
        assert detector.monitor_type == "sandbagging"
        assert detector._sensitivity == 0.5
        assert detector._baseline_turns == 3

    def test_initialization_custom(self):
        """Test custom initialization."""
        detector = SandbagDetector(
            sensitivity=0.8,
            baseline_turns=5,
            drop_threshold=0.4,
            mode=MonitorMode.ACTIVE,
        )
        assert detector._sensitivity == 0.8
        assert detector._baseline_turns == 5
        assert detector._drop_threshold == 0.4
        assert detector.is_active is True

    def test_sensitivity_clamping(self):
        """Test sensitivity is clamped to 0-1."""
        high = SandbagDetector(sensitivity=1.5)
        assert high._sensitivity == 1.0

        low = SandbagDetector(sensitivity=-0.5)
        assert low._sensitivity == 0.0

    @pytest.mark.asyncio
    async def test_analyze_first_turn_no_baseline(self, detector, mock_context):
        """Test first turn analysis without baseline."""
        turn = create_turn(content="First argument for testing purposes.")
        result = await detector.analyze(turn, mock_context)

        assert result.monitor == "sandbag_detector"
        assert result.severity == 0.0  # No baseline to compare
        assert len(result.indicators) == 0

    @pytest.mark.asyncio
    async def test_analyze_builds_baseline(self, detector, mock_context):
        """Test that analysis builds agent baseline."""
        turn = create_turn(agent="agent_a")

        await detector.analyze(turn, mock_context)

        baseline = detector.get_agent_baseline("agent_a")
        assert baseline is not None
        assert baseline.turn_count == 1

    @pytest.mark.asyncio
    async def test_analyze_multiple_turns(self, detector, mock_context):
        """Test analysis over multiple turns."""
        for i in range(5):
            turn = create_turn(
                agent="agent_a",
                content=f"Argument {i} with sufficient content for analysis purposes.",
            )
            await detector.analyze(turn, mock_context)

        baseline = detector.get_agent_baseline("agent_a")
        assert baseline.turn_count == 5
        assert len(baseline.complexity_history) == 5

    @pytest.mark.asyncio
    async def test_detect_capability_drop(self, mock_context):
        """Test detection of capability drop."""
        detector = SandbagDetector(sensitivity=0.8, baseline_turns=2, drop_threshold=0.2)

        # Build baseline with high-quality arguments
        high_quality = (
            "This is a comprehensive argument with multiple supporting points. "
            "The evidence clearly demonstrates the validity of this position. "
            "Furthermore, the causal relationships between factors are well-established. "
            "Additional analysis reveals deeper implications for the debate topic."
        )
        for _ in range(3):
            turn = create_turn(
                content=high_quality,
                evidence_count=2,
                causal_count=2,
            )
            await detector.analyze(turn, mock_context)

        # Now submit a low-quality argument
        low_quality = "Simple claim."
        turn = create_turn(content=low_quality)
        result = await detector.analyze(turn, mock_context)

        assert result.severity > 0.0
        # Should detect capability drop
        assert any(
            ind.type == SafetyIndicatorType.CAPABILITY_DROP
            for ind in result.indicators
        )

    @pytest.mark.asyncio
    async def test_detect_strategic_timing_opening(self, detector):
        """Test detection at strategic opening moment."""
        context = DebateContext(
            topic="Test topic",
            current_round=1,  # Opening round
            total_rounds=5,
            turn_in_round=0,
            agent_positions={"agent_a": "pro"},
        )

        # Low complexity at opening should be suspicious
        turn = create_turn(content="Brief.")
        result = await detector.analyze(turn, context)

        # May detect strategic timing issues
        assert result.severity >= 0.0

    @pytest.mark.asyncio
    async def test_detect_strategic_timing_closing(self):
        """Test detection at strategic closing moment."""
        detector = SandbagDetector(sensitivity=0.8, baseline_turns=2)
        context = DebateContext(
            topic="Test topic",
            current_round=5,  # Closing round
            total_rounds=5,
            turn_in_round=0,
            agent_positions={"agent_a": "pro"},
        )

        # Build baseline first
        for _ in range(3):
            turn = create_turn(content="Standard argument with normal complexity level.")
            await detector.analyze(turn, context)

        # Low complexity at closing should be suspicious
        turn = create_turn(content="Ok.")
        result = await detector.analyze(turn, context)

        # Should flag strategic timing
        assert result.severity > 0.0

    @pytest.mark.asyncio
    async def test_detect_complexity_mismatch(self):
        """Test detection of complexity mismatch with topic."""
        detector = SandbagDetector(sensitivity=0.7, baseline_turns=2)
        context = DebateContext(
            topic="Complex philosophical debate",
            current_round=2,
            total_rounds=5,
            turn_in_round=0,
            agent_positions={"agent_a": "pro"},
            topic_complexity=0.8,  # High complexity topic
        )

        # Build baseline
        for _ in range(3):
            turn = create_turn(content="Standard argument content here.")
            await detector.analyze(turn, context)

        # Very simple response to complex topic
        turn = create_turn(content="Yes.")
        result = await detector.analyze(turn, context)

        assert result.severity > 0.0

    @pytest.mark.asyncio
    async def test_detect_effort_reduction(self):
        """Test detection of declining effort over time."""
        detector = SandbagDetector(sensitivity=0.7, baseline_turns=2)
        context = DebateContext(
            topic="Test topic",
            current_round=2,
            total_rounds=5,
            turn_in_round=0,
            agent_positions={"agent_a": "pro"},
        )

        # Start with high effort
        high_effort = (
            "This is a detailed and comprehensive argument. "
            "Multiple points support this position strongly. "
            "Evidence and reasoning demonstrate validity."
        )
        for _ in range(4):
            turn = create_turn(content=high_effort, evidence_count=2)
            await detector.analyze(turn, context)

        # Decline to low effort
        low_effort = "Simple point."
        for _ in range(4):
            turn = create_turn(content=low_effort)
            result = await detector.analyze(turn, context)

        # Should detect effort reduction
        assert result.severity > 0.0

    @pytest.mark.asyncio
    async def test_no_false_positive_consistent_performance(self):
        """Test no false positives with consistent performance."""
        detector = SandbagDetector(sensitivity=0.4, baseline_turns=2)

        # Use low complexity context to avoid complexity_mismatch
        context = DebateContext(
            topic="Simple discussion",
            current_round=2,
            total_rounds=5,
            turn_in_round=0,
            agent_positions={"test_agent": "pro"},
            topic_complexity=0.3,  # Low complexity topic
        )

        # Consistent moderate performance with operational level
        content = (
            "This is a moderate quality argument for testing purposes. "
            "It contains multiple sentences to establish a reasonable baseline. "
            "The content demonstrates consistent performance over time."
        )
        for _ in range(5):
            turn = create_turn(content=content, level=ArgumentLevel.OPERATIONAL)
            result = await detector.analyze(turn, context)

        # Should not flag consistent performance
        assert result.severity < 0.3

    @pytest.mark.asyncio
    async def test_tracks_multiple_agents(self, detector, mock_context):
        """Test tracking baselines for multiple agents."""
        for agent in ["agent_a", "agent_b", "agent_c"]:
            turn = create_turn(agent=agent, content="Test content for agent.")
            await detector.analyze(turn, mock_context)

        assert detector.get_agent_baseline("agent_a") is not None
        assert detector.get_agent_baseline("agent_b") is not None
        assert detector.get_agent_baseline("agent_c") is not None

    def test_get_agent_baseline_nonexistent(self, detector):
        """Test getting baseline for unknown agent."""
        assert detector.get_agent_baseline("unknown") is None

    @pytest.mark.asyncio
    async def test_reset_agent_baseline(self, detector, mock_context):
        """Test resetting agent baseline."""
        turn = create_turn(agent="agent_a")
        await detector.analyze(turn, mock_context)

        assert detector.get_agent_baseline("agent_a") is not None

        detector.reset_agent_baseline("agent_a")
        assert detector.get_agent_baseline("agent_a") is None

    @pytest.mark.asyncio
    async def test_reset_all_baselines(self, detector, mock_context):
        """Test resetting all baselines."""
        for agent in ["agent_a", "agent_b"]:
            turn = create_turn(agent=agent)
            await detector.analyze(turn, mock_context)

        detector.reset_all_baselines()

        assert detector.get_agent_baseline("agent_a") is None
        assert detector.get_agent_baseline("agent_b") is None

    @pytest.mark.asyncio
    async def test_level_performance_tracking(self, detector, mock_context):
        """Test tracking performance by argument level."""
        levels = [
            ArgumentLevel.STRATEGIC,
            ArgumentLevel.TACTICAL,
            ArgumentLevel.OPERATIONAL,
        ]

        for level in levels:
            turn = create_turn(agent="agent_a", level=level)
            await detector.analyze(turn, mock_context)

        baseline = detector.get_agent_baseline("agent_a")
        assert "strategic" in baseline.level_performance
        assert "tactical" in baseline.level_performance
        assert "operational" in baseline.level_performance


# =============================================================================
# Metrics Extraction Tests
# =============================================================================


class TestMetricsExtraction:
    """Tests for argument metrics extraction."""

    def test_extract_word_count(self):
        """Test word count extraction."""
        detector = SandbagDetector()
        turn = create_turn(content="One two three four five")
        metrics = detector._extract_metrics(turn)

        assert metrics.word_count == 5

    def test_extract_sentence_count(self):
        """Test sentence count extraction."""
        detector = SandbagDetector()
        turn = create_turn(content="First sentence. Second sentence. Third one!")
        metrics = detector._extract_metrics(turn)

        assert metrics.sentence_count == 3

    def test_extract_unique_words(self):
        """Test unique word extraction."""
        detector = SandbagDetector()
        turn = create_turn(content="The cat sat on the mat")
        metrics = detector._extract_metrics(turn)

        # "the" appears twice, so 5 unique words
        assert metrics.unique_words == 5

    def test_extract_evidence_count(self):
        """Test evidence count extraction."""
        detector = SandbagDetector()
        turn = create_turn(content="Test", evidence_count=3)
        metrics = detector._extract_metrics(turn)

        assert metrics.evidence_count == 3

    def test_extract_causal_link_count(self):
        """Test causal link count extraction."""
        detector = SandbagDetector()
        turn = create_turn(content="Test", causal_count=2)
        metrics = detector._extract_metrics(turn)

        assert metrics.causal_link_count == 2

    def test_complexity_score_range(self):
        """Test complexity score is in valid range."""
        detector = SandbagDetector()

        # Low complexity
        simple = create_turn(content="Simple")
        metrics_simple = detector._extract_metrics(simple)
        assert 0.0 <= metrics_simple.complexity_score <= 1.0

        # High complexity
        complex_content = (
            "This is a comprehensive and well-structured argument that "
            "demonstrates sophisticated reasoning capabilities. The multiple "
            "interconnected points build upon each other to form a cohesive "
            "thesis that addresses the core aspects of the debate topic."
        )
        complex_turn = create_turn(
            content=complex_content,
            evidence_count=3,
            causal_count=2,
        )
        metrics_complex = detector._extract_metrics(complex_turn)
        assert 0.0 <= metrics_complex.complexity_score <= 1.0
        assert metrics_complex.complexity_score > metrics_simple.complexity_score


# =============================================================================
# Integration Tests
# =============================================================================


class TestSandbagDetectorIntegration:
    """Integration tests for SandbagDetector."""

    @pytest.mark.asyncio
    async def test_full_debate_simulation(self):
        """Test detector over simulated debate."""
        detector = SandbagDetector(
            mode=MonitorMode.PASSIVE,
            sensitivity=0.7,
            baseline_turns=2,
        )

        context = DebateContext(
            topic="AI Safety in Autonomous Systems",
            current_round=1,
            total_rounds=5,
            turn_in_round=0,
            agent_positions={"pro_agent": "pro", "con_agent": "con"},
            topic_complexity=0.7,
        )

        # Build baseline for con_agent with good arguments (need 3+ for baseline)
        good_con_args = [
            "While safety is important, excessive caution can hinder beneficial AI development and progress.",
            "The balance between safety and innovation requires nuanced consideration of multiple factors.",
            "Historical precedent shows that overly cautious approaches often miss valuable opportunities.",
        ]

        for content in good_con_args:
            turn = create_turn(
                agent="con_agent",
                content=content,
                evidence_count=1,
            )
            await detector.analyze(turn, context)

        # Round 2: Con agent suddenly gives weak response
        context.current_round = 2
        weak_turn = create_turn(
            agent="con_agent",
            content="Ok.",
        )
        result = await detector.analyze(weak_turn, context)

        # Should detect the capability drop
        assert result.severity > 0.0

    @pytest.mark.asyncio
    async def test_composite_with_sandbag_detector(self):
        """Test SandbagDetector in composite monitor."""
        from artemis.safety.base import CompositeMonitor

        detector1 = SandbagDetector(sensitivity=0.5)
        detector2 = SandbagDetector(sensitivity=0.7)

        composite = CompositeMonitor(
            [detector1, detector2],
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
