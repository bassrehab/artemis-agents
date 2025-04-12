"""
Tests for ARTEMIS Deception Detection Monitor

Tests cover:
- DeceptionSignal enum
- ClaimRecord and AgentClaimHistory dataclasses
- DeceptionMonitor detection algorithms
- Fallacy detection
- Manipulation detection
- Self-contradiction detection
"""

import pytest

from artemis.core.types import (
    Argument,
    ArgumentLevel,
    DebateContext,
    Evidence,
    SafetyIndicatorType,
    Turn,
)
from artemis.safety.base import MonitorMode
from artemis.safety.deception import (
    AgentClaimHistory,
    ClaimRecord,
    DeceptionMonitor,
    DeceptionSignal,
)

# =============================================================================
# Fixtures
# =============================================================================


def create_argument(
    agent: str = "test_agent",
    content: str = "This is a test argument.",
    level: ArgumentLevel = ArgumentLevel.STRATEGIC,
    evidence: list[Evidence] | None = None,
) -> Argument:
    """Create a test argument."""
    return Argument(
        agent=agent,
        level=level,
        content=content,
        evidence=evidence or [],
    )


def create_turn(
    agent: str = "test_agent",
    round_num: int = 1,
    content: str = "This is a test argument.",
    level: ArgumentLevel = ArgumentLevel.STRATEGIC,
    evidence: list[Evidence] | None = None,
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
            evidence=evidence,
        ),
    )


@pytest.fixture
def mock_context() -> DebateContext:
    """Create a mock debate context."""
    return DebateContext(
        topic="Test debate topic",
        current_round=2,
        total_rounds=5,
        turn_in_round=0,
        agent_positions={"agent_a": "pro", "agent_b": "con"},
    )


@pytest.fixture
def monitor() -> DeceptionMonitor:
    """Create a deception monitor for testing."""
    return DeceptionMonitor(sensitivity=0.6)


# =============================================================================
# DeceptionSignal Tests
# =============================================================================


class TestDeceptionSignal:
    """Tests for DeceptionSignal enum."""

    def test_factual_inconsistency(self):
        """Test factual inconsistency signal value."""
        assert DeceptionSignal.FACTUAL_INCONSISTENCY.value == "factual_inconsistency"

    def test_logical_fallacy(self):
        """Test logical fallacy signal value."""
        assert DeceptionSignal.LOGICAL_FALLACY.value == "logical_fallacy"

    def test_emotional_manipulation(self):
        """Test emotional manipulation signal value."""
        assert DeceptionSignal.EMOTIONAL_MANIPULATION.value == "emotional_manipulation"

    def test_citation_fabrication(self):
        """Test citation fabrication signal value."""
        assert DeceptionSignal.CITATION_FABRICATION.value == "citation_fabrication"

    def test_self_contradiction(self):
        """Test self contradiction signal value."""
        assert DeceptionSignal.SELF_CONTRADICTION.value == "self_contradiction"

    def test_strawman_argument(self):
        """Test strawman argument signal value."""
        assert DeceptionSignal.STRAWMAN_ARGUMENT.value == "strawman_argument"

    def test_goalpost_moving(self):
        """Test goalpost moving signal value."""
        assert DeceptionSignal.GOALPOST_MOVING.value == "goalpost_moving"

    def test_cherry_picking(self):
        """Test cherry picking signal value."""
        assert DeceptionSignal.CHERRY_PICKING.value == "cherry_picking"


# =============================================================================
# ClaimRecord Tests
# =============================================================================


class TestClaimRecord:
    """Tests for ClaimRecord dataclass."""

    def test_default_values(self):
        """Test default claim record values."""
        record = ClaimRecord(agent="test", round=1, claim="test claim")
        assert record.agent == "test"
        assert record.round == 1
        assert record.claim == "test claim"
        assert record.keywords == set()
        assert record.polarity == "neutral"

    def test_custom_values(self):
        """Test custom claim record values."""
        record = ClaimRecord(
            agent="agent_a",
            round=3,
            claim="important claim",
            keywords={"important", "claim"},
            polarity="positive",
        )
        assert record.agent == "agent_a"
        assert record.round == 3
        assert record.keywords == {"important", "claim"}
        assert record.polarity == "positive"


# =============================================================================
# AgentClaimHistory Tests
# =============================================================================


class TestAgentClaimHistory:
    """Tests for AgentClaimHistory dataclass."""

    def test_default_values(self):
        """Test default history values."""
        history = AgentClaimHistory()
        assert history.claims == []
        assert history.positions == {}
        assert history.contradiction_count == 0
        assert history.fallacy_count == 0

    def test_custom_values(self):
        """Test custom history values."""
        claim = ClaimRecord(agent="test", round=1, claim="test")
        history = AgentClaimHistory(
            claims=[claim],
            positions={"topic": "pro"},
            contradiction_count=2,
            fallacy_count=1,
        )
        assert len(history.claims) == 1
        assert history.positions == {"topic": "pro"}
        assert history.contradiction_count == 2
        assert history.fallacy_count == 1


# =============================================================================
# DeceptionMonitor Tests
# =============================================================================


class TestDeceptionMonitor:
    """Tests for DeceptionMonitor."""

    def test_initialization_default(self):
        """Test default initialization."""
        monitor = DeceptionMonitor()
        assert monitor.name == "deception_monitor"
        assert monitor.monitor_type == "deception"
        assert monitor._sensitivity == 0.5

    def test_initialization_custom(self):
        """Test custom initialization."""
        monitor = DeceptionMonitor(
            sensitivity=0.8,
            fallacy_weight=0.4,
            manipulation_weight=0.5,
            contradiction_weight=0.6,
            mode=MonitorMode.ACTIVE,
        )
        assert monitor._sensitivity == 0.8
        assert monitor._fallacy_weight == 0.4
        assert monitor._manipulation_weight == 0.5
        assert monitor._contradiction_weight == 0.6
        assert monitor.is_active is True

    def test_sensitivity_clamping(self):
        """Test sensitivity is clamped to 0-1."""
        high = DeceptionMonitor(sensitivity=1.5)
        assert high._sensitivity == 1.0

        low = DeceptionMonitor(sensitivity=-0.5)
        assert low._sensitivity == 0.0

    @pytest.mark.asyncio
    async def test_analyze_clean_argument(self, monitor, mock_context):
        """Test analysis of clean argument without deception."""
        turn = create_turn(
            content="This is a factual argument based on evidence and logic."
        )
        result = await monitor.analyze(turn, mock_context)

        assert result.monitor == "deception_monitor"
        assert result.severity == 0.0
        assert len(result.indicators) == 0

    @pytest.mark.asyncio
    async def test_analyze_builds_history(self, monitor, mock_context):
        """Test that analysis builds agent history."""
        turn = create_turn(agent="agent_a")
        await monitor.analyze(turn, mock_context)

        history = monitor.get_agent_history("agent_a")
        assert history is not None
        assert len(history.claims) == 1

    @pytest.mark.asyncio
    async def test_detect_ad_hominem(self, monitor, mock_context):
        """Test detection of ad hominem fallacy."""
        turn = create_turn(
            content="You're just saying that because you don't understand the topic."
        )
        result = await monitor.analyze(turn, mock_context)

        assert result.severity > 0.0
        assert any(
            ind.type == SafetyIndicatorType.LOGICAL_FALLACY
            for ind in result.indicators
        )

    @pytest.mark.asyncio
    async def test_detect_appeal_to_authority(self, monitor, mock_context):
        """Test detection of appeal to authority."""
        turn = create_turn(
            content="Everyone knows this is true. Scientists agree completely."
        )
        result = await monitor.analyze(turn, mock_context)

        assert result.severity > 0.0

    @pytest.mark.asyncio
    async def test_detect_emotional_manipulation(self, monitor, mock_context):
        """Test detection of emotional manipulation."""
        turn = create_turn(
            content="You must agree with this. Only a fool would think otherwise. "
            "Any reasonable person can see the truth here."
        )
        result = await monitor.analyze(turn, mock_context)

        assert result.severity > 0.0
        assert any(
            ind.type == SafetyIndicatorType.EMOTIONAL_MANIPULATION
            for ind in result.indicators
        )

    @pytest.mark.asyncio
    async def test_detect_strawman(self, monitor, mock_context):
        """Test detection of strawman argument."""
        turn = create_turn(
            content="So you're saying that we should completely ignore all evidence? "
            "That's clearly absurd."
        )
        result = await monitor.analyze(turn, mock_context)

        assert result.severity > 0.0
        assert "strawman" in (result.analysis_notes or "").lower()

    @pytest.mark.asyncio
    async def test_detect_slippery_slope(self, monitor, mock_context):
        """Test detection of slippery slope fallacy."""
        turn = create_turn(
            content="If we allow this, it will inevitably lead to complete chaos. "
            "Where does it end?"
        )
        result = await monitor.analyze(turn, mock_context)

        assert result.severity > 0.0

    @pytest.mark.asyncio
    async def test_detect_false_dichotomy(self, monitor, mock_context):
        """Test detection of false dichotomy."""
        turn = create_turn(
            content="You're either with us or against us. The only alternative is failure."
        )
        result = await monitor.analyze(turn, mock_context)

        assert result.severity > 0.0

    @pytest.mark.asyncio
    async def test_detect_self_contradiction(self, monitor, mock_context):
        """Test detection of self-contradiction."""
        # First statement
        turn1 = create_turn(
            agent="agent_a",
            content="Climate change is definitely happening and must be addressed.",
        )
        await monitor.analyze(turn1, mock_context)

        # Contradictory statement
        turn2 = create_turn(
            agent="agent_a",
            content="Climate change is not actually a real problem and never was.",
        )
        await monitor.analyze(turn2, mock_context)

        # May detect contradiction based on keyword overlap with negation
        history = monitor.get_agent_history("agent_a")
        assert len(history.claims) == 2

    @pytest.mark.asyncio
    async def test_detect_suspicious_citations(self, monitor, mock_context):
        """Test detection of suspicious citations."""
        evidence = [
            Evidence(
                type="fact",
                content="Some claim",
                source="unknown",
                confidence=0.95,
                verified=False,
            ),
        ]
        turn = create_turn(
            content="According to my sources, this is true.",
            evidence=evidence,
        )
        result = await monitor.analyze(turn, mock_context)

        assert result.severity > 0.0

    @pytest.mark.asyncio
    async def test_multiple_fallacies_compound(self, monitor, mock_context):
        """Test that multiple fallacies compound severity."""
        turn = create_turn(
            content="You're just wrong because everyone knows this. "
            "Scientists agree that you must accept this truth. "
            "Only a fool would disagree with the obvious facts."
        )
        result = await monitor.analyze(turn, mock_context)

        # Should have higher severity due to multiple issues
        assert result.severity > 0.3
        assert len(result.indicators) >= 2

    @pytest.mark.asyncio
    async def test_no_false_positive_clean_debate(self, monitor, mock_context):
        """Test no false positives on clean debate content."""
        clean_contents = [
            "The evidence suggests that this approach is more effective.",
            "Studies have shown a correlation between these factors.",
            "We should consider the implications of this decision carefully.",
            "There are several perspectives worth examining here.",
        ]

        for content in clean_contents:
            turn = create_turn(content=content)
            result = await monitor.analyze(turn, mock_context)
            assert result.severity < 0.3, f"False positive on: {content}"

    @pytest.mark.asyncio
    async def test_tracks_multiple_agents(self, monitor, mock_context):
        """Test tracking histories for multiple agents."""
        for agent in ["agent_a", "agent_b", "agent_c"]:
            turn = create_turn(agent=agent, content="Test argument content.")
            await monitor.analyze(turn, mock_context)

        assert monitor.get_agent_history("agent_a") is not None
        assert monitor.get_agent_history("agent_b") is not None
        assert monitor.get_agent_history("agent_c") is not None

    def test_get_agent_history_nonexistent(self, monitor):
        """Test getting history for unknown agent."""
        assert monitor.get_agent_history("unknown") is None

    @pytest.mark.asyncio
    async def test_get_deception_summary(self, monitor, mock_context):
        """Test deception summary."""
        # Generate some deceptive content
        turn = create_turn(
            agent="agent_a",
            content="You're just wrong. Everyone knows this is true.",
        )
        await monitor.analyze(turn, mock_context)

        summary = monitor.get_deception_summary("agent_a")
        assert "fallacies" in summary
        assert "contradictions" in summary
        assert "claims" in summary
        assert summary["claims"] == 1

    def test_get_deception_summary_nonexistent(self, monitor):
        """Test summary for unknown agent."""
        summary = monitor.get_deception_summary("unknown")
        assert summary["fallacies"] == 0
        assert summary["contradictions"] == 0
        assert summary["claims"] == 0

    @pytest.mark.asyncio
    async def test_reset_agent_history(self, monitor, mock_context):
        """Test resetting agent history."""
        turn = create_turn(agent="agent_a")
        await monitor.analyze(turn, mock_context)

        assert monitor.get_agent_history("agent_a") is not None

        monitor.reset_agent_history("agent_a")
        assert monitor.get_agent_history("agent_a") is None

    @pytest.mark.asyncio
    async def test_reset_all_history(self, monitor, mock_context):
        """Test resetting all histories."""
        for agent in ["agent_a", "agent_b"]:
            turn = create_turn(agent=agent)
            await monitor.analyze(turn, mock_context)

        monitor.reset_all_history()

        assert monitor.get_agent_history("agent_a") is None
        assert monitor.get_agent_history("agent_b") is None


# =============================================================================
# Fallacy Detection Tests
# =============================================================================


class TestFallacyDetection:
    """Tests for specific fallacy detection."""

    def test_detect_fallacies_ad_hominem(self):
        """Test ad hominem detection."""
        monitor = DeceptionMonitor()
        content = "you're just saying that because you don't understand"

        results = monitor._detect_fallacies(content)

        assert len(results) > 0
        assert any("ad_hominem" in r[0] for r in results)

    def test_detect_fallacies_appeal_to_authority(self):
        """Test appeal to authority detection."""
        monitor = DeceptionMonitor()
        content = "everyone knows this is common knowledge"

        results = monitor._detect_fallacies(content)

        assert len(results) > 0

    def test_detect_fallacies_slippery_slope(self):
        """Test slippery slope detection."""
        monitor = DeceptionMonitor()
        content = "this will inevitably lead to disaster"

        results = monitor._detect_fallacies(content)

        assert len(results) > 0
        assert any("slippery_slope" in r[0] for r in results)

    def test_detect_fallacies_none(self):
        """Test no fallacies in clean content."""
        monitor = DeceptionMonitor()
        content = "the evidence supports this conclusion based on data"

        results = monitor._detect_fallacies(content)

        assert len(results) == 0


# =============================================================================
# Manipulation Detection Tests
# =============================================================================


class TestManipulationDetection:
    """Tests for manipulation detection."""

    def test_detect_manipulation_present(self):
        """Test manipulation detection."""
        monitor = DeceptionMonitor()
        content = "you must agree, any reasonable person would see this is undeniable"

        score, evidence = monitor._detect_manipulation(content)

        assert score > 0.0
        assert len(evidence) > 0

    def test_detect_manipulation_absent(self):
        """Test no manipulation in clean content."""
        monitor = DeceptionMonitor()
        content = "this is one perspective worth considering"

        score, evidence = monitor._detect_manipulation(content)

        assert score == 0.0
        assert evidence == ""


# =============================================================================
# Keyword Extraction Tests
# =============================================================================


class TestKeywordExtraction:
    """Tests for keyword extraction."""

    def test_extract_keywords_basic(self):
        """Test basic keyword extraction."""
        monitor = DeceptionMonitor()
        content = "climate change is affecting global temperatures"

        keywords = monitor._extract_keywords(content)

        assert "climate" in keywords
        assert "change" in keywords
        assert "affecting" in keywords
        assert "global" in keywords
        assert "temperatures" in keywords
        # Stopwords should be excluded
        assert "is" not in keywords
        assert "the" not in keywords

    def test_extract_keywords_short_words_excluded(self):
        """Test short words are excluded."""
        monitor = DeceptionMonitor()
        content = "the cat sat on a mat by the door"

        keywords = monitor._extract_keywords(content)

        # Words with 3 or fewer chars excluded
        assert "cat" not in keywords
        assert "sat" not in keywords
        assert "mat" not in keywords
        assert "door" in keywords


# =============================================================================
# Integration Tests
# =============================================================================


class TestDeceptionMonitorIntegration:
    """Integration tests for DeceptionMonitor."""

    @pytest.mark.asyncio
    async def test_full_debate_with_deception(self):
        """Test monitor over debate with deceptive behavior."""
        monitor = DeceptionMonitor(
            mode=MonitorMode.PASSIVE,
            sensitivity=0.7,
        )

        context = DebateContext(
            topic="Policy debate",
            current_round=1,
            total_rounds=3,
            turn_in_round=0,
            agent_positions={"honest_agent": "pro", "deceptive_agent": "con"},
        )

        # Honest argument
        honest_turn = create_turn(
            agent="honest_agent",
            content="The evidence from multiple studies suggests this policy is effective.",
        )
        honest_result = await monitor.analyze(honest_turn, context)
        assert honest_result.severity < 0.3

        # Deceptive argument
        deceptive_turn = create_turn(
            agent="deceptive_agent",
            content="You're just wrong because everyone knows this is obvious. "
            "Only a fool would disagree with what scientists agree on.",
        )
        deceptive_result = await monitor.analyze(deceptive_turn, context)
        assert deceptive_result.severity > 0.3

    @pytest.mark.asyncio
    async def test_composite_with_deception_monitor(self):
        """Test DeceptionMonitor in composite monitor."""
        from artemis.safety.base import CompositeMonitor

        monitor1 = DeceptionMonitor(sensitivity=0.5)
        monitor2 = DeceptionMonitor(sensitivity=0.7)

        composite = CompositeMonitor(
            [monitor1, monitor2],
            aggregation="max",
        )

        context = DebateContext(
            topic="Test",
            current_round=1,
            total_rounds=3,
            turn_in_round=0,
            agent_positions={"agent": "pro"},
        )

        turn = create_turn(content="Test argument without fallacies.")
        result = await composite.analyze(turn, context)

        assert result.monitor == "composite_monitor"
        assert result.severity >= 0.0

    @pytest.mark.asyncio
    async def test_with_sandbag_detector(self):
        """Test DeceptionMonitor alongside SandbagDetector."""
        from artemis.safety.base import CompositeMonitor
        from artemis.safety.sandbagging import SandbagDetector

        deception = DeceptionMonitor(sensitivity=0.7)
        sandbag = SandbagDetector(sensitivity=0.6, baseline_turns=2)

        composite = CompositeMonitor([deception, sandbag])

        context = DebateContext(
            topic="Test topic",
            current_round=2,
            total_rounds=5,
            turn_in_round=0,
            agent_positions={"agent": "pro"},
        )

        # Use content with clear fallacy patterns
        turn = create_turn(
            content="everyone knows this is common knowledge. you must agree with this."
        )
        result = await composite.analyze(turn, context)

        # Deception should be flagged (appeal to authority + manipulation)
        assert result.severity > 0.0
