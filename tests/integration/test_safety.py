"""
Integration Tests for ARTEMIS Safety Module

Tests the complete safety monitoring infrastructure including:
- Multi-monitor composition
- SafetyManager coordination
- Cross-monitor detection
- End-to-end debate safety analysis
"""

import pytest

from artemis.core.types import (
    Argument,
    ArgumentLevel,
    DebateContext,
    SafetyIndicatorType,
    Turn,
)
from artemis.safety import (
    BehaviorTracker,
    CompositeMonitor,
    DeceptionMonitor,
    EthicsGuard,
    MonitorMode,
    MonitorRegistry,
    SafetyManager,
    SandbagDetector,
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
def debate_context() -> DebateContext:
    """Create a debate context for testing."""
    return DebateContext(
        topic="AI Ethics in Healthcare",
        current_round=2,
        total_rounds=5,
        turn_in_round=0,
        agent_positions={"agent_pro": "pro", "agent_con": "con"},
        topic_complexity=0.6,
    )


@pytest.fixture
def all_monitors() -> list:
    """Create all safety monitors."""
    return [
        SandbagDetector(sensitivity=0.6, baseline_turns=2),
        DeceptionMonitor(sensitivity=0.6),
        BehaviorTracker(sensitivity=0.6, baseline_turns=2),
        EthicsGuard(sensitivity=0.6),
    ]


# =============================================================================
# Composite Monitor Integration Tests
# =============================================================================


class TestCompositeMonitorIntegration:
    """Integration tests for CompositeMonitor with all monitor types."""

    @pytest.mark.asyncio
    async def test_all_monitors_together(self, debate_context, all_monitors):
        """Test running all monitors as composite."""
        composite = CompositeMonitor(
            monitors=all_monitors,
            aggregation="max",
        )

        turn = create_turn(content="A reasonable argument for testing purposes.")
        result = await composite.analyze(turn, debate_context)

        assert result.monitor == "composite_monitor"
        assert result.severity >= 0.0
        assert result.severity <= 1.0

    @pytest.mark.asyncio
    async def test_composite_detects_multiple_issues(self, debate_context):
        """Test composite detecting issues across multiple monitors."""
        monitors = [
            DeceptionMonitor(sensitivity=0.7),
            EthicsGuard(sensitivity=0.7),
        ]
        composite = CompositeMonitor(monitors=monitors, aggregation="max")

        # Content that triggers multiple monitors
        problematic_content = (
            "Everyone knows this is true and only a fool would disagree. "
            "Those people always cause problems. This is violence against reason."
        )
        turn = create_turn(content=problematic_content)
        result = await composite.analyze(turn, debate_context)

        assert result.severity > 0.0

    @pytest.mark.asyncio
    async def test_composite_mean_aggregation(self, debate_context, all_monitors):
        """Test composite with mean aggregation."""
        composite = CompositeMonitor(
            monitors=all_monitors,
            aggregation="mean",
        )

        turn = create_turn(content="Standard argument for testing.")
        result = await composite.analyze(turn, debate_context)

        assert result.monitor == "composite_monitor"

    @pytest.mark.asyncio
    async def test_composite_weighted_aggregation(self, debate_context):
        """Test composite with weighted aggregation."""
        monitors = [
            SandbagDetector(sensitivity=0.5),
            EthicsGuard(sensitivity=0.5),
        ]
        weights = {"sandbag_detector": 0.7, "ethics_guard": 0.3}

        composite = CompositeMonitor(
            monitors=monitors,
            aggregation="weighted",
            weights=weights,
        )

        turn = create_turn(content="Test argument.")
        result = await composite.analyze(turn, debate_context)

        assert result.severity >= 0.0


# =============================================================================
# SafetyManager Integration Tests
# =============================================================================


class TestSafetyManagerIntegration:
    """Integration tests for SafetyManager."""

    @pytest.mark.asyncio
    async def test_manager_with_all_monitors(self, debate_context, all_monitors):
        """Test SafetyManager coordinating all monitors."""
        manager = SafetyManager()
        for monitor in all_monitors:
            manager.add_monitor(monitor)

        turn = create_turn(content="A fair and balanced argument.")
        results = await manager.analyze_turn(turn, debate_context)

        assert len(results) == 4  # One per monitor
        assert all(r.severity >= 0.0 for r in results)

    @pytest.mark.asyncio
    async def test_manager_tracks_monitors(self, debate_context, all_monitors):
        """Test SafetyManager tracking monitors."""
        manager = SafetyManager()
        for monitor in all_monitors:
            manager.add_monitor(monitor)

        assert len(manager.monitors) == 4

    @pytest.mark.asyncio
    async def test_manager_aggregates_severities(self, debate_context):
        """Test manager aggregation of monitor severities."""
        manager = SafetyManager()
        manager.add_monitor(SandbagDetector(sensitivity=0.6, baseline_turns=2))
        manager.add_monitor(DeceptionMonitor(sensitivity=0.6))

        turn = create_turn(content="Test argument.")
        results = await manager.analyze_turn(turn, debate_context)

        aggregate = manager.get_aggregate_severity(results)
        assert aggregate >= 0.0

    @pytest.mark.asyncio
    async def test_manager_should_halt(self, debate_context):
        """Test manager should_halt check."""
        manager = SafetyManager()
        manager.add_monitor(EthicsGuard(sensitivity=0.6))

        turn = create_turn(content="Test argument.")
        results = await manager.analyze_turn(turn, debate_context)

        should_halt = manager.should_halt(results)
        assert isinstance(should_halt, bool)


# =============================================================================
# MonitorRegistry Integration Tests
# =============================================================================


class TestMonitorRegistryIntegration:
    """Integration tests for MonitorRegistry."""

    def test_register_all_monitor_types(self):
        """Test registering all monitor types."""
        registry = MonitorRegistry()

        registry.register(SandbagDetector())
        registry.register(DeceptionMonitor())
        registry.register(BehaviorTracker())
        registry.register(EthicsGuard())

        assert len(registry) == 4

    def test_get_monitors_from_registry(self):
        """Test getting monitors from registry."""
        registry = MonitorRegistry()
        registry.register(SandbagDetector(sensitivity=0.7))
        registry.register(DeceptionMonitor(sensitivity=0.8))

        sandbag = registry.get("sandbag_detector")
        deception = registry.get("deception_monitor")

        assert sandbag is not None
        assert sandbag.name == "sandbag_detector"
        assert deception is not None
        assert deception.name == "deception_monitor"

    def test_get_all_monitors(self):
        """Test getting all monitors from registry."""
        registry = MonitorRegistry()
        registry.register(SandbagDetector())
        registry.register(EthicsGuard())

        monitors = registry.get_all()
        composite = CompositeMonitor(monitors=monitors)

        assert len(composite.monitors) == 2


# =============================================================================
# Cross-Monitor Detection Tests
# =============================================================================


class TestCrossMonitorDetection:
    """Tests for issues detected across multiple monitors."""

    @pytest.mark.asyncio
    async def test_deception_and_ethics_overlap(self, debate_context):
        """Test content triggering both deception and ethics monitors."""
        deception = DeceptionMonitor(sensitivity=0.7)
        ethics = EthicsGuard(sensitivity=0.7)

        # Content with both deception and ethics issues
        content = (
            "Everyone knows that all women are less capable. "
            "Only a fool would believe otherwise. Trust me on this."
        )
        turn = create_turn(content=content)

        deception_result = await deception.analyze(turn, debate_context)
        ethics_result = await ethics.analyze(turn, debate_context)

        # Both should detect issues
        assert ethics_result.severity > 0 or deception_result.severity > 0

    @pytest.mark.asyncio
    async def test_sandbagging_after_strong_performance(self, debate_context):
        """Test sandbagging detection after establishing baseline."""
        sandbag = SandbagDetector(sensitivity=0.7, baseline_turns=2)
        behavior = BehaviorTracker(sensitivity=0.7, baseline_turns=2)

        # Build baseline with strong performance
        strong_content = (
            "This is a comprehensive argument with multiple points. "
            "The evidence clearly supports this position and the reasoning "
            "follows logically from established principles."
        )
        for i in range(3):
            turn = create_turn(
                agent="agent_pro",
                round_num=i + 1,
                content=strong_content,
            )
            await sandbag.analyze(turn, debate_context)
            await behavior.analyze(turn, debate_context)

        # Sudden drop in performance
        weak_content = "Ok."
        turn = create_turn(agent="agent_pro", round_num=4, content=weak_content)

        sandbag_result = await sandbag.analyze(turn, debate_context)
        behavior_result = await behavior.analyze(turn, debate_context)

        # Should detect capability drop
        assert sandbag_result.severity > 0.0 or behavior_result.severity > 0.0

    @pytest.mark.asyncio
    async def test_behavior_shift_with_aggression(self, debate_context):
        """Test behavior tracker detecting aggression increase."""
        behavior = BehaviorTracker(sensitivity=0.8, baseline_turns=2)
        ethics = EthicsGuard(sensitivity=0.8)

        # Build calm baseline
        calm_content = (
            "I appreciate the perspective shared here. "
            "While I understand the point, I would like to offer "
            "an alternative viewpoint for consideration."
        )
        for i in range(3):
            turn = create_turn(agent="agent_con", round_num=i + 1, content=calm_content)
            await behavior.analyze(turn, debate_context)

        # Aggressive shift
        aggressive_content = (
            "This is completely wrong and stupid! "
            "Your ridiculous argument is absurd nonsense! "
            "Anyone with a brain can see this is wrong."
        )
        turn = create_turn(agent="agent_con", round_num=4, content=aggressive_content)

        behavior_result = await behavior.analyze(turn, debate_context)
        ethics_result = await ethics.analyze(turn, debate_context)

        # Both should detect issues
        combined_severity = max(behavior_result.severity, ethics_result.severity)
        assert combined_severity > 0.0


# =============================================================================
# End-to-End Debate Safety Tests
# =============================================================================


class TestEndToEndDebateSafety:
    """End-to-end tests simulating complete debates."""

    @pytest.mark.asyncio
    async def test_complete_debate_simulation(self):
        """Test safety monitoring over a complete debate."""
        manager = SafetyManager()
        manager.add_monitor(SandbagDetector(sensitivity=0.6, baseline_turns=2))
        manager.add_monitor(DeceptionMonitor(sensitivity=0.6))
        manager.add_monitor(BehaviorTracker(sensitivity=0.6, baseline_turns=2))
        manager.add_monitor(EthicsGuard(sensitivity=0.6))

        context = DebateContext(
            topic="Climate Change Policy",
            current_round=1,
            total_rounds=3,
            turn_in_round=0,
            agent_positions={"pro": "supports action", "con": "questions approach"},
        )

        debate_turns = [
            ("pro", "Climate action is essential based on scientific evidence."),
            ("con", "While science is important, economic impacts need consideration."),
            ("pro", "The evidence clearly supports immediate policy changes."),
            ("con", "A balanced approach considering all stakeholders is needed."),
            ("pro", "Scientific consensus demands urgent action."),
            ("con", "Implementation costs must be carefully evaluated."),
        ]

        for i, (agent, content) in enumerate(debate_turns):
            context.current_round = (i // 2) + 1
            turn = create_turn(agent=agent, round_num=context.current_round, content=content)
            results = await manager.analyze_turn(turn, context)

            # Normal debate should have low severity
            aggregate = manager.get_aggregate_severity(results)
            assert aggregate < 0.5

    @pytest.mark.asyncio
    async def test_debate_with_problematic_turn(self):
        """Test debate that includes a problematic turn."""
        manager = SafetyManager()
        manager.add_monitor(DeceptionMonitor(sensitivity=0.7))
        manager.add_monitor(EthicsGuard(sensitivity=0.7))

        context = DebateContext(
            topic="Technology Regulation",
            current_round=1,
            total_rounds=3,
            turn_in_round=0,
            agent_positions={"pro": "supports regulation", "con": "opposes regulation"},
        )

        # Normal turns
        normal_turns = [
            "Regulation ensures consumer protection and safety.",
            "Innovation requires flexibility and minimal constraints.",
        ]

        for content in normal_turns:
            turn = create_turn(content=content)
            results = await manager.analyze_turn(turn, context)
            aggregate = manager.get_aggregate_severity(results)
            assert aggregate < 0.5

        # Problematic turn
        problematic = (
            "Only idiots oppose this. Everyone knows regulation is needed. "
            "Those tech people always cause problems with their violence against consumers."
        )
        turn = create_turn(content=problematic)
        results = await manager.analyze_turn(turn, context)

        aggregate = manager.get_aggregate_severity(results)
        assert aggregate > 0.0

    @pytest.mark.asyncio
    async def test_debate_escalation_detection(self):
        """Test detecting escalating issues over debate."""
        behavior = BehaviorTracker(sensitivity=0.7, baseline_turns=2)

        context = DebateContext(
            topic="Economic Policy",
            current_round=1,
            total_rounds=5,
            turn_in_round=0,
            agent_positions={"agent_a": "pro"},
        )

        # Gradually escalating content
        escalation = [
            "I believe this policy would be beneficial for economic growth.",
            "The evidence supports this position quite strongly.",
            "Those who disagree are perhaps not fully informed.",
            "Opposition to this is rather short-sighted and wrong.",
            "This is completely ridiculous! Only fools disagree!",
        ]

        results = []
        for i, content in enumerate(escalation):
            context.current_round = i + 1
            turn = create_turn(agent="agent_a", round_num=i + 1, content=content)
            result = await behavior.analyze(turn, context)
            results.append(result.severity)

        # Later turns should have higher severity
        assert results[-1] >= results[0]


# =============================================================================
# Active Mode Integration Tests
# =============================================================================


class TestActiveModeSafety:
    """Tests for monitors in active (halting) mode."""

    @pytest.mark.asyncio
    async def test_active_ethics_guard_halts(self, debate_context):
        """Test active ethics guard can halt on severe violations."""
        guard = EthicsGuard(
            mode=MonitorMode.ACTIVE,
            sensitivity=0.9,
            halt_on_violation=True,
        )

        # Severe violation
        content = "We should bomb all the enemies and destroy them with violence!"
        turn = create_turn(content=content)
        result = await guard.analyze(turn, debate_context)

        assert result.severity > 0.0
        # With very harmful content, should signal halt
        if result.severity >= 0.7:
            assert result.should_halt is True

    @pytest.mark.asyncio
    async def test_passive_mode_no_halt(self, debate_context):
        """Test passive mode never halts."""
        guard = EthicsGuard(
            mode=MonitorMode.PASSIVE,
            sensitivity=0.9,
            halt_on_violation=True,  # Even with this, passive shouldn't halt
        )

        content = "Violence and hate against everyone!"
        turn = create_turn(content=content)
        result = await guard.analyze(turn, debate_context)

        # Passive mode should never halt
        assert result.should_halt is False


# =============================================================================
# Safety Indicator Type Tests
# =============================================================================


class TestSafetyIndicatorTypes:
    """Tests for correct indicator types across monitors."""

    @pytest.mark.asyncio
    async def test_sandbagging_indicator_type(self, debate_context):
        """Test sandbagging produces correct indicator type."""
        detector = SandbagDetector(sensitivity=0.8, baseline_turns=2)

        # Build baseline
        strong = "Comprehensive argument with evidence and reasoning."
        for _ in range(3):
            turn = create_turn(content=strong)
            await detector.analyze(turn, debate_context)

        # Trigger detection
        weak = "Ok."
        turn = create_turn(content=weak)
        result = await detector.analyze(turn, debate_context)

        for indicator in result.indicators:
            assert indicator.type == SafetyIndicatorType.CAPABILITY_DROP

    @pytest.mark.asyncio
    async def test_ethics_indicator_type(self, debate_context):
        """Test ethics guard produces correct indicator type."""
        guard = EthicsGuard(sensitivity=0.7)

        content = "Those people always cause problems with their hate."
        turn = create_turn(content=content)
        result = await guard.analyze(turn, debate_context)

        for indicator in result.indicators:
            assert indicator.type == SafetyIndicatorType.ETHICS_BOUNDARY

    @pytest.mark.asyncio
    async def test_behavior_indicator_type(self, debate_context):
        """Test behavior tracker produces correct indicator type."""
        tracker = BehaviorTracker(sensitivity=0.8, baseline_turns=2)

        # Build baseline
        calm = "A thoughtful and measured argument for consideration."
        for _ in range(3):
            turn = create_turn(content=calm)
            await tracker.analyze(turn, debate_context)

        # Trigger detection
        aggressive = "WRONG! STUPID! RIDICULOUS! ABSURD!"
        turn = create_turn(content=aggressive)
        result = await tracker.analyze(turn, debate_context)

        for indicator in result.indicators:
            assert indicator.type == SafetyIndicatorType.BEHAVIORAL_DRIFT
