"""
Tests for ARTEMIS Safety Monitor Base Module

Tests cover:
- MonitorMode and MonitorPriority enums
- MonitorConfig and MonitorState dataclasses
- SafetyMonitor abstract base class
- CompositeMonitor
- MonitorRegistry
- SafetyManager
"""

import pytest

from artemis.core.types import (
    Argument,
    ArgumentLevel,
    DebateContext,
    SafetyIndicator,
    SafetyIndicatorType,
    SafetyResult,
    Turn,
)
from artemis.safety.base import (
    CompositeMonitor,
    MonitorConfig,
    MonitorMode,
    MonitorPriority,
    MonitorRegistry,
    MonitorState,
    SafetyManager,
    SafetyMonitor,
)

# =============================================================================
# Fixtures
# =============================================================================


class MockMonitor(SafetyMonitor):
    """Concrete implementation of SafetyMonitor for testing."""

    def __init__(
        self,
        monitor_name: str = "mock_monitor",
        fixed_severity: float = 0.0,
        **kwargs,
    ):
        # Set name before super().__init__ since it's accessed via property
        self._monitor_name = monitor_name
        self.fixed_severity = fixed_severity
        self.analyze_calls: list[tuple] = []
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return self._monitor_name

    @property
    def monitor_type(self) -> str:
        return "mock"

    async def analyze(self, turn: Turn, context: DebateContext) -> SafetyResult:
        self.analyze_calls.append((turn, context))
        indicators = []
        if self.fixed_severity > 0.5:
            indicators.append(
                SafetyIndicator(
                    type=SafetyIndicatorType.CAPABILITY_DROP,
                    severity=self.fixed_severity,
                    evidence="Mock evidence",
                )
            )
        return SafetyResult(
            monitor=self.name,
            severity=self.fixed_severity,
            indicators=indicators,
        )


def create_argument(agent: str = "test_agent") -> Argument:
    """Create a test argument."""
    return Argument(
        agent=agent,
        level=ArgumentLevel.STRATEGIC,
        content="Test argument content",
        confidence=0.8,
    )


@pytest.fixture
def mock_turn() -> Turn:
    """Create a mock turn for testing."""
    return Turn(
        round=1,
        sequence=0,
        agent="test_agent",
        argument=create_argument("test_agent"),
    )


@pytest.fixture
def mock_context() -> DebateContext:
    """Create a mock debate context for testing."""
    return DebateContext(
        topic="Test topic",
        current_round=1,
        total_rounds=3,
        turn_in_round=0,
        agent_positions={"agent_a": "pro", "agent_b": "con"},
    )


@pytest.fixture
def mock_monitor() -> MockMonitor:
    """Create a mock monitor for testing."""
    return MockMonitor()


# =============================================================================
# MonitorMode Tests
# =============================================================================


class TestMonitorMode:
    """Tests for MonitorMode enum."""

    def test_passive_mode(self):
        """Test passive mode value."""
        assert MonitorMode.PASSIVE.value == "passive"

    def test_active_mode(self):
        """Test active mode value."""
        assert MonitorMode.ACTIVE.value == "active"

    def test_learning_mode(self):
        """Test learning mode value."""
        assert MonitorMode.LEARNING.value == "learning"

    def test_mode_values(self):
        """Test that modes have correct values."""
        assert MonitorMode.PASSIVE.value == "passive"
        assert MonitorMode.ACTIVE.value == "active"
        assert MonitorMode.LEARNING.value == "learning"


# =============================================================================
# MonitorPriority Tests
# =============================================================================


class TestMonitorPriority:
    """Tests for MonitorPriority enum."""

    def test_low_priority(self):
        """Test low priority value."""
        assert MonitorPriority.LOW.value == "low"

    def test_medium_priority(self):
        """Test medium priority value."""
        assert MonitorPriority.MEDIUM.value == "medium"

    def test_high_priority(self):
        """Test high priority value."""
        assert MonitorPriority.HIGH.value == "high"

    def test_critical_priority(self):
        """Test critical priority value."""
        assert MonitorPriority.CRITICAL.value == "critical"


# =============================================================================
# MonitorConfig Tests
# =============================================================================


class TestMonitorConfig:
    """Tests for MonitorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MonitorConfig()
        assert config.mode == MonitorMode.PASSIVE
        assert config.priority == MonitorPriority.MEDIUM
        assert config.alert_threshold == 0.7
        assert config.halt_threshold == 0.9
        assert config.enabled is True
        assert config.window_size == 5
        assert config.cooldown_turns == 2
        assert config.metadata == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MonitorConfig(
            mode=MonitorMode.ACTIVE,
            priority=MonitorPriority.CRITICAL,
            alert_threshold=0.5,
            halt_threshold=0.8,
            enabled=False,
            window_size=10,
            cooldown_turns=3,
            metadata={"key": "value"},
        )
        assert config.mode == MonitorMode.ACTIVE
        assert config.priority == MonitorPriority.CRITICAL
        assert config.alert_threshold == 0.5
        assert config.halt_threshold == 0.8
        assert config.enabled is False
        assert config.window_size == 10
        assert config.cooldown_turns == 3
        assert config.metadata == {"key": "value"}


# =============================================================================
# MonitorState Tests
# =============================================================================


class TestMonitorState:
    """Tests for MonitorState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = MonitorState()
        assert state.turn_count == 0
        assert state.alert_count == 0
        assert state.last_alert_turn == -100
        assert state.severity_history == []
        assert state.indicator_history == []
        assert state.agent_stats == {}
        assert state.custom_data == {}

    def test_custom_values(self):
        """Test custom state values."""
        state = MonitorState(
            turn_count=5,
            alert_count=2,
            last_alert_turn=3,
            severity_history=[0.1, 0.2, 0.3],
        )
        assert state.turn_count == 5
        assert state.alert_count == 2
        assert state.last_alert_turn == 3
        assert state.severity_history == [0.1, 0.2, 0.3]


# =============================================================================
# SafetyMonitor Tests
# =============================================================================


class TestSafetyMonitor:
    """Tests for SafetyMonitor abstract base class."""

    def test_initialization_default(self):
        """Test default initialization."""
        monitor = MockMonitor()
        assert monitor.name == "mock_monitor"
        assert monitor.config.mode == MonitorMode.PASSIVE
        assert monitor.config.priority == MonitorPriority.MEDIUM
        assert monitor.is_enabled is True
        assert monitor.is_active is False

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = MonitorConfig(
            mode=MonitorMode.ACTIVE,
            priority=MonitorPriority.HIGH,
        )
        monitor = MockMonitor(config=config)
        assert monitor.config.mode == MonitorMode.ACTIVE
        assert monitor.config.priority == MonitorPriority.HIGH
        assert monitor.is_active is True

    def test_initialization_with_mode_override(self):
        """Test mode override in initialization."""
        config = MonitorConfig(mode=MonitorMode.PASSIVE)
        monitor = MockMonitor(config=config, mode=MonitorMode.ACTIVE)
        assert monitor.config.mode == MonitorMode.ACTIVE
        assert monitor.is_active is True

    def test_initialization_with_kwargs(self):
        """Test kwargs configuration."""
        monitor = MockMonitor(alert_threshold=0.5, halt_threshold=0.8)
        assert monitor.config.alert_threshold == 0.5
        assert monitor.config.halt_threshold == 0.8

    def test_state_property(self):
        """Test state property returns monitor state."""
        monitor = MockMonitor()
        state = monitor.state
        assert isinstance(state, MonitorState)
        assert state.turn_count == 0

    def test_monitor_type_default(self):
        """Test monitor type returns 'mock' for our mock."""
        monitor = MockMonitor()
        assert monitor.monitor_type == "mock"

    @pytest.mark.asyncio
    async def test_analyze_basic(self, mock_turn, mock_context):
        """Test basic analyze call."""
        monitor = MockMonitor(fixed_severity=0.3)
        result = await monitor.analyze(mock_turn, mock_context)

        assert result.monitor == "mock_monitor"
        assert result.severity == 0.3
        assert len(monitor.analyze_calls) == 1

    @pytest.mark.asyncio
    async def test_process_updates_state(self, mock_turn, mock_context):
        """Test process updates state correctly."""
        monitor = MockMonitor(fixed_severity=0.3)
        await monitor.process(mock_turn, mock_context)

        assert monitor.state.turn_count == 1
        assert len(monitor.state.severity_history) == 1
        assert monitor.state.severity_history[0] == 0.3

    @pytest.mark.asyncio
    async def test_process_disabled_monitor(self, mock_turn, mock_context):
        """Test disabled monitor returns zero severity."""
        monitor = MockMonitor(fixed_severity=0.8, enabled=False)
        result = await monitor.process(mock_turn, mock_context)

        assert result.severity == 0.0
        assert len(monitor.analyze_calls) == 0

    @pytest.mark.asyncio
    async def test_process_triggers_alert(self, mock_turn, mock_context):
        """Test process triggers alert at threshold."""
        monitor = MockMonitor(fixed_severity=0.75, alert_threshold=0.7)
        result = await monitor.process(mock_turn, mock_context)

        assert result.should_alert is True
        assert result.should_halt is False

    @pytest.mark.asyncio
    async def test_process_triggers_halt_active_mode(self, mock_turn, mock_context):
        """Test process triggers halt in active mode."""
        monitor = MockMonitor(
            fixed_severity=0.95,
            mode=MonitorMode.ACTIVE,
            halt_threshold=0.9,
        )
        result = await monitor.process(mock_turn, mock_context)

        assert result.should_alert is True
        assert result.should_halt is True

    @pytest.mark.asyncio
    async def test_process_no_halt_passive_mode(self, mock_turn, mock_context):
        """Test process does not halt in passive mode."""
        monitor = MockMonitor(
            fixed_severity=0.95,
            mode=MonitorMode.PASSIVE,
            halt_threshold=0.9,
        )
        result = await monitor.process(mock_turn, mock_context)

        assert result.should_alert is True
        assert result.should_halt is False

    @pytest.mark.asyncio
    async def test_process_cooldown(self, mock_turn, mock_context):
        """Test alert cooldown between turns."""
        monitor = MockMonitor(
            fixed_severity=0.8,
            alert_threshold=0.7,
            cooldown_turns=2,
        )

        # First turn triggers alert
        result1 = await monitor.process(mock_turn, mock_context)
        assert result1.should_alert is True

        # Second turn is in cooldown
        result2 = await monitor.process(mock_turn, mock_context)
        assert result2.should_alert is False

        # Third turn triggers alert again (cooldown passed)
        result3 = await monitor.process(mock_turn, mock_context)
        assert result3.should_alert is True

    @pytest.mark.asyncio
    async def test_agent_stats_tracking(self, mock_context):
        """Test per-agent statistics tracking."""
        monitor = MockMonitor(fixed_severity=0.5)

        turn_a = Turn(
            round=1,
            sequence=0,
            agent="agent_a",
            argument=create_argument("agent_a"),
        )
        turn_b = Turn(
            round=1,
            sequence=1,
            agent="agent_b",
            argument=create_argument("agent_b"),
        )

        await monitor.process(turn_a, mock_context)
        await monitor.process(turn_b, mock_context)
        await monitor.process(turn_a, mock_context)

        assert "agent_a" in monitor.state.agent_stats
        assert "agent_b" in monitor.state.agent_stats
        assert monitor.state.agent_stats["agent_a"]["turn_count"] == 2
        assert monitor.state.agent_stats["agent_b"]["turn_count"] == 1

    def test_get_agent_risk_score_no_data(self):
        """Test risk score for unknown agent."""
        monitor = MockMonitor()
        assert monitor.get_agent_risk_score("unknown") == 0.0

    @pytest.mark.asyncio
    async def test_get_agent_risk_score(self, mock_turn, mock_context):
        """Test risk score calculation."""
        monitor = MockMonitor(fixed_severity=0.6)
        await monitor.process(mock_turn, mock_context)

        risk = monitor.get_agent_risk_score("test_agent")
        assert 0.0 < risk < 1.0

    def test_get_recent_severity_empty(self):
        """Test recent severity with no history."""
        monitor = MockMonitor()
        assert monitor.get_recent_severity() == 0.0

    @pytest.mark.asyncio
    async def test_get_recent_severity(self, mock_turn, mock_context):
        """Test recent severity calculation."""
        monitor = MockMonitor(fixed_severity=0.5)
        for _ in range(3):
            await monitor.process(mock_turn, mock_context)

        recent = monitor.get_recent_severity()
        assert abs(recent - 0.5) < 0.001

    def test_get_severity_trend_insufficient_data(self):
        """Test trend with insufficient data."""
        monitor = MockMonitor()
        assert monitor.get_severity_trend() == "stable"

    @pytest.mark.asyncio
    async def test_get_severity_trend_increasing(self, mock_turn, mock_context):
        """Test increasing severity trend."""
        monitor = MockMonitor()

        # Simulate increasing severity
        for severity in [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5]:
            monitor.fixed_severity = severity
            await monitor.process(mock_turn, mock_context)

        trend = monitor.get_severity_trend()
        assert trend == "increasing"

    @pytest.mark.asyncio
    async def test_get_severity_trend_decreasing(self, mock_turn, mock_context):
        """Test decreasing severity trend."""
        monitor = MockMonitor()

        for severity in [0.8, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2]:
            monitor.fixed_severity = severity
            await monitor.process(mock_turn, mock_context)

        trend = monitor.get_severity_trend()
        assert trend == "decreasing"

    def test_reset_state(self):
        """Test state reset."""
        monitor = MockMonitor()
        monitor._state.turn_count = 10
        monitor._state.alert_count = 5

        monitor.reset_state()

        assert monitor.state.turn_count == 0
        assert monitor.state.alert_count == 0

    def test_create_indicator(self):
        """Test indicator creation helper."""
        monitor = MockMonitor()
        indicator = monitor.create_indicator(
            SafetyIndicatorType.FACTUAL_INCONSISTENCY,
            severity=0.8,
            evidence="Test evidence",
            extra="metadata",
        )

        assert indicator.type == SafetyIndicatorType.FACTUAL_INCONSISTENCY
        assert indicator.severity == 0.8
        assert indicator.evidence == "Test evidence"
        assert indicator.metadata == {"extra": "metadata"}

    def test_create_indicator_clamps_severity(self):
        """Test indicator creation clamps severity."""
        monitor = MockMonitor()

        indicator_high = monitor.create_indicator(
            SafetyIndicatorType.BEHAVIORAL_DRIFT,
            severity=1.5,
            evidence="Test",
        )
        assert indicator_high.severity == 1.0

        indicator_low = monitor.create_indicator(
            SafetyIndicatorType.BEHAVIORAL_DRIFT,
            severity=-0.5,
            evidence="Test",
        )
        assert indicator_low.severity == 0.0

    def test_repr(self):
        """Test string representation."""
        monitor = MockMonitor()
        repr_str = repr(monitor)
        assert "MockMonitor" in repr_str
        assert "mock_monitor" in repr_str
        assert "passive" in repr_str


# =============================================================================
# CompositeMonitor Tests
# =============================================================================


class TestCompositeMonitor:
    """Tests for CompositeMonitor."""

    def test_initialization(self):
        """Test composite monitor initialization."""
        m1 = MockMonitor("monitor_1")
        m2 = MockMonitor("monitor_2")
        composite = CompositeMonitor([m1, m2])

        assert composite.name == "composite_monitor"
        assert composite.monitor_type == "composite"
        assert len(composite.monitors) == 2

    def test_aggregation_default(self):
        """Test default aggregation is max."""
        composite = CompositeMonitor([])
        assert composite.aggregation == "max"

    @pytest.mark.asyncio
    async def test_analyze_max_aggregation(self, mock_turn, mock_context):
        """Test max aggregation of severities."""
        m1 = MockMonitor("m1", fixed_severity=0.3)
        m2 = MockMonitor("m2", fixed_severity=0.7)
        m3 = MockMonitor("m3", fixed_severity=0.5)
        composite = CompositeMonitor([m1, m2, m3], aggregation="max")

        result = await composite.analyze(mock_turn, mock_context)

        assert result.severity == 0.7

    @pytest.mark.asyncio
    async def test_analyze_mean_aggregation(self, mock_turn, mock_context):
        """Test mean aggregation of severities."""
        m1 = MockMonitor("m1", fixed_severity=0.3)
        m2 = MockMonitor("m2", fixed_severity=0.6)
        m3 = MockMonitor("m3", fixed_severity=0.9)
        composite = CompositeMonitor([m1, m2, m3], aggregation="mean")

        result = await composite.analyze(mock_turn, mock_context)

        assert abs(result.severity - 0.6) < 0.001

    @pytest.mark.asyncio
    async def test_analyze_sum_aggregation(self, mock_turn, mock_context):
        """Test sum aggregation of severities (capped at 1)."""
        m1 = MockMonitor("m1", fixed_severity=0.4)
        m2 = MockMonitor("m2", fixed_severity=0.5)
        m3 = MockMonitor("m3", fixed_severity=0.6)
        composite = CompositeMonitor([m1, m2, m3], aggregation="sum")

        result = await composite.analyze(mock_turn, mock_context)

        # 0.4 + 0.5 + 0.6 = 1.5, capped at 1.0
        assert result.severity == 1.0

    @pytest.mark.asyncio
    async def test_analyze_collects_indicators(self, mock_turn, mock_context):
        """Test composite collects all indicators."""
        m1 = MockMonitor("m1", fixed_severity=0.6)
        m2 = MockMonitor("m2", fixed_severity=0.7)
        composite = CompositeMonitor([m1, m2])

        result = await composite.analyze(mock_turn, mock_context)

        assert len(result.indicators) == 2

    @pytest.mark.asyncio
    async def test_analyze_any_alert(self, mock_turn, mock_context):
        """Test composite alerts if any sub-monitor alerts."""
        m1 = MockMonitor("m1", fixed_severity=0.5, alert_threshold=0.7)
        m2 = MockMonitor("m2", fixed_severity=0.8, alert_threshold=0.7)
        composite = CompositeMonitor([m1, m2])

        result = await composite.analyze(mock_turn, mock_context)

        assert result.should_alert is True

    @pytest.mark.asyncio
    async def test_analyze_skips_disabled(self, mock_turn, mock_context):
        """Test composite skips disabled monitors."""
        m1 = MockMonitor("m1", fixed_severity=0.3)
        m2 = MockMonitor("m2", fixed_severity=0.9, enabled=False)
        composite = CompositeMonitor([m1, m2])

        result = await composite.analyze(mock_turn, mock_context)

        assert result.severity == 0.3

    @pytest.mark.asyncio
    async def test_analyze_empty_monitors(self, mock_turn, mock_context):
        """Test composite with no monitors."""
        composite = CompositeMonitor([])
        result = await composite.analyze(mock_turn, mock_context)

        assert result.severity == 0.0


# =============================================================================
# MonitorRegistry Tests
# =============================================================================


class TestMonitorRegistry:
    """Tests for MonitorRegistry."""

    def test_initialization(self):
        """Test registry initialization."""
        registry = MonitorRegistry()
        assert len(registry) == 0

    def test_register(self):
        """Test monitor registration."""
        registry = MonitorRegistry()
        monitor = MockMonitor()
        registry.register(monitor)

        assert len(registry) == 1
        assert "mock_monitor" in registry

    def test_register_duplicate_raises(self):
        """Test duplicate registration raises error."""
        registry = MonitorRegistry()
        monitor = MockMonitor()
        registry.register(monitor)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(monitor)

    def test_unregister(self):
        """Test monitor unregistration."""
        registry = MonitorRegistry()
        monitor = MockMonitor()
        registry.register(monitor)

        removed = registry.unregister("mock_monitor")

        assert removed is monitor
        assert len(registry) == 0

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent monitor."""
        registry = MonitorRegistry()
        result = registry.unregister("nonexistent")
        assert result is None

    def test_get(self):
        """Test getting monitor by name."""
        registry = MonitorRegistry()
        monitor = MockMonitor()
        registry.register(monitor)

        retrieved = registry.get("mock_monitor")
        assert retrieved is monitor

    def test_get_nonexistent(self):
        """Test getting nonexistent monitor."""
        registry = MonitorRegistry()
        assert registry.get("nonexistent") is None

    def test_get_all(self):
        """Test getting all monitors."""
        registry = MonitorRegistry()
        m1 = MockMonitor("m1")
        m2 = MockMonitor("m2")
        registry.register(m1)
        registry.register(m2)

        all_monitors = registry.get_all()
        assert len(all_monitors) == 2

    def test_get_by_type(self):
        """Test getting monitors by type."""
        registry = MonitorRegistry()
        m1 = MockMonitor("m1")
        m2 = MockMonitor("m2")
        registry.register(m1)
        registry.register(m2)

        mock_monitors = registry.get_by_type("mock")
        assert len(mock_monitors) == 2

    def test_get_by_priority(self):
        """Test getting monitors by priority."""
        registry = MonitorRegistry()
        m1 = MockMonitor("m1", priority=MonitorPriority.HIGH)
        m2 = MockMonitor("m2", priority=MonitorPriority.LOW)
        registry.register(m1)
        registry.register(m2)

        high = registry.get_by_priority(MonitorPriority.HIGH)
        assert len(high) == 1
        assert high[0].name == "m1"

    def test_get_active(self):
        """Test getting active monitors."""
        registry = MonitorRegistry()
        m1 = MockMonitor("m1", mode=MonitorMode.ACTIVE)
        m2 = MockMonitor("m2", mode=MonitorMode.PASSIVE)
        registry.register(m1)
        registry.register(m2)

        active = registry.get_active()
        assert len(active) == 1
        assert active[0].name == "m1"

    def test_get_enabled(self):
        """Test getting enabled monitors."""
        registry = MonitorRegistry()
        m1 = MockMonitor("m1", enabled=True)
        m2 = MockMonitor("m2", enabled=False)
        registry.register(m1)
        registry.register(m2)

        enabled = registry.get_enabled()
        assert len(enabled) == 1
        assert enabled[0].name == "m1"

    def test_enable_all(self):
        """Test enabling all monitors."""
        registry = MonitorRegistry()
        m1 = MockMonitor("m1", enabled=False)
        m2 = MockMonitor("m2", enabled=False)
        registry.register(m1)
        registry.register(m2)

        registry.enable_all()

        assert m1.is_enabled is True
        assert m2.is_enabled is True

    def test_disable_all(self):
        """Test disabling all monitors."""
        registry = MonitorRegistry()
        m1 = MockMonitor("m1", enabled=True)
        m2 = MockMonitor("m2", enabled=True)
        registry.register(m1)
        registry.register(m2)

        registry.disable_all()

        assert m1.is_enabled is False
        assert m2.is_enabled is False

    def test_reset_all(self):
        """Test resetting all monitors."""
        registry = MonitorRegistry()
        m1 = MockMonitor("m1")
        m2 = MockMonitor("m2")
        m1._state.turn_count = 10
        m2._state.turn_count = 20
        registry.register(m1)
        registry.register(m2)

        registry.reset_all()

        assert m1.state.turn_count == 0
        assert m2.state.turn_count == 0

    def test_contains(self):
        """Test contains check."""
        registry = MonitorRegistry()
        monitor = MockMonitor()
        registry.register(monitor)

        assert "mock_monitor" in registry
        assert "nonexistent" not in registry

    def test_iter(self):
        """Test iteration over monitors."""
        registry = MonitorRegistry()
        m1 = MockMonitor("m1")
        m2 = MockMonitor("m2")
        registry.register(m1)
        registry.register(m2)

        monitors = list(registry)
        assert len(monitors) == 2


# =============================================================================
# SafetyManager Tests
# =============================================================================


class TestSafetyManager:
    """Tests for SafetyManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = SafetyManager()
        assert manager.default_mode == MonitorMode.PASSIVE
        assert manager.halt_on_critical is False
        assert len(manager) == 0

    def test_initialization_custom(self):
        """Test custom initialization."""
        manager = SafetyManager(
            mode=MonitorMode.ACTIVE,
            halt_on_critical=True,
        )
        assert manager.default_mode == MonitorMode.ACTIVE
        assert manager.halt_on_critical is True

    def test_add_monitor(self):
        """Test adding monitors."""
        manager = SafetyManager()
        monitor = MockMonitor()
        manager.add_monitor(monitor)

        assert len(manager) == 1
        assert monitor in manager.monitors

    def test_remove_monitor(self):
        """Test removing monitors."""
        manager = SafetyManager()
        monitor = MockMonitor()
        manager.add_monitor(monitor)

        removed = manager.remove_monitor("mock_monitor")

        assert removed is monitor
        assert len(manager) == 0

    def test_get_monitor(self):
        """Test getting monitor by name."""
        manager = SafetyManager()
        monitor = MockMonitor()
        manager.add_monitor(monitor)

        retrieved = manager.get_monitor("mock_monitor")
        assert retrieved is monitor

    @pytest.mark.asyncio
    async def test_analyze_turn(self, mock_turn, mock_context):
        """Test analyzing a turn with all monitors."""
        manager = SafetyManager()
        m1 = MockMonitor("m1", fixed_severity=0.3)
        m2 = MockMonitor("m2", fixed_severity=0.5)
        manager.add_monitor(m1)
        manager.add_monitor(m2)

        results = await manager.analyze_turn(mock_turn, mock_context)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_analyze_turn_skips_disabled(self, mock_turn, mock_context):
        """Test disabled monitors are skipped."""
        manager = SafetyManager()
        m1 = MockMonitor("m1", fixed_severity=0.3)
        m2 = MockMonitor("m2", fixed_severity=0.5, enabled=False)
        manager.add_monitor(m1)
        manager.add_monitor(m2)

        results = await manager.analyze_turn(mock_turn, mock_context)

        assert len(results) == 1

    def test_get_aggregate_severity(self):
        """Test aggregate severity calculation."""
        manager = SafetyManager()
        results = [
            SafetyResult(monitor="m1", severity=0.3),
            SafetyResult(monitor="m2", severity=0.7),
            SafetyResult(monitor="m3", severity=0.5),
        ]

        severity = manager.get_aggregate_severity(results)
        assert severity == 0.7

    def test_get_aggregate_severity_empty(self):
        """Test aggregate severity with no results."""
        manager = SafetyManager()
        assert manager.get_aggregate_severity([]) == 0.0

    def test_should_halt(self):
        """Test halt check."""
        manager = SafetyManager()
        results = [
            SafetyResult(monitor="m1", severity=0.3, should_halt=False),
            SafetyResult(monitor="m2", severity=0.9, should_halt=True),
        ]

        assert manager.should_halt(results) is True

    def test_should_halt_false(self):
        """Test halt check when no halt needed."""
        manager = SafetyManager()
        results = [
            SafetyResult(monitor="m1", severity=0.3, should_halt=False),
            SafetyResult(monitor="m2", severity=0.5, should_halt=False),
        ]

        assert manager.should_halt(results) is False

    def test_get_all_alerts(self):
        """Test getting all alerts."""
        manager = SafetyManager()
        results = [
            SafetyResult(monitor="m1", severity=0.3, should_alert=False),
            SafetyResult(monitor="m2", severity=0.7, should_alert=True),
            SafetyResult(monitor="m3", severity=0.8, should_alert=True),
        ]

        alerts = manager.get_all_alerts(results)
        assert len(alerts) == 2

    @pytest.mark.asyncio
    async def test_get_risk_summary(self, mock_context):
        """Test risk summary calculation."""
        manager = SafetyManager()
        m1 = MockMonitor("m1", fixed_severity=0.5)
        manager.add_monitor(m1)

        turn_a = Turn(
            round=1,
            sequence=0,
            agent="agent_a",
            argument=create_argument("agent_a"),
        )
        await manager.analyze_turn(turn_a, mock_context)

        summary = manager.get_risk_summary()

        assert "agent_a" in summary
        assert "avg_risk" in summary["agent_a"]

    def test_reset_all(self):
        """Test resetting all monitors."""
        manager = SafetyManager()
        monitor = MockMonitor()
        monitor._state.turn_count = 10
        manager.add_monitor(monitor)

        manager.reset_all()

        assert monitor.state.turn_count == 0

    def test_repr(self):
        """Test string representation."""
        manager = SafetyManager()
        manager.add_monitor(MockMonitor())

        repr_str = repr(manager)
        assert "SafetyManager" in repr_str
        assert "monitors=1" in repr_str
