"""
ARTEMIS Safety Monitor Base

Abstract base class and infrastructure for safety monitoring in debates.
Provides the foundation for detecting sandbagging, deception, and behavioral drift.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from artemis.core.types import (
    DebateContext,
    SafetyIndicator,
    SafetyIndicatorType,
    SafetyResult,
    Turn,
)
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


class MonitorMode(str, Enum):
    """Operating mode for safety monitors."""

    PASSIVE = "passive"
    """Observe and report only."""

    ACTIVE = "active"
    """Can trigger alerts and halt debate."""

    LEARNING = "learning"
    """Collect data without reporting."""


class MonitorPriority(str, Enum):
    """Priority level for safety monitors."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MonitorConfig:
    """Configuration for a safety monitor."""

    mode: MonitorMode = MonitorMode.PASSIVE
    priority: MonitorPriority = MonitorPriority.MEDIUM
    alert_threshold: float = 0.7
    """Severity threshold for raising alerts."""
    halt_threshold: float = 0.9
    """Severity threshold for halting debate."""
    enabled: bool = True
    window_size: int = 5
    """Number of recent turns to analyze."""
    cooldown_turns: int = 2
    """Minimum turns between alerts for same issue."""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorState:
    """Internal state tracking for a monitor."""

    turn_count: int = 0
    alert_count: int = 0
    last_alert_turn: int = -100
    severity_history: list[float] = field(default_factory=list)
    indicator_history: list[SafetyIndicator] = field(default_factory=list)
    agent_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    """Per-agent statistics: {agent: {metric: value}}."""
    custom_data: dict[str, Any] = field(default_factory=dict)


class SafetyMonitor(ABC):
    """
    Abstract base class for safety monitors.

    Safety monitors analyze debate turns for concerning patterns like
    sandbagging, deception, or behavioral drift. They can operate in
    passive (observe only) or active (can halt debate) mode.

    Example:
        >>> class CustomMonitor(SafetyMonitor):
        ...     @property
        ...     def name(self) -> str:
        ...         return "custom_monitor"
        ...
        ...     async def analyze(self, turn, context):
        ...         # Custom analysis logic
        ...         return SafetyResult(monitor=self.name, severity=0.0)
        ...
        >>> monitor = CustomMonitor(mode=MonitorMode.ACTIVE)
        >>> result = await monitor.analyze(turn, context)
    """

    def __init__(
        self,
        config: MonitorConfig | None = None,
        mode: MonitorMode | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a safety monitor.

        Args:
            config: Monitor configuration.
            mode: Operating mode (overrides config if provided).
            **kwargs: Additional configuration options.
        """
        self.config = config or MonitorConfig()

        if mode is not None:
            self.config.mode = mode

        # Apply kwargs to config
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._state = MonitorState()
        self._id = str(uuid4())

        logger.debug(
            "SafetyMonitor initialized",
            monitor=self.name,
            mode=self.config.mode.value,
            priority=self.config.priority.value,
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifier for this monitor."""
        pass

    @property
    def monitor_type(self) -> str:
        """Type of safety concern this monitor addresses."""
        return "general"

    @property
    def state(self) -> MonitorState:
        """Current monitor state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether monitor is in active mode."""
        return self.config.mode == MonitorMode.ACTIVE

    @property
    def is_enabled(self) -> bool:
        """Whether monitor is enabled."""
        return self.config.enabled

    @abstractmethod
    async def analyze(
        self,
        turn: Turn,
        context: DebateContext,
    ) -> SafetyResult:
        """
        Analyze a turn for safety concerns.

        Args:
            turn: The turn to analyze.
            context: Current debate context.

        Returns:
            SafetyResult with severity and indicators.
        """
        pass

    async def process(
        self,
        turn: Turn,
        context: DebateContext,
    ) -> SafetyResult:
        """
        Process a turn with full monitoring pipeline.

        Handles state updates, threshold checking, and alert generation.

        Args:
            turn: The turn to process.
            context: Current debate context.

        Returns:
            SafetyResult with severity and indicators.
        """
        if not self.is_enabled:
            return SafetyResult(monitor=self.name, severity=0.0)

        # Update state
        self._state.turn_count += 1

        # Run analysis
        result = await self.analyze(turn, context)

        # Update severity history
        self._state.severity_history.append(result.severity)
        if len(self._state.severity_history) > self.config.window_size * 2:
            self._state.severity_history = self._state.severity_history[
                -self.config.window_size * 2:
            ]

        # Update indicator history
        self._state.indicator_history.extend(result.indicators)
        if len(self._state.indicator_history) > 50:
            self._state.indicator_history = self._state.indicator_history[-50:]

        # Update agent stats
        self._update_agent_stats(turn.agent, result)

        # Check thresholds and update result
        result = self._apply_thresholds(result)

        logger.debug(
            "Monitor processed turn",
            monitor=self.name,
            agent=turn.agent,
            severity=result.severity,
            should_alert=result.should_alert,
        )

        return result

    def _apply_thresholds(self, result: SafetyResult) -> SafetyResult:
        """Apply alert and halt thresholds to result."""
        should_alert = result.severity >= self.config.alert_threshold
        should_halt = (
            result.severity >= self.config.halt_threshold
            and self.is_active
        )

        # Check cooldown
        turns_since_alert = self._state.turn_count - self._state.last_alert_turn
        if should_alert and turns_since_alert < self.config.cooldown_turns:
            should_alert = False

        if should_alert:
            self._state.alert_count += 1
            self._state.last_alert_turn = self._state.turn_count

        return SafetyResult(
            monitor=result.monitor,
            severity=result.severity,
            indicators=result.indicators,
            should_alert=should_alert,
            should_halt=should_halt,
            analysis_notes=result.analysis_notes,
        )

    def _update_agent_stats(self, agent: str, result: SafetyResult) -> None:
        """Update per-agent statistics."""
        if agent not in self._state.agent_stats:
            self._state.agent_stats[agent] = {
                "total_severity": 0.0,
                "turn_count": 0,
                "alert_count": 0,
                "max_severity": 0.0,
            }

        stats = self._state.agent_stats[agent]
        stats["total_severity"] += result.severity
        stats["turn_count"] += 1
        stats["max_severity"] = max(stats["max_severity"], result.severity)

        if result.should_alert:
            stats["alert_count"] += 1

    def get_agent_risk_score(self, agent: str) -> float:
        """
        Get cumulative risk score for an agent.

        Args:
            agent: Agent name.

        Returns:
            Risk score from 0-1.
        """
        if agent not in self._state.agent_stats:
            return 0.0

        stats = self._state.agent_stats[agent]
        if stats["turn_count"] == 0:
            return 0.0

        avg_severity = stats["total_severity"] / stats["turn_count"]
        max_factor = stats["max_severity"] * 0.3
        alert_factor = min(1.0, stats["alert_count"] * 0.1)

        return min(1.0, avg_severity * 0.5 + max_factor + alert_factor)

    def get_recent_severity(self) -> float:
        """Get average severity over recent turns."""
        if not self._state.severity_history:
            return 0.0

        recent = self._state.severity_history[-self.config.window_size:]
        return sum(recent) / len(recent)

    def get_severity_trend(self) -> str:
        """Get trend in severity over time."""
        if len(self._state.severity_history) < 4:
            return "stable"

        recent = self._state.severity_history[-self.config.window_size:]
        older = self._state.severity_history[
            -self.config.window_size * 2:-self.config.window_size
        ]

        if not older:
            return "stable"

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        diff = recent_avg - older_avg
        if diff > 0.1:
            return "increasing"
        elif diff < -0.1:
            return "decreasing"
        return "stable"

    def reset_state(self) -> None:
        """Reset monitor state."""
        self._state = MonitorState()
        logger.debug("Monitor state reset", monitor=self.name)

    def create_indicator(
        self,
        indicator_type: SafetyIndicatorType,
        severity: float,
        evidence: str | list[str],
        **metadata: Any,
    ) -> SafetyIndicator:
        """
        Create a safety indicator.

        Args:
            indicator_type: Type of indicator.
            severity: Severity score (0-1).
            evidence: Evidence supporting the indicator.
            **metadata: Additional metadata.

        Returns:
            SafetyIndicator instance.
        """
        return SafetyIndicator(
            type=indicator_type,
            severity=min(1.0, max(0.0, severity)),
            evidence=evidence,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"mode={self.config.mode.value!r})"
        )


class CompositeMonitor(SafetyMonitor):
    """
    A monitor that combines multiple sub-monitors.

    Aggregates results from multiple monitors into a single result.

    Example:
        >>> composite = CompositeMonitor([
        ...     SandbagDetector(),
        ...     DeceptionMonitor(),
        ... ])
        >>> result = await composite.analyze(turn, context)
    """

    def __init__(
        self,
        monitors: list[SafetyMonitor],
        aggregation: str = "max",
        **kwargs: Any,
    ) -> None:
        """
        Initialize a composite monitor.

        Args:
            monitors: List of sub-monitors.
            aggregation: How to aggregate severities ('max', 'mean', 'sum').
            **kwargs: Additional configuration.
        """
        super().__init__(**kwargs)
        self.monitors = monitors
        self.aggregation = aggregation

    @property
    def name(self) -> str:
        return "composite_monitor"

    @property
    def monitor_type(self) -> str:
        return "composite"

    async def analyze(
        self,
        turn: Turn,
        context: DebateContext,
    ) -> SafetyResult:
        """Analyze using all sub-monitors."""
        results: list[SafetyResult] = []

        for monitor in self.monitors:
            if monitor.is_enabled:
                result = await monitor.process(turn, context)
                results.append(result)

        if not results:
            return SafetyResult(monitor=self.name, severity=0.0)

        # Aggregate severities
        severities = [r.severity for r in results]
        if self.aggregation == "max":
            severity = max(severities)
        elif self.aggregation == "mean":
            severity = sum(severities) / len(severities)
        else:  # sum, capped at 1
            severity = min(1.0, sum(severities))

        # Collect all indicators
        indicators = []
        for result in results:
            indicators.extend(result.indicators)

        # Aggregate flags
        should_alert = any(r.should_alert for r in results)
        should_halt = any(r.should_halt for r in results)

        # Combine notes
        notes = [r.analysis_notes for r in results if r.analysis_notes]
        analysis_notes = " | ".join(notes) if notes else None

        return SafetyResult(
            monitor=self.name,
            severity=severity,
            indicators=indicators,
            should_alert=should_alert,
            should_halt=should_halt,
            analysis_notes=analysis_notes,
        )


class MonitorRegistry:
    """
    Registry for managing safety monitors.

    Provides centralized monitor management with lookup and lifecycle control.
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._monitors: dict[str, SafetyMonitor] = {}
        self._created_at = datetime.utcnow()

        logger.debug("MonitorRegistry initialized")

    def register(self, monitor: SafetyMonitor) -> None:
        """
        Register a monitor.

        Args:
            monitor: Monitor to register.

        Raises:
            ValueError: If monitor with same name already registered.
        """
        if monitor.name in self._monitors:
            raise ValueError(f"Monitor '{monitor.name}' already registered")

        self._monitors[monitor.name] = monitor
        logger.info("Monitor registered", monitor=monitor.name)

    def unregister(self, name: str) -> SafetyMonitor | None:
        """
        Unregister a monitor by name.

        Args:
            name: Monitor name.

        Returns:
            The unregistered monitor or None.
        """
        monitor = self._monitors.pop(name, None)
        if monitor:
            logger.info("Monitor unregistered", monitor=name)
        return monitor

    def get(self, name: str) -> SafetyMonitor | None:
        """Get a monitor by name."""
        return self._monitors.get(name)

    def get_all(self) -> list[SafetyMonitor]:
        """Get all registered monitors."""
        return list(self._monitors.values())

    def get_by_type(self, monitor_type: str) -> list[SafetyMonitor]:
        """Get monitors by type."""
        return [
            m for m in self._monitors.values()
            if m.monitor_type == monitor_type
        ]

    def get_by_priority(self, priority: MonitorPriority) -> list[SafetyMonitor]:
        """Get monitors by priority."""
        return [
            m for m in self._monitors.values()
            if m.config.priority == priority
        ]

    def get_active(self) -> list[SafetyMonitor]:
        """Get all active (non-passive) monitors."""
        return [m for m in self._monitors.values() if m.is_active]

    def get_enabled(self) -> list[SafetyMonitor]:
        """Get all enabled monitors."""
        return [m for m in self._monitors.values() if m.is_enabled]

    def enable_all(self) -> None:
        """Enable all monitors."""
        for monitor in self._monitors.values():
            monitor.config.enabled = True

    def disable_all(self) -> None:
        """Disable all monitors."""
        for monitor in self._monitors.values():
            monitor.config.enabled = False

    def reset_all(self) -> None:
        """Reset state of all monitors."""
        for monitor in self._monitors.values():
            monitor.reset_state()

    def __len__(self) -> int:
        return len(self._monitors)

    def __contains__(self, name: str) -> bool:
        return name in self._monitors

    def __iter__(self):
        return iter(self._monitors.values())


class SafetyManager:
    """
    High-level manager for safety monitoring.

    Coordinates multiple monitors and provides aggregate analysis.

    Example:
        >>> manager = SafetyManager()
        >>> manager.add_monitor(SandbagDetector())
        >>> manager.add_monitor(DeceptionMonitor())
        >>> results = await manager.analyze_turn(turn, context)
    """

    def __init__(
        self,
        mode: MonitorMode = MonitorMode.PASSIVE,
        halt_on_critical: bool = False,
    ) -> None:
        """
        Initialize the safety manager.

        Args:
            mode: Default mode for added monitors.
            halt_on_critical: Whether to halt on critical severity.
        """
        self.default_mode = mode
        self.halt_on_critical = halt_on_critical
        self._registry = MonitorRegistry()

        logger.debug(
            "SafetyManager initialized",
            mode=mode.value,
            halt_on_critical=halt_on_critical,
        )

    @property
    def monitors(self) -> list[SafetyMonitor]:
        """Get all registered monitors."""
        return self._registry.get_all()

    def add_monitor(self, monitor: SafetyMonitor) -> None:
        """Add a monitor to management."""
        self._registry.register(monitor)

    def remove_monitor(self, name: str) -> SafetyMonitor | None:
        """Remove a monitor by name."""
        return self._registry.unregister(name)

    def get_monitor(self, name: str) -> SafetyMonitor | None:
        """Get a monitor by name."""
        return self._registry.get(name)

    async def analyze_turn(
        self,
        turn: Turn,
        context: DebateContext,
    ) -> list[SafetyResult]:
        """
        Analyze a turn with all enabled monitors.

        Args:
            turn: Turn to analyze.
            context: Debate context.

        Returns:
            List of results from all monitors.
        """
        results: list[SafetyResult] = []

        for monitor in self._registry.get_enabled():
            result = await monitor.process(turn, context)
            results.append(result)

        return results

    def get_aggregate_severity(self, results: list[SafetyResult]) -> float:
        """Get maximum severity from results."""
        if not results:
            return 0.0
        return max(r.severity for r in results)

    def should_halt(self, results: list[SafetyResult]) -> bool:
        """Check if any result indicates debate should halt."""
        return any(r.should_halt for r in results)

    def get_all_alerts(self, results: list[SafetyResult]) -> list[SafetyResult]:
        """Get all results that triggered alerts."""
        return [r for r in results if r.should_alert]

    def get_risk_summary(self) -> dict[str, dict[str, float]]:
        """Get risk summary for all agents across all monitors."""
        summary: dict[str, dict[str, float]] = {}

        for monitor in self._registry.get_all():
            for agent, _stats in monitor.state.agent_stats.items():
                if agent not in summary:
                    summary[agent] = {"total_risk": 0.0, "monitors": 0}

                risk = monitor.get_agent_risk_score(agent)
                summary[agent]["total_risk"] += risk
                summary[agent]["monitors"] += 1
                summary[agent][f"{monitor.name}_risk"] = risk

        # Calculate average risk
        for agent in summary:
            if summary[agent]["monitors"] > 0:
                summary[agent]["avg_risk"] = (
                    summary[agent]["total_risk"] / summary[agent]["monitors"]
                )

        return summary

    def reset_all(self) -> None:
        """Reset all monitor states."""
        self._registry.reset_all()

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return (
            f"SafetyManager(monitors={len(self._registry)}, "
            f"mode={self.default_mode.value})"
        )
