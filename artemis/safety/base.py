"""Safety monitor base classes and infrastructure."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from artemis.core.types import (
    DebateContext,
    SafetyIndicator,
    SafetyResult,
    Turn,
)
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


class MonitorMode(str, Enum):
    PASSIVE = "passive"
    ACTIVE = "active"
    LEARNING = "learning"


class MonitorPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MonitorConfig:
    mode: MonitorMode = MonitorMode.PASSIVE
    priority: MonitorPriority = MonitorPriority.MEDIUM
    alert_threshold: float = 0.7
    halt_threshold: float = 0.9
    enabled: bool = True
    window_size: int = 5
    cooldown_turns: int = 2
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorState:
    turn_count: int = 0
    alert_count: int = 0
    last_alert_turn: int = -100
    severity_history: list[float] = field(default_factory=list)
    indicator_history: list[SafetyIndicator] = field(default_factory=list)
    agent_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    custom_data: dict[str, Any] = field(default_factory=dict)


class SafetyMonitor(ABC):
    """Base class for safety monitors."""

    # FIXME: could use a cleaner config pattern here

    def __init__(
        self,
        config: MonitorConfig | None = None,
        mode: MonitorMode | None = None,
        **kwargs: Any,
    ) -> None:
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
        pass

    @property
    def monitor_type(self) -> str:
        return "general"

    @property
    def state(self) -> MonitorState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self.config.mode == MonitorMode.ACTIVE

    @property
    def is_enabled(self) -> bool:
        return self.config.enabled

    @abstractmethod
    async def analyze(
        self,
        turn: Turn,
        context: DebateContext,
    ) -> SafetyResult:
        pass

    async def process(
        self,
        turn: Turn,
        context: DebateContext,
    ) -> SafetyResult:
        """Run full monitoring pipeline on a turn."""
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

    def _apply_thresholds(self, result):
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

    def _update_agent_stats(self, agent, result):
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
        """Cumulative risk score for an agent (0-1)."""
        if agent not in self._state.agent_stats:
            return 0.0

        stats = self._state.agent_stats[agent]
        if stats["turn_count"] == 0:
            return 0.0

        avg_severity = stats["total_severity"] / stats["turn_count"]
        max_factor = stats["max_severity"] * 0.3
        alert_factor = min(1.0, stats["alert_count"] * 0.1)

        return min(1.0, avg_severity * 0.5 + max_factor + alert_factor)

    def get_recent_severity(self):
        if not self._state.severity_history:
            return 0.0

        recent = self._state.severity_history[-self.config.window_size:]
        return sum(recent) / len(recent)

    def get_severity_trend(self):
        # NOTE: returns "stable", "increasing", or "decreasing"
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

    def reset_state(self):
        self._state = MonitorState()
        logger.debug("Monitor state reset", monitor=self.name)

    def create_indicator(self, indicator_type, severity, evidence, **metadata):
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
    """Combines multiple monitors into one."""

    def __init__(
        self,
        monitors: list[SafetyMonitor],
        aggregation: str = "max",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.monitors = monitors
        self.aggregation = aggregation

    @property
    def name(self) -> str:
        return "composite_monitor"

    @property
    def monitor_type(self) -> str:
        return "composite"

    async def analyze(self, turn: Turn, context: DebateContext) -> SafetyResult:
        results = []

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
    """Simple registry for safety monitors."""

    def __init__(self):
        self._monitors: dict[str, SafetyMonitor] = {}
        self._created_at = datetime.utcnow()

        logger.debug("MonitorRegistry initialized")

    def register(self, monitor: SafetyMonitor) -> None:
        if monitor.name in self._monitors:
            raise ValueError(f"Monitor '{monitor.name}' already registered")

        self._monitors[monitor.name] = monitor
        logger.info("Monitor registered", monitor=monitor.name)

    def unregister(self, name: str):
        monitor = self._monitors.pop(name, None)
        if monitor:
            logger.info("Monitor unregistered", monitor=name)
        return monitor

    def get(self, name: str):
        return self._monitors.get(name)

    def get_all(self):
        return list(self._monitors.values())

    def get_by_type(self, monitor_type: str):
        return [
            m for m in self._monitors.values()
            if m.monitor_type == monitor_type
        ]

    def get_by_priority(self, priority):
        return [
            m for m in self._monitors.values()
            if m.config.priority == priority
        ]

    def get_active(self):
        return [m for m in self._monitors.values() if m.is_active]

    def get_enabled(self):
        return [m for m in self._monitors.values() if m.is_enabled]

    def enable_all(self):
        for monitor in self._monitors.values():
            monitor.config.enabled = True

    def disable_all(self):
        for monitor in self._monitors.values():
            monitor.config.enabled = False

    def reset_all(self):
        for monitor in self._monitors.values():
            monitor.reset_state()

    def __len__(self) -> int:
        return len(self._monitors)

    def __contains__(self, name: str) -> bool:
        return name in self._monitors

    def __iter__(self):
        return iter(self._monitors.values())


class SafetyManager:
    """Coordinates multiple safety monitors."""

    def __init__(
        self,
        mode: MonitorMode = MonitorMode.PASSIVE,
        halt_on_critical: bool = False,
    ) -> None:
        self.default_mode = mode
        self.halt_on_critical = halt_on_critical
        self._registry = MonitorRegistry()

        logger.debug(
            "SafetyManager initialized",
            mode=mode.value,
            halt_on_critical=halt_on_critical,
        )

    @property
    def monitors(self):
        return self._registry.get_all()

    def add_monitor(self, monitor):
        self._registry.register(monitor)

    def remove_monitor(self, name: str):
        return self._registry.unregister(name)

    def get_monitor(self, name: str):
        return self._registry.get(name)

    async def analyze_turn(self, turn: Turn, context: DebateContext):
        """Run all enabled monitors on a turn."""
        results = []

        for monitor in self._registry.get_enabled():
            result = await monitor.process(turn, context)
            results.append(result)

        return results

    def get_aggregate_severity(self, results):
        if not results:
            return 0.0
        return max(r.severity for r in results)

    def should_halt(self, results):
        return any(r.should_halt for r in results)

    def get_all_alerts(self, results):
        return [r for r in results if r.should_alert]

    def get_risk_summary(self):
        # FIXME: this could be more efficient
        summary = {}

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

    def reset_all(self):
        self._registry.reset_all()

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return (
            f"SafetyManager(monitors={len(self._registry)}, "
            f"mode={self.default_mode.value})"
        )
