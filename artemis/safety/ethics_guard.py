"""
ARTEMIS Ethics Guard

Enforces ethical boundaries during multi-agent debates:
- Harmful content detection
- Bias monitoring
- Fairness enforcement
- Ethical principle adherence
- Boundary violation alerts
"""

from dataclasses import dataclass, field
from enum import Enum

from artemis.core.types import (
    DebateContext,
    SafetyIndicator,
    SafetyIndicatorType,
    SafetyResult,
    Turn,
)
from artemis.safety.base import MonitorConfig, MonitorMode, SafetyMonitor
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


class EthicsViolation(str, Enum):
    """Types of ethics violations."""

    HARMFUL_CONTENT = "harmful_content"
    """Content that could cause harm."""

    BIAS_DETECTED = "bias_detected"
    """Discriminatory or biased content."""

    UNFAIR_TREATMENT = "unfair_treatment"
    """Unfair treatment of perspectives."""

    PRIVACY_VIOLATION = "privacy_violation"
    """Potential privacy concerns."""

    MISINFORMATION = "misinformation"
    """Potentially false or misleading claims."""

    MANIPULATION = "manipulation"
    """Manipulative rhetoric or tactics."""

    BOUNDARY_BREACH = "boundary_breach"
    """Crossing ethical boundaries."""

    PRINCIPLE_VIOLATION = "principle_violation"
    """Violation of ethical principles."""


@dataclass
class EthicsConfig:
    """Configuration for ethics enforcement."""

    harmful_content_threshold: float = 0.5
    """Threshold for flagging harmful content."""
    bias_threshold: float = 0.4
    """Threshold for bias detection."""
    fairness_threshold: float = 0.3
    """Threshold for fairness violations."""
    enabled_checks: list[str] = field(
        default_factory=lambda: [
            "harmful_content",
            "bias",
            "fairness",
            "privacy",
            "manipulation",
        ]
    )
    """List of enabled ethics checks."""
    custom_boundaries: dict[str, str] = field(default_factory=dict)
    """Custom ethical boundaries."""


@dataclass
class EthicsScore:
    """Ethics scoring for an argument."""

    overall: float = 1.0
    """Overall ethics score (1=ethical, 0=unethical)."""
    harmful_score: float = 0.0
    """Harmfulness score."""
    bias_score: float = 0.0
    """Bias score."""
    fairness_score: float = 1.0
    """Fairness score."""
    privacy_score: float = 1.0
    """Privacy protection score."""
    violations: list[tuple[EthicsViolation, float, str]] = field(default_factory=list)
    """(violation_type, severity, evidence) tuples."""


class EthicsGuard(SafetyMonitor):
    """
    Enforces ethical boundaries in agent debates.

    Monitors for harmful content, bias, unfairness, and other
    ethical violations to maintain debate integrity.

    Example:
        >>> guard = EthicsGuard(
        ...     mode=MonitorMode.ACTIVE,
        ...     sensitivity=0.7,
        ... )
        >>> result = await guard.analyze(turn, context)
        >>> if result.should_halt:
        ...     print(f"Ethics violation: {result.analysis_notes}")
    """

    # Harmful content patterns
    HARMFUL_PATTERNS: list[tuple[str, float]] = [
        ("kill", 0.7),
        ("murder", 0.8),
        ("attack", 0.5),
        ("destroy", 0.4),
        ("hate", 0.6),
        ("violence", 0.7),
        ("weapon", 0.6),
        ("bomb", 0.8),
        ("terror", 0.8),
        ("suicide", 0.7),
    ]

    # Bias indicators
    BIAS_PATTERNS: list[tuple[str, float]] = [
        ("all women", 0.6),
        ("all men", 0.6),
        ("those people", 0.5),
        ("they always", 0.5),
        ("they never", 0.5),
        ("typical", 0.4),
        ("obviously inferior", 0.8),
        ("superior race", 0.9),
        ("lesser", 0.5),
        ("backwards", 0.5),
    ]

    # Manipulation indicators
    MANIPULATION_PATTERNS: list[tuple[str, float]] = [
        ("you must believe", 0.6),
        ("only a fool", 0.7),
        ("everyone knows", 0.5),
        ("don't be stupid", 0.6),
        ("trust me", 0.4),
        ("wake up", 0.5),
        ("sheep", 0.6),
        ("brainwashed", 0.6),
        ("they don't want you to know", 0.7),
        ("hidden agenda", 0.5),
    ]

    # Privacy concern patterns
    PRIVACY_PATTERNS: list[tuple[str, float]] = [
        ("personal address", 0.7),
        ("phone number", 0.6),
        ("social security", 0.9),
        ("credit card", 0.8),
        ("private information", 0.5),
        ("confidential", 0.4),
        ("medical records", 0.7),
        ("financial details", 0.6),
    ]

    def __init__(
        self,
        config: MonitorConfig | None = None,
        mode: MonitorMode | None = None,
        sensitivity: float = 0.5,
        ethics_config: EthicsConfig | None = None,
        halt_on_violation: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the ethics guard.

        Args:
            config: Monitor configuration.
            mode: Operating mode.
            sensitivity: Detection sensitivity (0-1).
            ethics_config: Specific ethics configuration.
            halt_on_violation: Whether to halt on violations.
            **kwargs: Additional configuration.
        """
        self._sensitivity = min(1.0, max(0.0, sensitivity))
        self._ethics_config = ethics_config or EthicsConfig()
        self._halt_on_violation = halt_on_violation
        self._violation_history: dict[str, list[tuple[int, EthicsViolation, float]]] = (
            {}
        )

        super().__init__(config=config, mode=mode, **kwargs)

        logger.debug(
            "EthicsGuard initialized",
            sensitivity=self._sensitivity,
            halt_on_violation=self._halt_on_violation,
        )

    @property
    def name(self) -> str:
        return "ethics_guard"

    @property
    def monitor_type(self) -> str:
        return "ethics_enforcement"

    async def analyze(
        self,
        turn: Turn,
        _context: DebateContext,
    ) -> SafetyResult:
        """
        Analyze a turn for ethics violations.

        Args:
            turn: The turn to analyze.
            _context: Current debate context.

        Returns:
            SafetyResult with ethics severity and indicators.
        """
        agent = turn.agent
        content = turn.argument.content

        # Initialize violation history for agent
        if agent not in self._violation_history:
            self._violation_history[agent] = []

        indicators: list[SafetyIndicator] = []
        score = self._evaluate_ethics(content)

        # Process violations
        for violation, severity, evidence in score.violations:
            if severity >= 0.3:
                indicators.append(
                    self._create_violation_indicator(violation, severity, evidence)
                )
                self._violation_history[agent].append(
                    (turn.round, violation, severity)
                )

        # Calculate overall severity
        if score.violations:
            max_severity = max(sev for _, sev, _ in score.violations)
            severity = min(1.0, max_severity * self._sensitivity)

            violation_names = [v.value for v, _, _ in score.violations]
            notes = f"Ethics violations: {', '.join(violation_names)}"

            # Check if we should halt
            should_halt = (
                self._halt_on_violation
                and self.is_active
                and severity >= 0.7
            )
        else:
            severity = 0.0
            notes = None
            should_halt = False

        logger.debug(
            "Ethics analysis complete",
            agent=agent,
            severity=severity,
            violation_count=len(score.violations),
        )

        return SafetyResult(
            monitor=self.name,
            severity=severity,
            indicators=indicators,
            should_halt=should_halt,
            analysis_notes=notes,
        )

    def _evaluate_ethics(self, content: str) -> EthicsScore:
        """Evaluate content for ethics issues."""
        content_lower = content.lower()
        violations: list[tuple[EthicsViolation, float, str]] = []

        # Check harmful content
        if "harmful_content" in self._ethics_config.enabled_checks:
            harmful_result = self._check_harmful_content(content_lower)
            if harmful_result:
                violations.append(harmful_result)

        # Check bias
        if "bias" in self._ethics_config.enabled_checks:
            bias_result = self._check_bias(content_lower)
            if bias_result:
                violations.append(bias_result)

        # Check manipulation
        if "manipulation" in self._ethics_config.enabled_checks:
            manipulation_result = self._check_manipulation(content_lower)
            if manipulation_result:
                violations.append(manipulation_result)

        # Check privacy
        if "privacy" in self._ethics_config.enabled_checks:
            privacy_result = self._check_privacy(content_lower)
            if privacy_result:
                violations.append(privacy_result)

        # Check fairness
        if "fairness" in self._ethics_config.enabled_checks:
            fairness_result = self._check_fairness(content_lower)
            if fairness_result:
                violations.append(fairness_result)

        # Calculate scores
        harmful_score = sum(
            sev for v, sev, _ in violations if v == EthicsViolation.HARMFUL_CONTENT
        )
        bias_score = sum(
            sev for v, sev, _ in violations if v == EthicsViolation.BIAS_DETECTED
        )

        overall = 1.0 - min(1.0, (harmful_score + bias_score) / 2)

        return EthicsScore(
            overall=overall,
            harmful_score=harmful_score,
            bias_score=bias_score,
            violations=violations,
        )

    def _check_harmful_content(
        self, content: str
    ) -> tuple[EthicsViolation, float, str] | None:
        """Check for harmful content."""
        matches = []
        for pattern, weight in self.HARMFUL_PATTERNS:
            if pattern in content:
                matches.append((pattern, weight))

        if matches:
            max_weight = max(w for _, w in matches)
            if max_weight >= self._ethics_config.harmful_content_threshold:
                patterns_found = ", ".join(p for p, _ in matches)
                return (
                    EthicsViolation.HARMFUL_CONTENT,
                    max_weight,
                    f"Harmful patterns: {patterns_found}",
                )

        return None

    def _check_bias(self, content: str) -> tuple[EthicsViolation, float, str] | None:
        """Check for biased content."""
        matches = []
        for pattern, weight in self.BIAS_PATTERNS:
            if pattern in content:
                matches.append((pattern, weight))

        if matches:
            max_weight = max(w for _, w in matches)
            if max_weight >= self._ethics_config.bias_threshold:
                patterns_found = ", ".join(p for p, _ in matches)
                return (
                    EthicsViolation.BIAS_DETECTED,
                    max_weight,
                    f"Bias patterns: {patterns_found}",
                )

        return None

    def _check_manipulation(
        self, content: str
    ) -> tuple[EthicsViolation, float, str] | None:
        """Check for manipulative content."""
        matches = []
        for pattern, weight in self.MANIPULATION_PATTERNS:
            if pattern in content:
                matches.append((pattern, weight))

        if matches:
            max_weight = max(w for _, w in matches)
            if max_weight >= 0.4:
                patterns_found = ", ".join(p for p, _ in matches)
                return (
                    EthicsViolation.MANIPULATION,
                    max_weight,
                    f"Manipulation patterns: {patterns_found}",
                )

        return None

    def _check_privacy(
        self, content: str
    ) -> tuple[EthicsViolation, float, str] | None:
        """Check for privacy concerns."""
        matches = []
        for pattern, weight in self.PRIVACY_PATTERNS:
            if pattern in content:
                matches.append((pattern, weight))

        if matches:
            max_weight = max(w for _, w in matches)
            if max_weight >= 0.5:
                patterns_found = ", ".join(p for p, _ in matches)
                return (
                    EthicsViolation.PRIVACY_VIOLATION,
                    max_weight,
                    f"Privacy concerns: {patterns_found}",
                )

        return None

    def _check_fairness(
        self, content: str
    ) -> tuple[EthicsViolation, float, str] | None:
        """Check for fairness issues."""
        unfair_markers = [
            ("obviously right", 0.4),
            ("only idiots", 0.7),
            ("anyone with a brain", 0.6),
            ("no reasonable person", 0.5),
            ("any intelligent person", 0.5),
        ]

        matches = []
        for pattern, weight in unfair_markers:
            if pattern in content:
                matches.append((pattern, weight))

        if matches:
            max_weight = max(w for _, w in matches)
            if max_weight >= self._ethics_config.fairness_threshold:
                patterns_found = ", ".join(p for p, _ in matches)
                return (
                    EthicsViolation.UNFAIR_TREATMENT,
                    max_weight,
                    f"Fairness concerns: {patterns_found}",
                )

        return None

    def _create_violation_indicator(
        self,
        violation: EthicsViolation,
        severity: float,
        evidence: str,
    ) -> SafetyIndicator:
        """Create a SafetyIndicator for a violation."""
        return SafetyIndicator(
            type=SafetyIndicatorType.ETHICS_BOUNDARY,
            severity=severity,
            evidence=evidence,
            metadata={"violation": violation.value},
        )

    def get_violation_history(
        self, agent: str
    ) -> list[tuple[int, EthicsViolation, float]]:
        """Get violation history for an agent."""
        return self._violation_history.get(agent, [])

    def get_agent_ethics_summary(self, agent: str) -> dict:
        """Get ethics summary for an agent."""
        history = self._violation_history.get(agent, [])
        if not history:
            return {
                "total_violations": 0,
                "max_severity": 0.0,
                "violation_types": [],
            }

        violation_types = list({v.value for _, v, _ in history})
        max_severity = max(sev for _, _, sev in history)

        return {
            "total_violations": len(history),
            "max_severity": max_severity,
            "violation_types": violation_types,
        }

    def reset_agent_history(self, agent: str) -> None:
        """Reset violation history for an agent."""
        if agent in self._violation_history:
            del self._violation_history[agent]
            logger.debug("Agent ethics history reset", agent=agent)

    def reset_all_history(self) -> None:
        """Reset all violation history."""
        self._violation_history.clear()
        logger.debug("All ethics history reset")
