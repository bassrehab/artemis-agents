"""Ethics boundary enforcement for debates."""

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
    HARMFUL_CONTENT = "harmful_content"
    BIAS_DETECTED = "bias_detected"
    UNFAIR_TREATMENT = "unfair_treatment"
    PRIVACY_VIOLATION = "privacy_violation"
    MISINFORMATION = "misinformation"
    MANIPULATION = "manipulation"
    BOUNDARY_BREACH = "boundary_breach"
    PRINCIPLE_VIOLATION = "principle_violation"


@dataclass
class EthicsConfig:
    """Config for ethics checks."""

    harmful_content_threshold: float = 0.5
    bias_threshold: float = 0.4
    fairness_threshold: float = 0.3
    enabled_checks: list[str] = field(
        default_factory=lambda: [
            "harmful_content",
            "bias",
            "fairness",
            "privacy",
            "manipulation",
        ]
    )
    custom_boundaries: dict[str, str] = field(default_factory=dict)


@dataclass
class EthicsScore:
    """Ethics scoring for an argument."""

    overall: float = 1.0  # 1=ethical, 0=unethical
    harmful_score: float = 0.0
    bias_score: float = 0.0
    fairness_score: float = 1.0
    privacy_score: float = 1.0
    violations: list[tuple[EthicsViolation, float, str]] = field(default_factory=list)


class EthicsGuard(SafetyMonitor):
    """Enforces ethical boundaries in agent debates."""

    # harmful content patterns - weights are rough estimates
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

    # bias indicators
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
        return SafetyIndicator(
            type=SafetyIndicatorType.ETHICS_BOUNDARY,
            severity=severity,
            evidence=evidence,
            metadata={"violation": violation.value},
        )

    def get_violation_history(
        self, agent: str
    ) -> list[tuple[int, EthicsViolation, float]]:
        return self._violation_history.get(agent, [])

    def get_agent_ethics_summary(self, agent: str) -> dict:
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
        if agent in self._violation_history:
            del self._violation_history[agent]
            logger.debug("Agent ethics history reset", agent=agent)

    def reset_all_history(self) -> None:
        self._violation_history.clear()
        logger.debug("All ethics history reset")
