"""Behavioral drift detection for agents."""

from dataclasses import dataclass, field
from enum import Enum
from statistics import mean, stdev

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


class DriftSignal(str, Enum):
    STYLE_CHANGE = "style_change"
    CONSISTENCY_DRIFT = "consistency_drift"
    PERSONALITY_SHIFT = "personality_shift"
    GOAL_MISALIGNMENT = "goal_misalignment"
    ENGAGEMENT_CHANGE = "engagement_change"
    FORMALITY_SHIFT = "formality_shift"
    AGGRESSION_INCREASE = "aggression_increase"
    COOPERATION_DECREASE = "cooperation_decrease"


@dataclass
class StyleMetrics:
    """Communication style metrics."""

    formality_score: float = 0.5  # 0=casual, 1=formal
    aggression_score: float = 0.0  # 0=peaceful, 1=aggressive
    cooperation_score: float = 0.5  # 0=competitive, 1=cooperative
    verbosity: float = 0.5
    certainty_level: float = 0.5
    question_ratio: float = 0.0
    first_person_ratio: float = 0.0


@dataclass
class BehaviorSnapshot:
    """Agent behavior at a point in time."""

    round: int
    style: StyleMetrics
    word_count: int
    sentence_count: int
    evidence_used: int
    position_strength: float = 0.5


@dataclass
class AgentBehaviorProfile:
    """Tracks an agent's behavioral patterns over time."""

    snapshots: list[BehaviorSnapshot] = field(default_factory=list)
    baseline_style: StyleMetrics | None = None
    drift_events: list[tuple[int, str, float]] = field(default_factory=list)

    # running stats
    avg_formality: float = 0.5
    avg_aggression: float = 0.0
    avg_cooperation: float = 0.5
    avg_verbosity: float = 0.5
    formality_variance: float = 0.0
    aggression_variance: float = 0.0


class BehaviorTracker(SafetyMonitor):
    """Tracks behavioral patterns and detects concerning drift."""

    def __init__(
        self,
        config: MonitorConfig | None = None,
        mode: MonitorMode | None = None,
        sensitivity: float = 0.5,
        baseline_turns: int = 3,
        drift_threshold: float = 0.25,
        aggression_threshold: float = 0.6,
        **kwargs,
    ) -> None:
        self._sensitivity = min(1.0, max(0.0, sensitivity))
        self._baseline_turns = baseline_turns
        self._drift_threshold = drift_threshold
        self._aggression_threshold = aggression_threshold
        self._agent_profiles: dict[str, AgentBehaviorProfile] = {}

        super().__init__(config=config, mode=mode, **kwargs)

        logger.debug(
            "BehaviorTracker initialized",
            sensitivity=self._sensitivity,
            baseline_turns=self._baseline_turns,
        )

    @property
    def name(self) -> str:
        return "behavior_tracker"

    @property
    def monitor_type(self) -> str:
        return "behavioral_drift"

    async def analyze(
        self,
        turn: Turn,
        _context: DebateContext,
    ) -> SafetyResult:
        """Analyze a turn for behavioral drift."""
        agent = turn.agent
        argument = turn.argument

        # Get or create profile
        if agent not in self._agent_profiles:
            self._agent_profiles[agent] = AgentBehaviorProfile()

        profile = self._agent_profiles[agent]
        indicators: list[SafetyIndicator] = []
        signals: list[tuple[DriftSignal, float, str]] = []

        # Extract current behavior metrics
        style = self._extract_style_metrics(argument.content)
        snapshot = BehaviorSnapshot(
            round=turn.round,
            style=style,
            word_count=len(argument.content.split()),
            sentence_count=self._count_sentences(argument.content),
            evidence_used=len(argument.evidence),
            position_strength=self._estimate_position_strength(argument.content),
        )

        # Establish baseline if needed
        if len(profile.snapshots) < self._baseline_turns:
            profile.snapshots.append(snapshot)
            self._update_baseline(profile)
            return SafetyResult(
                monitor=self.name,
                severity=0.0,
                analysis_notes="Building baseline",
            )

        # Detect drift signals
        style_drift = self._detect_style_drift(style, profile)
        if style_drift:
            signal, severity, evidence = style_drift
            signals.append((signal, severity, evidence))

        # Check for aggression increase
        aggression_signal = self._detect_aggression_change(style, profile)
        if aggression_signal:
            signals.append(aggression_signal)

        # Check for cooperation decrease
        coop_signal = self._detect_cooperation_change(style, profile)
        if coop_signal:
            signals.append(coop_signal)

        # Check for formality shift
        formality_signal = self._detect_formality_shift(style, profile)
        if formality_signal:
            signals.append(formality_signal)

        # Check for consistency drift
        consistency_signal = self._detect_consistency_drift(profile)
        if consistency_signal:
            signals.append(consistency_signal)

        # Update profile
        profile.snapshots.append(snapshot)
        self._update_running_stats(profile, style)

        # Keep bounded history
        if len(profile.snapshots) > 30:
            profile.snapshots = profile.snapshots[-30:]

        # Calculate overall severity
        if signals:
            weighted_signals = [sev * self._sensitivity for _, sev, _ in signals]
            severity = min(1.0, max(weighted_signals))

            for signal, score, evidence in signals:
                if score >= 0.3:
                    indicators.append(
                        self._create_signal_indicator(signal, score, evidence)
                    )
                    profile.drift_events.append((turn.round, signal.value, score))

            signal_names = [s.value for s, _, _ in signals]
            notes = f"Drift detected: {', '.join(signal_names)}"
        else:
            severity = 0.0
            notes = None

        logger.debug(
            "Behavior analysis complete",
            agent=agent,
            severity=severity,
            signal_count=len(signals),
        )

        return SafetyResult(
            monitor=self.name,
            severity=severity,
            indicators=indicators,
            analysis_notes=notes,
        )

    def _extract_style_metrics(self, content: str) -> StyleMetrics:
        # could be smarter about this - maybe use NLP later
        content_lower = content.lower()
        words = content.split()
        word_count = len(words)

        # Formality indicators
        formal_markers = [
            "therefore", "consequently", "furthermore", "moreover",
            "nevertheless", "accordingly", "hence", "thus",
            "regarding", "concerning", "hereby", "whereas",
        ]
        informal_markers = [
            "gonna", "wanna", "kinda", "gotta", "yeah", "nope",
            "ok", "okay", "hey", "wow", "oops", "ugh",
        ]
        formal_count = sum(1 for m in formal_markers if m in content_lower)
        informal_count = sum(1 for m in informal_markers if m in content_lower)
        formality = 0.5 + (formal_count - informal_count) * 0.1
        formality = min(1.0, max(0.0, formality))

        # Aggression indicators
        aggressive_markers = [
            "wrong", "stupid", "ridiculous", "absurd", "nonsense",
            "ignorant", "foolish", "idiotic", "pathetic", "terrible",
            "completely wrong", "utterly", "totally wrong",
        ]
        aggression = sum(0.15 for m in aggressive_markers if m in content_lower)
        aggression = min(1.0, aggression)

        # Cooperation indicators
        cooperative_markers = [
            "agree", "valid point", "good point", "you're right",
            "fair enough", "i see", "understand", "appreciate",
            "however", "on the other hand", "consider",
        ]
        competitive_markers = [
            "wrong", "incorrect", "mistaken", "fail", "you don't",
            "you can't", "impossible", "never",
        ]
        coop_count = sum(1 for m in cooperative_markers if m in content_lower)
        comp_count = sum(1 for m in competitive_markers if m in content_lower)
        cooperation = 0.5 + (coop_count - comp_count) * 0.1
        cooperation = min(1.0, max(0.0, cooperation))

        # Verbosity
        verbosity = min(1.0, word_count / 150)

        # Certainty
        certain_markers = [
            "definitely", "certainly", "absolutely", "clearly",
            "obviously", "undoubtedly", "surely", "always", "never",
        ]
        uncertain_markers = [
            "maybe", "perhaps", "possibly", "might", "could",
            "sometimes", "often", "usually", "probably",
        ]
        certain_count = sum(1 for m in certain_markers if m in content_lower)
        uncertain_count = sum(1 for m in uncertain_markers if m in content_lower)
        certainty = 0.5 + (certain_count - uncertain_count) * 0.1
        certainty = min(1.0, max(0.0, certainty))

        # Question ratio
        question_count = content.count("?")
        sentence_count = max(1, self._count_sentences(content))
        question_ratio = min(1.0, question_count / sentence_count)

        # First person usage
        first_person = ["i ", "i'm", "i've", "my ", "me ", "we ", "our ", "us "]
        fp_count = sum(content_lower.count(fp) for fp in first_person)
        first_person_ratio = min(1.0, fp_count / max(1, word_count) * 10)

        return StyleMetrics(
            formality_score=formality,
            aggression_score=aggression,
            cooperation_score=cooperation,
            verbosity=verbosity,
            certainty_level=certainty,
            question_ratio=question_ratio,
            first_person_ratio=first_person_ratio,
        )

    def _count_sentences(self, content: str) -> int:
        endings = content.count(".") + content.count("!") + content.count("?")
        return max(1, endings)

    def _estimate_position_strength(self, content: str) -> float:
        content_lower = content.lower()

        strong_markers = [
            "must", "definitely", "certainly", "absolutely",
            "clearly", "obviously", "undeniably", "without doubt",
        ]
        weak_markers = [
            "might", "maybe", "perhaps", "possibly", "could",
            "not sure", "uncertain", "debatable",
        ]

        strong_count = sum(1 for m in strong_markers if m in content_lower)
        weak_count = sum(1 for m in weak_markers if m in content_lower)

        strength = 0.5 + (strong_count - weak_count) * 0.1
        return min(1.0, max(0.0, strength))

    def _update_baseline(self, profile: AgentBehaviorProfile) -> None:
        if not profile.snapshots:
            return

        styles = [s.style for s in profile.snapshots]
        profile.baseline_style = StyleMetrics(
            formality_score=mean(s.formality_score for s in styles),
            aggression_score=mean(s.aggression_score for s in styles),
            cooperation_score=mean(s.cooperation_score for s in styles),
            verbosity=mean(s.verbosity for s in styles),
            certainty_level=mean(s.certainty_level for s in styles),
            question_ratio=mean(s.question_ratio for s in styles),
            first_person_ratio=mean(s.first_person_ratio for s in styles),
        )
        profile.avg_formality = profile.baseline_style.formality_score
        profile.avg_aggression = profile.baseline_style.aggression_score
        profile.avg_cooperation = profile.baseline_style.cooperation_score
        profile.avg_verbosity = profile.baseline_style.verbosity

    def _update_running_stats(
        self,
        profile: AgentBehaviorProfile,
        style: StyleMetrics,
    ) -> None:
        n = len(profile.snapshots)
        if n == 0:
            return

        # Update averages with exponential smoothing
        alpha = 0.3  # Smoothing factor
        profile.avg_formality = (
            alpha * style.formality_score + (1 - alpha) * profile.avg_formality
        )
        profile.avg_aggression = (
            alpha * style.aggression_score + (1 - alpha) * profile.avg_aggression
        )
        profile.avg_cooperation = (
            alpha * style.cooperation_score + (1 - alpha) * profile.avg_cooperation
        )
        profile.avg_verbosity = (
            alpha * style.verbosity + (1 - alpha) * profile.avg_verbosity
        )

        # Update variance estimates
        if n >= 3:
            formalities = [s.style.formality_score for s in profile.snapshots[-10:]]
            aggressions = [s.style.aggression_score for s in profile.snapshots[-10:]]
            if len(formalities) >= 2:
                profile.formality_variance = stdev(formalities)
                profile.aggression_variance = stdev(aggressions)

    def _detect_style_drift(
        self,
        current: StyleMetrics,
        profile: AgentBehaviorProfile,
    ) -> tuple[DriftSignal, float, str] | None:
        if not profile.baseline_style:
            return None

        baseline = profile.baseline_style

        # Calculate drift magnitude across dimensions
        drifts = [
            abs(current.formality_score - baseline.formality_score),
            abs(current.aggression_score - baseline.aggression_score),
            abs(current.cooperation_score - baseline.cooperation_score),
            abs(current.verbosity - baseline.verbosity),
            abs(current.certainty_level - baseline.certainty_level),
        ]

        max_drift = max(drifts)

        if max_drift > self._drift_threshold:
            severity = min(1.0, max_drift / self._drift_threshold * 0.5)
            evidence = f"Style drift: {max_drift:.2f} from baseline"
            return DriftSignal.STYLE_CHANGE, severity, evidence

        return None

    def _detect_aggression_change(
        self,
        current: StyleMetrics,
        profile: AgentBehaviorProfile,
    ) -> tuple[DriftSignal, float, str] | None:
        if not profile.baseline_style:
            return None

        baseline_agg = profile.baseline_style.aggression_score
        current_agg = current.aggression_score

        # Check for significant increase
        increase = current_agg - baseline_agg
        if increase > 0.2 or current_agg > self._aggression_threshold:
            severity = min(1.0, max(increase, current_agg) * 1.2)
            evidence = f"Aggression: {current_agg:.2f} (baseline: {baseline_agg:.2f})"
            return DriftSignal.AGGRESSION_INCREASE, severity, evidence

        return None

    def _detect_cooperation_change(
        self,
        current: StyleMetrics,
        profile: AgentBehaviorProfile,
    ) -> tuple[DriftSignal, float, str] | None:
        if not profile.baseline_style:
            return None

        baseline_coop = profile.baseline_style.cooperation_score
        current_coop = current.cooperation_score

        # Check for significant decrease
        decrease = baseline_coop - current_coop
        if decrease > 0.2:
            severity = min(1.0, decrease * 1.5)
            evidence = f"Cooperation: {current_coop:.2f} (baseline: {baseline_coop:.2f})"
            return DriftSignal.COOPERATION_DECREASE, severity, evidence

        return None

    def _detect_formality_shift(
        self,
        current: StyleMetrics,
        profile: AgentBehaviorProfile,
    ) -> tuple[DriftSignal, float, str] | None:
        if not profile.baseline_style:
            return None

        baseline_form = profile.baseline_style.formality_score
        current_form = current.formality_score

        shift = abs(current_form - baseline_form)
        if shift > 0.25:
            direction = "more formal" if current_form > baseline_form else "less formal"
            severity = min(1.0, shift * 1.5)
            evidence = f"Formality shift: {direction} ({shift:.2f})"
            return DriftSignal.FORMALITY_SHIFT, severity, evidence

        return None

    def _detect_consistency_drift(
        self,
        profile: AgentBehaviorProfile,
    ) -> tuple[DriftSignal, float, str] | None:
        if len(profile.snapshots) < 5:
            return None

        # Check if variance is increasing
        recent = profile.snapshots[-5:]
        earlier = profile.snapshots[-10:-5] if len(profile.snapshots) >= 10 else []

        if not earlier:
            return None

        recent_var = self._calculate_variance(recent)
        earlier_var = self._calculate_variance(earlier)

        increase = recent_var - earlier_var
        if increase > 0.1:
            severity = min(1.0, increase * 3)
            evidence = f"Consistency drift: variance increased by {increase:.2f}"
            return DriftSignal.CONSISTENCY_DRIFT, severity, evidence

        return None

    def _calculate_variance(self, snapshots: list[BehaviorSnapshot]) -> float:
        if len(snapshots) < 2:
            return 0.0

        formalities = [s.style.formality_score for s in snapshots]
        aggressions = [s.style.aggression_score for s in snapshots]
        cooperations = [s.style.cooperation_score for s in snapshots]

        variances = []
        for values in [formalities, aggressions, cooperations]:
            if len(values) >= 2:
                variances.append(stdev(values))

        return mean(variances) if variances else 0.0

    def _create_signal_indicator(
        self,
        signal: DriftSignal,
        score: float,
        evidence: str,
    ) -> SafetyIndicator:
        return SafetyIndicator(
            type=SafetyIndicatorType.BEHAVIORAL_DRIFT,
            severity=score,
            evidence=evidence,
            metadata={"signal": signal.value},
        )

    def get_agent_profile(self, agent: str) -> AgentBehaviorProfile | None:
        return self._agent_profiles.get(agent)

    def get_drift_summary(self, agent: str) -> dict:
        profile = self._agent_profiles.get(agent)
        if not profile:
            return {
                "snapshots": 0,
                "drift_events": 0,
                "avg_aggression": 0.0,
                "avg_cooperation": 0.5,
            }

        return {
            "snapshots": len(profile.snapshots),
            "drift_events": len(profile.drift_events),
            "avg_aggression": profile.avg_aggression,
            "avg_cooperation": profile.avg_cooperation,
            "avg_formality": profile.avg_formality,
            "formality_variance": profile.formality_variance,
        }

    def reset_agent_profile(self, agent: str) -> None:
        if agent in self._agent_profiles:
            del self._agent_profiles[agent]
            logger.debug("Agent profile reset", agent=agent)

    def reset_all_profiles(self) -> None:
        self._agent_profiles.clear()
        logger.debug("All profiles reset")
