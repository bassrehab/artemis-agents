"""Steering effectiveness analyzer.

Analyzes outputs to measure how well they match target steering vectors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from artemis.steering.vectors import SteeringVector


@dataclass
class StyleMetrics:
    """Measured style metrics from text analysis."""

    formality: float = 0.5
    aggression: float = 0.3
    evidence_emphasis: float = 0.5
    conciseness: float = 0.5
    emotional_appeal: float = 0.3
    confidence: float = 0.5
    creativity: float = 0.5  # harder to assess heuristically

    def to_vector(self) -> SteeringVector:
        return SteeringVector(
            formality=self.formality,
            aggression=self.aggression,
            evidence_emphasis=self.evidence_emphasis,
            conciseness=self.conciseness,
            emotional_appeal=self.emotional_appeal,
            confidence=self.confidence,
            creativity=self.creativity,
        )


class SteeringEffectivenessAnalyzer:
    """Analyzes text to measure steering effectiveness."""

    # Formal language indicators
    FORMAL_INDICATORS = [
        r"\bfurthermore\b",
        r"\bmoreover\b",
        r"\btherefore\b",
        r"\bconsequently\b",
        r"\bnevertheless\b",
        r"\bnotwithstanding\b",
        r"\bheretofore\b",
        r"\bwherein\b",
        r"\bthereby\b",
        r"\binasmuch\b",
    ]

    INFORMAL_INDICATORS = [
        r"\bkinda\b",
        r"\bgonna\b",
        r"\bwanna\b",
        r"\byeah\b",
        r"\bnope\b",
        r"\bstuff\b",
        r"\bthings\b",
        r"\blike\b",
        r"!{2,}",
        r"\.{3,}",
    ]

    # Aggression indicators
    AGGRESSIVE_INDICATORS = [
        r"\bwrong\b",
        r"\bfailed\b",
        r"\babsurd\b",
        r"\bridiculous\b",
        r"\bfoolish\b",
        r"\bobviously\b",
        r"\bclearly\b",
        r"\bundeniably\b",
        r"\brefute\b",
        r"\bdisprove\b",
    ]

    COOPERATIVE_INDICATORS = [
        r"\bperhaps\b",
        r"\bmight\b",
        r"\bconsider\b",
        r"\bagree\b",
        r"\bvalid point\b",
        r"\bunderstand\b",
        r"\bappreciate\b",
        r"\backnowledge\b",
        r"\bcommon ground\b",
    ]

    # Evidence indicators
    EVIDENCE_INDICATORS = [
        r"\bstudy\b",
        r"\bresearch\b",
        r"\bdata\b",
        r"\bstatistic",
        r"\bpercent\b",
        r"\b\d+%\b",
        r"\bfinding",
        r"\bevidence\b",
        r"\bsource",
        r"\baccording to\b",
        r"\bcitation\b",
        r"\breport\b",
    ]

    # Hedging indicators (low confidence)
    HEDGING_INDICATORS = [
        r"\bperhaps\b",
        r"\bmight\b",
        r"\bcould\b",
        r"\bmay\b",
        r"\bpossibly\b",
        r"\bseems?\b",
        r"\bappears?\b",
        r"\bsuggests?\b",
        r"\btends? to\b",
    ]

    # Assertive indicators (high confidence)
    ASSERTIVE_INDICATORS = [
        r"\bdefinitely\b",
        r"\bcertainly\b",
        r"\bundoubtedly\b",
        r"\bunquestionably\b",
        r"\bis\b",
        r"\bare\b",
        r"\bwill\b",
        r"\bmust\b",
        r"\bproves?\b",
    ]

    # Emotional indicators
    EMOTIONAL_INDICATORS = [
        r"\bfeel\b",
        r"\bheart\b",
        r"\bpassion",
        r"\bhope\b",
        r"\bfear\b",
        r"\blove\b",
        r"\bhate\b",
        r"\bexcit",
        r"\bworr",
        r"!",
    ]

    def analyze_output(self, text: str) -> StyleMetrics:
        """Analyze text to extract style metrics."""
        text_lower = text.lower()
        word_count = len(text.split())

        if word_count == 0:
            return StyleMetrics()

        # Analyze formality
        formal_count = self._count_patterns(text_lower, self.FORMAL_INDICATORS)
        informal_count = self._count_patterns(text_lower, self.INFORMAL_INDICATORS)
        formality = self._calculate_ratio(formal_count, informal_count, base=0.5)

        # Analyze aggression
        aggressive_count = self._count_patterns(text_lower, self.AGGRESSIVE_INDICATORS)
        cooperative_count = self._count_patterns(
            text_lower, self.COOPERATIVE_INDICATORS
        )
        aggression = self._calculate_ratio(
            aggressive_count, cooperative_count, base=0.3
        )

        # Analyze evidence emphasis
        evidence_count = self._count_patterns(text_lower, self.EVIDENCE_INDICATORS)
        evidence_emphasis = min(1.0, 0.3 + (evidence_count / word_count) * 20)

        # Analyze conciseness
        avg_sentence_length = self._avg_sentence_length(text)
        # Lower sentence length = more concise
        conciseness = max(0.0, min(1.0, 1.0 - (avg_sentence_length - 10) / 40))

        # Analyze confidence
        hedging_count = self._count_patterns(text_lower, self.HEDGING_INDICATORS)
        assertive_count = self._count_patterns(text_lower, self.ASSERTIVE_INDICATORS)
        confidence = self._calculate_ratio(assertive_count, hedging_count, base=0.5)

        # Analyze emotional appeal
        emotional_count = self._count_patterns(text_lower, self.EMOTIONAL_INDICATORS)
        emotional_appeal = min(1.0, 0.2 + (emotional_count / word_count) * 15)

        # TODO: creativity is hard to measure heuristically - needs better approach
        creativity = 0.5

        return StyleMetrics(
            formality=formality,
            aggression=aggression,
            evidence_emphasis=evidence_emphasis,
            conciseness=conciseness,
            emotional_appeal=emotional_appeal,
            confidence=confidence,
            creativity=creativity,
        )

    def _count_patterns(self, text: str, patterns: list[str]) -> int:
        total = 0
        for pattern in patterns:
            total += len(re.findall(pattern, text, re.IGNORECASE))
        return total

    def _calculate_ratio(
        self, positive: int, negative: int, base: float = 0.5
    ) -> float:
        total = positive + negative
        if total == 0:
            return base

        # Ratio adjusted around base
        ratio = positive / total
        return ratio

    def _avg_sentence_length(self, text: str) -> float:
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 20  # Default

        total_words = sum(len(s.split()) for s in sentences)
        return total_words / len(sentences)

    def calculate_effectiveness(
        self, metrics: StyleMetrics, target: SteeringVector
    ) -> float:
        """Calculate how well metrics match target vector."""
        measured = metrics.to_vector()
        distance = measured.distance(target)

        # Max possible distance is sqrt(7) ≈ 2.65 (7 dimensions, max diff 1.0 each)
        max_distance = 7 ** 0.5

        # Convert distance to effectiveness (inverse)
        effectiveness = 1.0 - (distance / max_distance)

        return max(0.0, min(1.0, effectiveness))

    def get_dimension_matches(
        self, metrics: StyleMetrics, target: SteeringVector
    ) -> dict[str, float]:
        """Get per-dimension match scores."""
        matches = {}
        measured = metrics.to_vector()

        for dim in target._dimensions():
            target_val = getattr(target, dim)
            measured_val = getattr(measured, dim)
            diff = abs(target_val - measured_val)
            matches[dim] = 1.0 - diff

        return matches
