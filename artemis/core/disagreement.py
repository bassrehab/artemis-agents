"""Disagreement analysis for adaptive H-L-DAG level selection.

Analyzes where agents fundamentally disagree to guide argument level selection.
Instead of mechanically assigning levels by round number, this module examines
the debate transcript to determine whether disagreement is at the strategic,
tactical, or operational level.
"""

from __future__ import annotations

from enum import Enum

from artemis.core.types import ArgumentLevel, Turn
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


class DisagreementType(str, Enum):
    """Type of disagreement between agents."""

    STRATEGIC = "strategic"
    """Fundamental disagreement on core position or thesis."""

    TACTICAL = "tactical"
    """Disagreement on supporting arguments or evidence interpretation."""

    OPERATIONAL = "operational"
    """Disagreement on specific facts, examples, or details."""

    MIXED = "mixed"
    """Disagreement spans multiple levels."""


class DisagreementAnalyzer:
    """Analyzes debate transcripts to determine disagreement level.

    The key insight is that arguments improve when they target the actual
    level of disagreement. If agents disagree on core values (strategic),
    operational details won't resolve the debate. If they agree on values
    but disagree on facts (operational), strategic arguments waste time.
    """

    # Keywords that indicate different levels of discourse
    STRATEGIC_INDICATORS = [
        "fundamentally",
        "principle",
        "value",
        "core",
        "essential",
        "philosophy",
        "worldview",
        "belief",
        "ideology",
        "morally",
        "ethically",
        "should",
        "ought",
        "must",
        "right",
        "wrong",
    ]

    TACTICAL_INDICATORS = [
        "evidence",
        "study",
        "research",
        "data",
        "analysis",
        "support",
        "demonstrate",
        "show",
        "prove",
        "argument",
        "reasoning",
        "logic",
        "because",
        "therefore",
        "consequently",
    ]

    OPERATIONAL_INDICATORS = [
        "specifically",
        "example",
        "instance",
        "case",
        "statistic",
        "percent",
        "number",
        "fact",
        "quote",
        "said",
        "reported",
        "according",
        "detail",
        "precisely",
        "exactly",
    ]

    def analyze(
        self,
        transcript: list[Turn],
        current_round: int,
        total_rounds: int,
    ) -> DisagreementType:
        """Analyze transcript to determine disagreement level.

        Args:
            transcript: Debate transcript so far
            current_round: Current round number
            total_rounds: Total rounds in debate

        Returns:
            DisagreementType indicating where agents disagree
        """
        if not transcript:
            # No transcript yet, default to strategic
            return DisagreementType.STRATEGIC

        # Get recent turns (last 4-6 turns are most relevant)
        recent_turns = transcript[-6:] if len(transcript) > 6 else transcript

        # Count level indicators in recent content
        strategic_count = 0
        tactical_count = 0
        operational_count = 0

        for turn in recent_turns:
            content = turn.argument.content.lower()

            strategic_count += sum(1 for ind in self.STRATEGIC_INDICATORS if ind in content)
            tactical_count += sum(1 for ind in self.TACTICAL_INDICATORS if ind in content)
            operational_count += sum(1 for ind in self.OPERATIONAL_INDICATORS if ind in content)

        total_indicators = strategic_count + tactical_count + operational_count

        if total_indicators == 0:
            # No clear indicators, use round-based heuristic
            return self._fallback_by_round(current_round, total_rounds)

        # Calculate proportions
        strategic_prop = strategic_count / total_indicators
        tactical_prop = tactical_count / total_indicators
        operational_prop = operational_count / total_indicators

        # Determine primary disagreement level
        # Use 0.4 as threshold for dominance
        if strategic_prop >= 0.4:
            result = DisagreementType.STRATEGIC
        elif operational_prop >= 0.4:
            result = DisagreementType.OPERATIONAL
        elif tactical_prop >= 0.4:
            result = DisagreementType.TACTICAL
        else:
            result = DisagreementType.MIXED

        logger.debug(
            "Disagreement analyzed",
            strategic=strategic_prop,
            tactical=tactical_prop,
            operational=operational_prop,
            result=result.value,
        )

        return result

    def _fallback_by_round(
        self,
        current_round: int,
        total_rounds: int,
    ) -> DisagreementType:
        """Fallback to round-based heuristic when indicators are unclear."""
        progress = current_round / max(total_rounds, 1)

        if progress <= 0.3:
            return DisagreementType.STRATEGIC
        elif progress <= 0.7:
            return DisagreementType.TACTICAL
        else:
            return DisagreementType.OPERATIONAL

    def recommend_level(
        self,
        disagreement: DisagreementType,
        current_round: int,
        total_rounds: int,
    ) -> ArgumentLevel:
        """Recommend argument level based on disagreement analysis.

        Args:
            disagreement: Analyzed disagreement type
            current_round: Current round number
            total_rounds: Total rounds in debate

        Returns:
            Recommended ArgumentLevel for next argument
        """
        progress = current_round / max(total_rounds, 1)

        if disagreement == DisagreementType.STRATEGIC:
            # Stay strategic early, shift tactical late
            if progress <= 0.7:
                return ArgumentLevel.STRATEGIC
            else:
                return ArgumentLevel.TACTICAL

        elif disagreement == DisagreementType.TACTICAL:
            # Match tactical disagreement
            return ArgumentLevel.TACTICAL

        elif disagreement == DisagreementType.OPERATIONAL:
            # Match operational, but conclude strategically
            if progress >= 0.9:
                return ArgumentLevel.STRATEGIC
            else:
                return ArgumentLevel.OPERATIONAL

        else:
            # Mixed disagreement - adapt to debate phase
            if progress <= 0.3:
                return ArgumentLevel.STRATEGIC
            elif progress <= 0.7:
                return ArgumentLevel.TACTICAL
            else:
                return ArgumentLevel.OPERATIONAL
