"""Feedback synthesis for closed-loop argument generation.

Converts evaluations into actionable guidance that agents can use
to improve their subsequent arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

from artemis.core.types import ArgumentEvaluation, Turn
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeedbackSummary:
    """Synthesized feedback for an agent."""

    agent: str
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]
    opponent_vulnerabilities: list[str]
    overall_guidance: str


class FeedbackSynthesizer:
    """Converts evaluation results into actionable feedback for agents.

    The key insight is that arguments improve when agents understand
    what's working and what's not. This creates a closed-loop system
    where evaluation signals shape future generation.
    """

    # Threshold for considering a score "strong" or "weak"
    STRONG_THRESHOLD = 0.7
    WEAK_THRESHOLD = 0.4

    def synthesize(
        self,
        agent_name: str,
        own_turns: list[Turn],
        opponent_turns: list[Turn],
    ) -> FeedbackSummary:
        """Synthesize feedback from evaluation history.

        Args:
            agent_name: Name of the agent receiving feedback
            own_turns: Agent's previous turns with evaluations
            opponent_turns: Opponent's turns with evaluations

        Returns:
            FeedbackSummary with actionable guidance
        """
        strengths: list[str] = []
        weaknesses: list[str] = []
        suggestions: list[str] = []
        opponent_vulnerabilities: list[str] = []

        # Analyze own performance
        if own_turns:
            own_evals = [t.evaluation for t in own_turns if t.evaluation]
            if own_evals:
                strengths, weaknesses = self._analyze_performance(own_evals)
                suggestions = self._generate_suggestions(weaknesses)

        # Analyze opponent's weaknesses
        if opponent_turns:
            opponent_evals = [t.evaluation for t in opponent_turns if t.evaluation]
            if opponent_evals:
                opponent_vulnerabilities = self._find_vulnerabilities(opponent_evals)

        # Generate overall guidance
        overall_guidance = self._generate_guidance(strengths, weaknesses, opponent_vulnerabilities)

        return FeedbackSummary(
            agent=agent_name,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            opponent_vulnerabilities=opponent_vulnerabilities,
            overall_guidance=overall_guidance,
        )

    def _analyze_performance(
        self, evaluations: list[ArgumentEvaluation]
    ) -> tuple[list[str], list[str]]:
        """Analyze evaluations to find strengths and weaknesses."""
        # Aggregate scores by criterion
        criterion_scores: dict[str, list[float]] = {}
        for evaluation in evaluations:
            for criterion, score in evaluation.scores.items():
                if criterion not in criterion_scores:
                    criterion_scores[criterion] = []
                criterion_scores[criterion].append(score)

        # Calculate averages and categorize
        strengths = []
        weaknesses = []

        criterion_labels = {
            "logical_coherence": "logical structure",
            "evidence_quality": "evidence and citations",
            "causal_reasoning": "causal reasoning",
            "ethical_alignment": "ethical considerations",
            "persuasiveness": "persuasive impact",
        }

        for criterion, scores in criterion_scores.items():
            avg_score = sum(scores) / len(scores)
            label = criterion_labels.get(criterion, criterion)

            if avg_score >= self.STRONG_THRESHOLD:
                strengths.append(f"Strong {label} (avg: {avg_score:.0%})")
            elif avg_score <= self.WEAK_THRESHOLD:
                weaknesses.append(f"Weak {label} (avg: {avg_score:.0%})")

        return strengths, weaknesses

    def _generate_suggestions(self, weaknesses: list[str]) -> list[str]:
        """Generate improvement suggestions based on weaknesses."""
        suggestions = []

        for weakness in weaknesses:
            if "logical" in weakness.lower():
                suggestions.append("Strengthen logical flow with explicit premises and conclusions")
            elif "evidence" in weakness.lower():
                suggestions.append("Add more specific evidence, statistics, or expert citations")
            elif "causal" in weakness.lower():
                suggestions.append("Clarify cause-effect relationships with explicit mechanisms")
            elif "ethical" in weakness.lower():
                suggestions.append("Address ethical implications and stakeholder impacts")
            elif "persuasive" in weakness.lower():
                suggestions.append(
                    "Use more compelling rhetorical techniques and address counterarguments"
                )

        return suggestions

    def _find_vulnerabilities(self, opponent_evals: list[ArgumentEvaluation]) -> list[str]:
        """Identify weaknesses in opponent's arguments to exploit."""
        vulnerabilities = []

        # Aggregate opponent's criterion scores
        criterion_scores: dict[str, list[float]] = {}
        for evaluation in opponent_evals:
            for criterion, score in evaluation.scores.items():
                if criterion not in criterion_scores:
                    criterion_scores[criterion] = []
                criterion_scores[criterion].append(score)

        # Find weak areas
        for criterion, scores in criterion_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score <= self.WEAK_THRESHOLD:
                if criterion == "logical_coherence":
                    vulnerabilities.append(
                        "Opponent's arguments have logical gaps - challenge their reasoning"
                    )
                elif criterion == "evidence_quality":
                    vulnerabilities.append("Opponent lacks strong evidence - demand citations")
                elif criterion == "causal_reasoning":
                    vulnerabilities.append(
                        "Opponent's causal claims are weak - question mechanisms"
                    )
                elif criterion == "ethical_alignment":
                    vulnerabilities.append("Opponent overlooks ethical concerns - highlight them")

        return vulnerabilities

    def _generate_guidance(
        self,
        strengths: list[str],
        weaknesses: list[str],
        opponent_vulnerabilities: list[str],
    ) -> str:
        """Generate overall guidance string for the agent."""
        parts = []

        if strengths:
            parts.append(f"Build on your strengths: {strengths[0].split('(')[0].strip()}")

        if weaknesses:
            parts.append(f"Focus on improving: {weaknesses[0].split('(')[0].strip()}")

        if opponent_vulnerabilities:
            parts.append(opponent_vulnerabilities[0])

        if not parts:
            parts.append("Continue with balanced, well-reasoned arguments")

        return ". ".join(parts) + "."

    def format_for_prompt(self, feedback: FeedbackSummary) -> str:
        """Format feedback summary for inclusion in generation prompt.

        Args:
            feedback: The synthesized feedback

        Returns:
            String suitable for including in agent prompts
        """
        lines = ["## Performance Feedback"]

        if feedback.strengths:
            lines.append("\n**Your Strengths:**")
            for s in feedback.strengths[:2]:
                lines.append(f"- {s}")

        if feedback.weaknesses:
            lines.append("\n**Areas to Improve:**")
            for w in feedback.weaknesses[:2]:
                lines.append(f"- {w}")

        if feedback.suggestions:
            lines.append("\n**Suggestions:**")
            for s in feedback.suggestions[:2]:
                lines.append(f"- {s}")

        if feedback.opponent_vulnerabilities:
            lines.append("\n**Opponent Weaknesses to Exploit:**")
            for v in feedback.opponent_vulnerabilities[:2]:
                lines.append(f"- {v}")

        lines.append(f"\n**Guidance:** {feedback.overall_guidance}")

        return "\n".join(lines)
