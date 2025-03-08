"""
ARTEMIS Jury Panel

Implements the multi-perspective jury scoring mechanism for debate evaluation.
Provides independent evaluation, consensus building, and verdict generation.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from artemis.core.evaluation import AdaptiveEvaluator
from artemis.core.types import (
    ArgumentEvaluation,
    DebateContext,
    DissentingOpinion,
    JuryPerspective,
    Message,
    Turn,
    Verdict,
)
from artemis.models import BaseModel, create_model
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


# Default criteria for jury evaluation
DEFAULT_JURY_CRITERIA = [
    "argument_quality",
    "evidence_strength",
    "logical_consistency",
    "persuasiveness",
    "ethical_alignment",
]


@dataclass
class JurorEvaluation:
    """Evaluation result from a single juror."""

    juror_id: str
    perspective: JuryPerspective
    agent_scores: dict[str, float]
    """Scores for each agent in the debate."""
    criterion_scores: dict[str, dict[str, float]]
    """Scores by criterion for each agent."""
    winner: str
    """The juror's choice for winner."""
    confidence: float
    """Confidence in the evaluation."""
    reasoning: str
    """Explanation of the evaluation."""


@dataclass
class ConsensusResult:
    """Result of consensus building among jurors."""

    decision: str
    agreement_score: float
    """How much jurors agreed (0-1)."""
    supporting_jurors: list[str]
    dissenting_jurors: list[str]
    reasoning: str


class JuryMember:
    """
    A single jury member with a specific evaluation perspective.

    Each juror evaluates debates from their assigned perspective,
    providing independent scores and reasoning.

    Example:
        >>> juror = JuryMember(
        ...     juror_id="juror_0",
        ...     perspective=JuryPerspective.ANALYTICAL,
        ...     model="gpt-4o",
        ... )
        >>> evaluation = await juror.evaluate(transcript, context)
    """

    # Perspective-specific evaluation focus
    PERSPECTIVE_PROMPTS = {
        JuryPerspective.ANALYTICAL: (
            "You are an analytical juror who focuses on logic, evidence quality, "
            "and the strength of reasoning chains. Evaluate arguments based on: "
            "- Logical consistency and validity of inferences "
            "- Quality and relevance of evidence cited "
            "- Strength of causal reasoning "
            "- Absence of logical fallacies"
        ),
        JuryPerspective.ETHICAL: (
            "You are an ethical juror who focuses on moral implications and values. "
            "Evaluate arguments based on: "
            "- Consideration of ethical principles "
            "- Attention to stakeholder welfare "
            "- Fairness and justice concerns "
            "- Long-term societal impact"
        ),
        JuryPerspective.PRACTICAL: (
            "You are a practical juror who focuses on feasibility and real-world impact. "
            "Evaluate arguments based on: "
            "- Practicality of proposed solutions "
            "- Real-world applicability "
            "- Implementation challenges considered "
            "- Cost-benefit analysis"
        ),
        JuryPerspective.ADVERSARIAL: (
            "You are an adversarial juror who challenges all arguments critically. "
            "Evaluate arguments based on: "
            "- Ability to withstand counterarguments "
            "- Acknowledgment of weaknesses "
            "- Response to opposing views "
            "- Robustness under scrutiny"
        ),
        JuryPerspective.SYNTHESIZING: (
            "You are a synthesizing juror who seeks common ground and integration. "
            "Evaluate arguments based on: "
            "- Recognition of valid points from all sides "
            "- Ability to build on others' arguments "
            "- Constructive framing of disagreements "
            "- Movement toward resolution"
        ),
    }

    def __init__(
        self,
        juror_id: str,
        perspective: JuryPerspective,
        model: str | BaseModel = "gpt-4o",
        criteria: list[str] | None = None,
        api_key: str | None = None,
        **model_kwargs: Any,
    ) -> None:
        """
        Initialize a jury member.

        Args:
            juror_id: Unique identifier for this juror.
            perspective: The evaluation perspective to use.
            model: Model identifier or BaseModel instance.
            criteria: Custom evaluation criteria.
            api_key: Optional API key for the model provider.
            **model_kwargs: Additional model arguments.
        """
        self.juror_id = juror_id
        self.perspective = perspective
        self.criteria = criteria or DEFAULT_JURY_CRITERIA

        # Initialize model
        if isinstance(model, BaseModel):
            self._model = model
        else:
            self._model = create_model(model, api_key=api_key, **model_kwargs)

        # Internal evaluator for scoring
        self._evaluator = AdaptiveEvaluator()

        logger.debug(
            "JuryMember initialized",
            juror_id=juror_id,
            perspective=perspective.value,
            model=self._model.model,
        )

    async def evaluate(
        self,
        transcript: list[Turn],
        context: DebateContext,
    ) -> JurorEvaluation:
        """
        Evaluate a debate transcript from this juror's perspective.

        Args:
            transcript: List of debate turns.
            context: Debate context.

        Returns:
            JurorEvaluation with scores and reasoning.
        """
        logger.info(
            "Juror evaluating debate",
            juror_id=self.juror_id,
            perspective=self.perspective.value,
            turns=len(transcript),
        )

        # Collect agents from transcript
        agents = list({turn.agent for turn in transcript})

        # Evaluate each argument using internal evaluator
        argument_evaluations: dict[str, list[ArgumentEvaluation]] = {
            agent: [] for agent in agents
        }

        for turn in transcript:
            evaluation = await self._evaluator.evaluate_argument(
                turn.argument, context
            )
            argument_evaluations[turn.agent].append(evaluation)

        # Compute agent scores
        agent_scores = self._compute_agent_scores(argument_evaluations)

        # Apply perspective weighting
        weighted_scores = self._apply_perspective_weighting(agent_scores)

        # Handle empty transcript case
        if not weighted_scores:
            return JurorEvaluation(
                juror_id=self.juror_id,
                perspective=self.perspective,
                agent_scores={},
                criterion_scores={},
                winner="",
                confidence=0.0,
                reasoning="No arguments to evaluate.",
            )

        # Determine winner
        winner = max(weighted_scores, key=weighted_scores.get)
        score_spread = max(weighted_scores.values()) - min(weighted_scores.values())

        # Generate reasoning using LLM
        reasoning = await self._generate_reasoning(
            transcript, context, weighted_scores, winner
        )

        # Calculate confidence based on score spread
        confidence = min(1.0, 0.5 + score_spread)

        return JurorEvaluation(
            juror_id=self.juror_id,
            perspective=self.perspective,
            agent_scores=weighted_scores,
            criterion_scores=self._compute_criterion_scores(argument_evaluations),
            winner=winner,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _compute_agent_scores(
        self,
        evaluations: dict[str, list[ArgumentEvaluation]],
    ) -> dict[str, float]:
        """Compute overall scores for each agent."""
        scores: dict[str, float] = {}

        for agent, evals in evaluations.items():
            if not evals:
                scores[agent] = 0.0
                continue

            # Average of total scores
            scores[agent] = sum(e.total_score for e in evals) / len(evals)

        return scores

    def _compute_criterion_scores(
        self,
        evaluations: dict[str, list[ArgumentEvaluation]],
    ) -> dict[str, dict[str, float]]:
        """Compute criterion-level scores for each agent."""
        result: dict[str, dict[str, float]] = {}

        for agent, evals in evaluations.items():
            if not evals:
                result[agent] = {}
                continue

            # Average scores by criterion
            criterion_totals: dict[str, float] = {}
            for evaluation in evals:
                for criterion, score in evaluation.scores.items():
                    if criterion not in criterion_totals:
                        criterion_totals[criterion] = 0.0
                    criterion_totals[criterion] += score

            result[agent] = {
                criterion: total / len(evals)
                for criterion, total in criterion_totals.items()
            }

        return result

    def _apply_perspective_weighting(
        self,
        scores: dict[str, float],
    ) -> dict[str, float]:
        """
        Apply perspective-based weighting to scores.

        Different perspectives emphasize different criteria.
        """
        # For now, return unweighted scores
        # Could be enhanced to weight by perspective focus
        return scores

    async def _generate_reasoning(
        self,
        transcript: list[Turn],
        context: DebateContext,
        scores: dict[str, float],
        winner: str,
    ) -> str:
        """Generate reasoning explanation using LLM."""
        # Build prompt for reasoning
        perspective_prompt = self.PERSPECTIVE_PROMPTS.get(
            self.perspective,
            "You are a fair and balanced juror.",
        )

        # Summarize arguments
        summary_parts = []
        for turn in transcript[-6:]:  # Last 6 turns
            summary_parts.append(
                f"{turn.agent} ({turn.argument.level.value}): "
                f"{turn.argument.content[:200]}..."
            )
        argument_summary = "\n".join(summary_parts)

        system_prompt = f"""You are a debate juror evaluating the arguments.

{perspective_prompt}

Topic: {context.topic}

Provide a brief (2-3 sentence) explanation of why {winner} won this debate
from your perspective. Be specific about which arguments were most convincing."""

        user_prompt = f"""Recent arguments:

{argument_summary}

Final scores: {scores}

Why did {winner} win according to your evaluation perspective?"""

        try:
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]
            response = await self._model.generate(messages=messages, max_tokens=200)
            return response.content
        except Exception as e:
            logger.warning(
                "Failed to generate reasoning", juror_id=self.juror_id, error=str(e)
            )
            return f"{winner} demonstrated stronger arguments overall."

    def __repr__(self) -> str:
        return (
            f"JuryMember(id={self.juror_id!r}, "
            f"perspective={self.perspective.value!r})"
        )


class JuryPanel:
    """
    Multi-perspective jury panel for debate evaluation.

    Manages multiple jurors with different perspectives, coordinates
    their evaluations, and builds consensus verdicts.

    Example:
        >>> panel = JuryPanel(evaluators=3, model="gpt-4o")
        >>> verdict = await panel.deliberate(transcript, context)
        >>> print(f"Winner: {verdict.decision}")
    """

    def __init__(
        self,
        evaluators: int = 3,
        criteria: list[str] | None = None,
        model: str = "gpt-4o",
        consensus_threshold: float = 0.7,
        api_key: str | None = None,
        **model_kwargs: Any,
    ) -> None:
        """
        Initialize a jury panel.

        Args:
            evaluators: Number of jury members.
            criteria: Custom evaluation criteria.
            model: Model identifier for jurors.
            consensus_threshold: Required agreement for consensus (0-1).
            api_key: Optional API key for model provider.
            **model_kwargs: Additional model arguments.
        """
        self.criteria = criteria or DEFAULT_JURY_CRITERIA
        self.consensus_threshold = consensus_threshold

        # Create jurors with different perspectives
        self.jurors = [
            JuryMember(
                juror_id=f"juror_{i}",
                perspective=self._assign_perspective(i),
                model=model,
                criteria=self.criteria,
                api_key=api_key,
                **model_kwargs,
            )
            for i in range(evaluators)
        ]

        logger.debug(
            "JuryPanel initialized",
            evaluators=evaluators,
            perspectives=[j.perspective.value for j in self.jurors],
            consensus_threshold=consensus_threshold,
        )

    def _assign_perspective(self, index: int) -> JuryPerspective:
        """Assign diverse perspectives to jurors."""
        perspectives = list(JuryPerspective)
        return perspectives[index % len(perspectives)]

    async def deliberate(
        self,
        transcript: list[Turn],
        context: DebateContext,
    ) -> Verdict:
        """
        Conduct jury deliberation and reach a verdict.

        Args:
            transcript: Complete debate transcript.
            context: Debate context.

        Returns:
            Final Verdict with decision and reasoning.
        """
        logger.info(
            "Jury deliberation started",
            jurors=len(self.jurors),
            turns=len(transcript),
        )

        # Each juror evaluates independently (in parallel)
        evaluations = await asyncio.gather(
            *[juror.evaluate(transcript, context) for juror in self.jurors]
        )

        # Build consensus
        consensus = self._build_consensus(evaluations)

        # Calculate confidence
        confidence = self._calculate_confidence(evaluations, consensus)

        # Collect dissenting opinions
        dissents = self._collect_dissents(evaluations, consensus)

        # Aggregate scores
        score_breakdown = self._aggregate_scores(evaluations)

        # Generate final reasoning
        reasoning = self._generate_verdict_reasoning(
            evaluations, consensus, score_breakdown
        )

        verdict = Verdict(
            decision=consensus.decision,
            confidence=confidence,
            reasoning=reasoning,
            dissenting_opinions=dissents,
            score_breakdown=score_breakdown,
            unanimous=len(dissents) == 0,
        )

        logger.info(
            "Jury verdict reached",
            decision=verdict.decision,
            confidence=verdict.confidence,
            unanimous=verdict.unanimous,
            dissents=len(dissents),
        )

        return verdict

    def _build_consensus(
        self,
        evaluations: list[JurorEvaluation],
    ) -> ConsensusResult:
        """
        Build consensus from individual evaluations.

        Uses weighted voting based on juror confidence.
        """
        # Count votes weighted by confidence
        vote_scores: dict[str, float] = {}
        for evaluation in evaluations:
            winner = evaluation.winner
            if winner not in vote_scores:
                vote_scores[winner] = 0.0
            vote_scores[winner] += evaluation.confidence

        # Determine winner
        if not vote_scores:
            return ConsensusResult(
                decision="draw",
                agreement_score=0.0,
                supporting_jurors=[],
                dissenting_jurors=[],
                reasoning="No votes cast.",
            )

        decision = max(vote_scores, key=vote_scores.get)
        total_confidence = sum(e.confidence for e in evaluations)

        # Calculate agreement score
        agreement_score = (
            vote_scores[decision] / total_confidence if total_confidence > 0 else 0.0
        )

        # Identify supporting and dissenting jurors
        supporting = [e.juror_id for e in evaluations if e.winner == decision]
        dissenting = [e.juror_id for e in evaluations if e.winner != decision]

        # Check for draw conditions
        if agreement_score < self.consensus_threshold:
            # Check if it's a close call
            sorted_scores = sorted(vote_scores.values(), reverse=True)
            if len(sorted_scores) >= 2:
                margin = sorted_scores[0] - sorted_scores[1]
                if margin < 0.1:  # Very close
                    decision = "draw"
                    agreement_score = 1.0 - margin

        return ConsensusResult(
            decision=decision,
            agreement_score=agreement_score,
            supporting_jurors=supporting,
            dissenting_jurors=dissenting,
            reasoning=f"{len(supporting)} of {len(evaluations)} jurors support this decision.",
        )

    def _calculate_confidence(
        self,
        evaluations: list[JurorEvaluation],
        consensus: ConsensusResult,
    ) -> float:
        """Calculate overall confidence in the verdict."""
        if not evaluations:
            return 0.0

        # Base confidence from agreement
        base_confidence = consensus.agreement_score

        # Adjust by average juror confidence
        avg_juror_confidence = sum(e.confidence for e in evaluations) / len(evaluations)

        # Combine: 60% agreement, 40% individual confidence
        combined = 0.6 * base_confidence + 0.4 * avg_juror_confidence

        return min(1.0, combined)

    def _collect_dissents(
        self,
        evaluations: list[JurorEvaluation],
        consensus: ConsensusResult,
    ) -> list[DissentingOpinion]:
        """Collect dissenting opinions from jurors who disagreed."""
        dissents: list[DissentingOpinion] = []

        for evaluation in evaluations:
            if evaluation.winner != consensus.decision:
                # Calculate score deviation
                consensus_scores = [
                    e.agent_scores.get(consensus.decision, 0)
                    for e in evaluations
                ]
                avg_consensus_score = (
                    sum(consensus_scores) / len(consensus_scores)
                    if consensus_scores
                    else 0
                )
                juror_consensus_score = evaluation.agent_scores.get(
                    consensus.decision, 0
                )
                deviation = avg_consensus_score - juror_consensus_score

                dissents.append(
                    DissentingOpinion(
                        juror_id=evaluation.juror_id,
                        perspective=evaluation.perspective,
                        position=evaluation.winner,
                        reasoning=evaluation.reasoning,
                        score_deviation=deviation,
                    )
                )

        return dissents

    def _aggregate_scores(
        self,
        evaluations: list[JurorEvaluation],
    ) -> dict[str, float]:
        """Aggregate scores across all jurors."""
        if not evaluations:
            return {}

        # Collect all agents
        all_agents: set[str] = set()
        for evaluation in evaluations:
            all_agents.update(evaluation.agent_scores.keys())

        # Average scores across jurors
        aggregated: dict[str, float] = {}
        for agent in all_agents:
            scores = [
                e.agent_scores.get(agent, 0)
                for e in evaluations
                if agent in e.agent_scores
            ]
            if scores:
                aggregated[agent] = sum(scores) / len(scores)

        return aggregated

    def _generate_verdict_reasoning(
        self,
        evaluations: list[JurorEvaluation],
        consensus: ConsensusResult,
        scores: dict[str, float],
    ) -> str:
        """Generate comprehensive verdict reasoning."""
        parts = []

        # Overall decision
        if consensus.decision == "draw":
            parts.append("The jury has reached a draw verdict.")
        else:
            parts.append(f"The jury has decided in favor of {consensus.decision}.")

        # Score summary
        if scores:
            score_summary = ", ".join(
                f"{agent}: {score:.2f}" for agent, score in sorted(scores.items())
            )
            parts.append(f"Final scores: {score_summary}.")

        # Agreement level
        parts.append(
            f"Agreement level: {consensus.agreement_score:.0%} "
            f"({len(consensus.supporting_jurors)} supporting, "
            f"{len(consensus.dissenting_jurors)} dissenting)."
        )

        # Key perspectives
        perspective_summary = []
        for evaluation in evaluations:
            if evaluation.winner == consensus.decision:
                perspective_summary.append(
                    f"The {evaluation.perspective.value} perspective "
                    f"(confidence: {evaluation.confidence:.0%})"
                )

        if perspective_summary:
            parts.append(
                f"Supporting perspectives: {', '.join(perspective_summary[:2])}."
            )

        return " ".join(parts)

    def get_juror(self, juror_id: str) -> JuryMember | None:
        """Get a specific juror by ID."""
        for juror in self.jurors:
            if juror.juror_id == juror_id:
                return juror
        return None

    def get_perspectives(self) -> list[JuryPerspective]:
        """Get list of perspectives represented in the panel."""
        return [juror.perspective for juror in self.jurors]

    def __len__(self) -> int:
        return len(self.jurors)

    def __repr__(self) -> str:
        return (
            f"JuryPanel(jurors={len(self.jurors)}, "
            f"threshold={self.consensus_threshold})"
        )


@dataclass
class JuryConfig:
    """Configuration for jury panel creation."""

    evaluators: int = 3
    """Number of jury members."""
    model: str = "gpt-4o"
    """Model to use for jurors."""
    consensus_threshold: float = 0.7
    """Required agreement for consensus."""
    criteria: list[str] = field(default_factory=lambda: DEFAULT_JURY_CRITERIA)
    """Evaluation criteria."""
    require_reasoning: bool = True
    """Whether to generate detailed reasoning."""
