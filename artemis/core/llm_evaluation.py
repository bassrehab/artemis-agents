"""LLM-native evaluation for ARTEMIS debates.

Uses LLM judgment instead of heuristics for evaluating argument quality.
This aligns the internal evaluation function with how benchmarks judge debates.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel as PydanticBaseModel

from artemis.core.types import (
    Argument,
    ArgumentEvaluation,
    CriterionScore,
    DebateContext,
    EvaluationMode,
    Message,
)
from artemis.utils.logging import get_logger

if TYPE_CHECKING:
    from artemis.models.base import BaseModel

logger = get_logger(__name__)


class CriterionResult(PydanticBaseModel):
    """Result for a single evaluation criterion."""

    score: float  # 0-100 scale, will be normalized to 0-1
    reasoning: str


class LLMEvaluationResult(PydanticBaseModel):
    """Structured result from LLM evaluation."""

    logical_coherence: CriterionResult
    evidence_quality: CriterionResult
    causal_reasoning: CriterionResult
    ethical_alignment: CriterionResult
    persuasiveness: CriterionResult
    overall_assessment: str


# Prompt for LLM-based evaluation
EVALUATION_SYSTEM_PROMPT = """You are an expert debate evaluator. Your task is to assess arguments on multiple dimensions with precise, objective scoring.

For each criterion, provide:
1. A score from 0-100 (be discriminating - use the full range)
2. A brief reasoning (1-2 sentences) explaining the score

Scoring guidelines:
- 0-20: Very poor, major flaws
- 21-40: Below average, notable weaknesses
- 41-60: Average, meets basic expectations
- 61-80: Good, clear strengths
- 81-100: Excellent, exceptional quality

Be objective and consistent. Don't inflate scores."""

EVALUATION_USER_PROMPT = """Evaluate the following argument from a debate on: "{topic}"

**Argument** (Level: {level}):
{content}

**Context**:
- Round {round} of {total_rounds}
- Agent position: {position}
- Previous arguments in debate: {prev_count}

Evaluate on these criteria:

1. **Logical Coherence**: Does the argument have clear premises that lead to valid conclusions? Are there logical fallacies?

2. **Evidence Quality**: Does the argument cite credible sources, statistics, or examples? Is evidence relevant and well-integrated?

3. **Causal Reasoning**: Does the argument establish clear cause-effect relationships? Are causal claims well-supported?

4. **Ethical Alignment**: Does the argument consider ethical implications? Does it avoid manipulation or misleading rhetoric?

5. **Persuasiveness**: How compelling is the argument? Does it effectively address the audience and counter opposing views?

Respond with a JSON object in this exact format:
{{
    "logical_coherence": {{"score": <0-100>, "reasoning": "<brief explanation>"}},
    "evidence_quality": {{"score": <0-100>, "reasoning": "<brief explanation>"}},
    "causal_reasoning": {{"score": <0-100>, "reasoning": "<brief explanation>"}},
    "ethical_alignment": {{"score": <0-100>, "reasoning": "<brief explanation>"}},
    "persuasiveness": {{"score": <0-100>, "reasoning": "<brief explanation>"}},
    "overall_assessment": "<1-2 sentence summary of argument quality>"
}}"""


class LLMCriterionEvaluator:
    """Evaluates arguments using LLM judgment.

    This evaluator uses an LLM to assess each criterion, providing
    scores that align with how human judges (and benchmark evaluators)
    assess argument quality.
    """

    def __init__(
        self,
        model: str | BaseModel = "gpt-4o-mini",
        api_key: str | None = None,
        cache_enabled: bool = True,
        **model_kwargs: Any,
    ) -> None:
        """Initialize the LLM evaluator.

        Args:
            model: Model name or instance to use for evaluation
            api_key: Optional API key
            cache_enabled: Whether to cache evaluations by argument hash
            **model_kwargs: Additional model configuration
        """
        from artemis.models.base import BaseModel as ModelBase
        from artemis.models.base import ModelRegistry

        if isinstance(model, ModelBase):
            self._model = model
        else:
            self._model = ModelRegistry.create(model, api_key=api_key, **model_kwargs)

        self._cache_enabled = cache_enabled
        self._cache: dict[str, ArgumentEvaluation] = {}

        logger.debug(
            "LLMCriterionEvaluator initialized",
            model=self._model.model,
            cache_enabled=cache_enabled,
        )

    def _get_cache_key(self, argument: Argument, context: DebateContext) -> str:
        """Generate cache key from argument content and context."""
        key_data = f"{argument.content}:{context.topic}:{context.current_round}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def evaluate_argument(
        self,
        argument: Argument,
        context: DebateContext,
        weights: dict[str, float] | None = None,
    ) -> ArgumentEvaluation:
        """Evaluate an argument using LLM judgment.

        Args:
            argument: The argument to evaluate
            context: Current debate context
            weights: Optional criterion weights (defaults to equal)

        Returns:
            ArgumentEvaluation with LLM-generated scores and reasoning
        """
        # Check cache first
        if self._cache_enabled:
            cache_key = self._get_cache_key(argument, context)
            if cache_key in self._cache:
                logger.debug("Cache hit for argument evaluation", argument_id=argument.id)
                return self._cache[cache_key]

        # Build evaluation prompt
        position = context.agent_positions.get(argument.agent, "unknown")
        prev_count = len(context.transcript)

        user_prompt = EVALUATION_USER_PROMPT.format(
            topic=context.topic,
            level=argument.level.value,
            content=argument.content[:2000],  # Truncate very long arguments
            round=context.current_round,
            total_rounds=context.total_rounds,
            position=position,
            prev_count=prev_count,
        )

        messages = [
            Message(role="system", content=EVALUATION_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]

        try:
            response = await self._model.generate(messages=messages, max_tokens=800)
            result = self._parse_response(response.content)
        except Exception as e:
            logger.warning(
                "LLM evaluation failed, using fallback scores",
                error=str(e),
                argument_id=argument.id,
            )
            result = self._fallback_evaluation()

        # Convert to ArgumentEvaluation
        default_weights = {
            "logical_coherence": 0.25,
            "evidence_quality": 0.25,
            "causal_reasoning": 0.20,
            "ethical_alignment": 0.15,
            "persuasiveness": 0.15,
        }
        weights = weights or default_weights

        # Normalize scores from 0-100 to 0-1
        scores = {
            "logical_coherence": result.logical_coherence.score / 100,
            "evidence_quality": result.evidence_quality.score / 100,
            "causal_reasoning": result.causal_reasoning.score / 100,
            "ethical_alignment": result.ethical_alignment.score / 100,
            "persuasiveness": result.persuasiveness.score / 100,
        }

        # Compute weighted total
        total_score = sum(scores[k] * weights[k] for k in scores)

        # Build criterion details
        criterion_details = [
            CriterionScore(
                criterion="logical_coherence",
                score=scores["logical_coherence"],
                weight=weights["logical_coherence"],
                reasoning=result.logical_coherence.reasoning,
            ),
            CriterionScore(
                criterion="evidence_quality",
                score=scores["evidence_quality"],
                weight=weights["evidence_quality"],
                reasoning=result.evidence_quality.reasoning,
            ),
            CriterionScore(
                criterion="causal_reasoning",
                score=scores["causal_reasoning"],
                weight=weights["causal_reasoning"],
                reasoning=result.causal_reasoning.reasoning,
            ),
            CriterionScore(
                criterion="ethical_alignment",
                score=scores["ethical_alignment"],
                weight=weights["ethical_alignment"],
                reasoning=result.ethical_alignment.reasoning,
            ),
            CriterionScore(
                criterion="persuasiveness",
                score=scores["persuasiveness"],
                weight=weights["persuasiveness"],
                reasoning=result.persuasiveness.reasoning,
            ),
        ]

        evaluation = ArgumentEvaluation(
            argument_id=argument.id,
            scores=scores,
            weights=weights,
            criterion_details=criterion_details,
            causal_score=scores["causal_reasoning"],
            total_score=total_score,
            evaluator_notes=result.overall_assessment,
        )

        # Cache the result
        if self._cache_enabled:
            self._cache[cache_key] = evaluation

        logger.debug(
            "LLM evaluation complete",
            argument_id=argument.id,
            total_score=total_score,
        )

        return evaluation

    def _parse_response(self, content: str) -> LLMEvaluationResult:
        """Parse LLM response into structured result."""
        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            return LLMEvaluationResult(
                logical_coherence=CriterionResult(**data["logical_coherence"]),
                evidence_quality=CriterionResult(**data["evidence_quality"]),
                causal_reasoning=CriterionResult(**data["causal_reasoning"]),
                ethical_alignment=CriterionResult(**data["ethical_alignment"]),
                persuasiveness=CriterionResult(**data["persuasiveness"]),
                overall_assessment=data.get("overall_assessment", ""),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse LLM response", error=str(e))
            return self._fallback_evaluation()

    def _fallback_evaluation(self) -> LLMEvaluationResult:
        """Return neutral scores when LLM fails."""
        neutral = CriterionResult(score=50.0, reasoning="Evaluation unavailable")
        return LLMEvaluationResult(
            logical_coherence=neutral,
            evidence_quality=neutral,
            causal_reasoning=neutral,
            ethical_alignment=neutral,
            persuasiveness=neutral,
            overall_assessment="Unable to evaluate argument.",
        )

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._cache.clear()


class EvaluatorFactory:
    """Factory for creating evaluators based on evaluation mode."""

    @staticmethod
    def create(
        mode: EvaluationMode,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """Create an evaluator based on the specified mode.

        Args:
            mode: The evaluation mode (QUALITY, BALANCED, FAST)
            model: Model to use for LLM evaluation
            api_key: Optional API key
            **kwargs: Additional configuration

        Returns:
            An evaluator instance appropriate for the mode
        """
        from artemis.core.evaluation import AdaptiveEvaluator

        if mode == EvaluationMode.QUALITY:
            # Full LLM evaluation
            return LLMCriterionEvaluator(model=model, api_key=api_key, **kwargs)
        elif mode == EvaluationMode.BALANCED:
            # Use LLM for final jury, heuristics for in-debate
            # Return heuristic evaluator for in-debate use
            # (Jury will use LLM separately)
            return AdaptiveEvaluator(**kwargs)
        else:
            # FAST mode: pure heuristics
            return AdaptiveEvaluator(**kwargs)
