"""LLM-as-judge evaluator for debate benchmarks."""

import json
import os
from dataclasses import dataclass

from openai import AsyncOpenAI

from artemis.prompts import get_prompt
from benchmarks.adapters.base import DebateResult


@dataclass
class EvaluationScores:
    """Scores from debate evaluation."""

    argument_quality: float  # 0-100
    decision_accuracy: float  # 0-100
    reasoning_depth: float  # 0-100
    raw_responses: dict | None = None

    @property
    def average(self) -> float:
        """Calculate average score."""
        return (self.argument_quality + self.decision_accuracy + self.reasoning_depth) / 3


class DebateEvaluator:
    """Evaluates debate quality using LLM-as-judge."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def evaluate(self, result: DebateResult) -> EvaluationScores:
        """
        Evaluate a debate result on all metrics.

        Args:
            result: The debate result to evaluate.

        Returns:
            EvaluationScores with scores for each metric.
        """
        transcript = result.full_transcript
        topic = result.topic

        # Get prompts from centralized store
        arg_quality_prompt = get_prompt("benchmark.argument_quality")
        decision_acc_prompt = get_prompt("benchmark.decision_accuracy")
        reasoning_depth_prompt = get_prompt("benchmark.reasoning_depth")

        # Run all evaluations
        arg_quality = await self._evaluate_metric(
            arg_quality_prompt, topic, transcript
        )
        decision_acc = await self._evaluate_metric(
            decision_acc_prompt, topic, transcript
        )
        reasoning_depth = await self._evaluate_metric(
            reasoning_depth_prompt, topic, transcript
        )

        return EvaluationScores(
            argument_quality=arg_quality["score"],
            decision_accuracy=decision_acc["score"],
            reasoning_depth=reasoning_depth["score"],
            raw_responses={
                "argument_quality": arg_quality,
                "decision_accuracy": decision_acc,
                "reasoning_depth": reasoning_depth,
            },
        )

    async def _evaluate_metric(
        self,
        prompt_template: str,
        topic: str,
        transcript: str,
    ) -> dict:
        """Evaluate a single metric."""
        prompt = prompt_template.format(topic=topic, transcript=transcript)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert debate judge. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,  # Deterministic for consistency
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)

            # Ensure score is in valid range
            result["score"] = max(0, min(100, float(result.get("score", 50))))

            return result

        except Exception as e:
            # Return middle score on error
            return {
                "score": 50.0,
                "reasoning": f"Evaluation error: {str(e)}",
                "error": True,
            }


async def evaluate_debate(
    result: DebateResult,
    model: str = "gpt-4o",
) -> EvaluationScores:
    """
    Convenience function to evaluate a debate.

    Args:
        result: The debate result to evaluate.
        model: Model to use for evaluation.

    Returns:
        EvaluationScores with all metrics.
    """
    evaluator = DebateEvaluator(model=model)
    return await evaluator.evaluate(result)
