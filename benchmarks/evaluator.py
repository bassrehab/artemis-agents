"""LLM-as-judge evaluator for debate benchmarks."""

import json
import os
from dataclasses import dataclass

from openai import AsyncOpenAI

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


ARGUMENT_QUALITY_PROMPT = """You are an expert debate judge evaluating argument quality.

Rate the following debate transcript on ARGUMENT QUALITY (0-100):

Scoring criteria:
- 0-20: Vague assertions with no evidence or structure
- 21-40: Basic claims with minimal supporting evidence
- 41-60: Clear claims with some evidence and basic structure
- 61-80: Well-structured arguments with good evidence and logical flow
- 81-100: Sophisticated, multi-layered reasoning with strong evidence and excellent rhetorical skill

Consider:
1. Clarity and specificity of claims
2. Quality and relevance of evidence
3. Logical structure (premises lead to conclusions)
4. Acknowledgment and handling of counterarguments
5. Persuasiveness and rhetorical effectiveness

DEBATE TOPIC: {topic}

TRANSCRIPT:
{transcript}

Respond with ONLY a JSON object in this exact format:
{{"score": <number 0-100>, "reasoning": "<brief explanation>"}}"""


DECISION_ACCURACY_PROMPT = """You are an expert debate judge evaluating decision quality.

Rate the following debate on DECISION ACCURACY (0-100):

This measures how well the debate process reaches a justified, well-reasoned conclusion.

Scoring criteria:
- 0-20: No clear conclusion or completely unjustified verdict
- 21-40: Weak conclusion with poor justification
- 41-60: Reasonable conclusion but justification could be stronger
- 61-80: Good conclusion with solid justification
- 81-100: Excellent conclusion with comprehensive justification that weighs all arguments

Consider:
1. Does the conclusion align with the strength of arguments presented?
2. Is the reasoning for the conclusion transparent?
3. Are edge cases and nuances acknowledged?
4. Is uncertainty appropriately expressed?
5. Would a neutral observer agree with the assessment?

DEBATE TOPIC: {topic}

TRANSCRIPT:
{transcript}

Respond with ONLY a JSON object in this exact format:
{{"score": <number 0-100>, "reasoning": "<brief explanation>"}}"""


REASONING_DEPTH_PROMPT = """You are an expert debate judge evaluating reasoning depth.

Rate the following debate on REASONING DEPTH (0-100):

This measures the sophistication of causal reasoning and argument interconnection.

Scoring criteria:
- 0-20: Surface-level reasoning with no causal analysis
- 21-40: Basic cause-effect statements without deeper analysis
- 41-60: Some causal chains identified but limited depth
- 61-80: Good causal reasoning with multi-step analysis
- 81-100: Sophisticated reasoning with complex causal chains, second-order effects, and synthesis across arguments

Consider:
1. Identification of causal relationships
2. Multi-step reasoning chains
3. Consideration of second and third-order effects
4. Synthesis and integration across different arguments
5. Recognition of complexity and interdependencies

DEBATE TOPIC: {topic}

TRANSCRIPT:
{transcript}

Respond with ONLY a JSON object in this exact format:
{{"score": <number 0-100>, "reasoning": "<brief explanation>"}}"""


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

        # Run all evaluations
        arg_quality = await self._evaluate_metric(
            ARGUMENT_QUALITY_PROMPT, topic, transcript
        )
        decision_acc = await self._evaluate_metric(
            DECISION_ACCURACY_PROMPT, topic, transcript
        )
        reasoning_depth = await self._evaluate_metric(
            REASONING_DEPTH_PROMPT, topic, transcript
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
