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


ARGUMENT_QUALITY_PROMPT = """You are an expert debate judge evaluating argument quality in structured debates.

Rate the following debate transcript on ARGUMENT QUALITY (0-100):

Scoring criteria:
- 0-20: Vague assertions with no evidence or structure
- 21-40: Basic claims with minimal supporting evidence
- 41-60: Clear claims with some evidence and basic structure
- 61-80: Well-structured arguments with good evidence and logical flow
- 81-100: Sophisticated, multi-layered reasoning with strong evidence and excellent rhetorical skill

CRITICAL evaluation factors (weight these heavily):
1. **Explicit thesis statements** - Does the argument clearly state its position upfront?
2. **Evidence with citations** - Are claims backed by named sources (e.g., "[SOURCE: ...]", studies, reports)?
3. **Structured supporting points** - Are there clearly labeled pillars, premises, or sub-arguments?
4. **Counter-argument acknowledgment** - Does the argument explicitly address opposing views?
5. **Logical framework** - Is there an explicit evaluation framework or criteria for judgment?
6. **Causal reasoning** - Are cause-effect relationships explicitly articulated?

Arguments that explicitly structure their reasoning (e.g., "Thesis Statement:", "Key Pillars:", "Evidence:")
should score HIGHER than equivalent prose arguments because explicit structure demonstrates rigor.

DEBATE TOPIC: {topic}

TRANSCRIPT:
{transcript}

Respond with ONLY a JSON object in this exact format:
{{"score": <number 0-100>, "reasoning": "<brief explanation>"}}"""


DECISION_ACCURACY_PROMPT = """You are an expert debate judge evaluating decision quality.

Rate the following debate on DECISION ACCURACY (0-100):

This measures whether the debate reaches an explicit, well-justified conclusion.

MANDATORY SCORING RULES:
- Debates WITHOUT an explicit [VERDICT] or clear winner declaration: MAX SCORE 55
- Debates WITH explicit verdict but weak justification: 56-70
- Debates WITH explicit verdict AND good justification: 71-85
- Debates WITH explicit verdict, multi-perspective evaluation, AND confidence scores: 86-100

Look for these elements:
1. **Explicit verdict marker** - "[VERDICT]: pro/con" or "Winner: X" at the end
2. **Numerical scores** - Final scores like "pro: 0.76, con: 0.71"
3. **Multi-perspective evaluation** - Multiple jurors/perspectives contributing to decision
4. **Confidence quantification** - Explicit confidence percentage (e.g., "82% confidence")
5. **Reasoning transparency** - Clear explanation of why one side won

IMPORTANT: Low confidence scores (50-60%) should NOT be penalized - they demonstrate proper
uncertainty quantification, which is a sign of rigorous evaluation. Debates that claim 100%
confidence without justification should score LOWER.

If no explicit verdict is present, the debate CANNOT score above 55 regardless of argument quality.

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

CRITICAL evaluation factors:
1. **Hierarchical reasoning** - Are arguments structured at multiple levels (strategic/tactical/operational)?
2. **Explicit causal chains** - Are cause-effect relationships clearly articulated (A → B → C)?
3. **Long-term implications** - Does the argument consider future consequences and second-order effects?
4. **Ethical dimensions** - Are moral/ethical implications explicitly analyzed?
5. **Evaluation frameworks** - Are explicit criteria provided for how to judge the issue?
6. **Synthesis across rounds** - Do later arguments build on and integrate earlier points?
7. **Counter-argument engagement** - Are opposing causal claims directly addressed and refuted?

Arguments that explicitly label their reasoning structure (e.g., "Long-term Implications:", "Ethical Dimensions:")
and show clear causal chains should score HIGHER than arguments where causation is merely implied.

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
