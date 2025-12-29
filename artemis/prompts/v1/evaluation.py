"""
Evaluation Prompts - Version 1

Prompts for LLM-based argument evaluation using multi-criteria assessment.
"""

# =============================================================================
# LLM Evaluation Prompts
# =============================================================================

SYSTEM = """You are an expert debate evaluator. Your task is to assess arguments on multiple dimensions with precise, objective scoring.

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

USER = """Evaluate the following argument from a debate on: "{topic}"

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

# =============================================================================
# Criteria Definitions
# =============================================================================

CRITERIA_DEFINITIONS = {
    "logical_coherence": "Clear premises leading to valid conclusions without logical fallacies",
    "evidence_quality": "Credible sources, statistics, and examples that are relevant and well-integrated",
    "causal_reasoning": "Clear cause-effect relationships with well-supported causal claims",
    "ethical_alignment": "Consideration of ethical implications without manipulation or misleading rhetoric",
    "persuasiveness": "Compelling argument that effectively addresses audience and counters opposing views",
}

DEFAULT_WEIGHTS = {
    "logical_coherence": 0.25,
    "evidence_quality": 0.25,
    "causal_reasoning": 0.20,
    "ethical_alignment": 0.15,
    "persuasiveness": 0.15,
}
