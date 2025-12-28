"""
ARTEMIS Jury Prompt Templates

Prompts for jury evaluation from different perspectives.
"""

from artemis.core.types import JuryPerspective

# =============================================================================
# Perspective-Specific Prompts
# =============================================================================

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

DEFAULT_PERSPECTIVE_PROMPT = "You are a fair and balanced juror."


# =============================================================================
# Prompt Builders
# =============================================================================

def build_reasoning_system_prompt(
    perspective: JuryPerspective,
    topic: str,
    winner: str,
) -> str:
    """Build system prompt for jury reasoning generation."""
    perspective_prompt = PERSPECTIVE_PROMPTS.get(
        perspective,
        DEFAULT_PERSPECTIVE_PROMPT,
    )

    return f"""You are a debate juror evaluating the arguments.

{perspective_prompt}

Topic: {topic}

Provide a brief (2-3 sentence) explanation of why {winner} won this debate
from your perspective. Be specific about which arguments were most convincing."""


def build_reasoning_user_prompt(
    argument_summary: str,
    scores: dict,
    winner: str,
) -> str:
    """Build user prompt for jury reasoning generation."""
    return f"""Recent arguments:

{argument_summary}

Final scores: {scores}

Why did {winner} win according to your evaluation perspective?"""
