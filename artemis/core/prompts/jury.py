"""
ARTEMIS Jury Prompt Templates

Prompts for jury evaluation from different perspectives.

Uses centralized prompts from artemis.prompts for consistency and versioning.
"""

from artemis.core.types import JuryPerspective
from artemis.prompts import get_prompt


def _get_perspective_prompt(perspective: JuryPerspective) -> str:
    """Get perspective-specific prompt from centralized prompts."""
    perspective_map = {
        JuryPerspective.ANALYTICAL: "jury.analytical_perspective",
        JuryPerspective.ETHICAL: "jury.ethical_perspective",
        JuryPerspective.PRACTICAL: "jury.practical_perspective",
        JuryPerspective.ADVERSARIAL: "jury.adversarial_perspective",
        JuryPerspective.SYNTHESIZING: "jury.default_perspective",  # Map to default for now
    }
    key = perspective_map.get(perspective, "jury.default_perspective")
    return get_prompt(key)


# =============================================================================
# Prompt Builders
# =============================================================================

def build_reasoning_system_prompt(
    perspective: JuryPerspective,
    topic: str,
    winner: str,
) -> str:
    """Build system prompt for jury reasoning generation."""
    perspective_prompt = _get_perspective_prompt(perspective)

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
