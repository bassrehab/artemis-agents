"""
ARTEMIS Prompt Templates

Prompt templates for argument generation, evaluation, and debate management.
"""

from artemis.core.prompts.hdag import (
    LEVEL_INSTRUCTIONS,
    build_closing_prompt,
    build_context_prompt,
    build_generation_prompt,
    build_opening_prompt,
    build_rebuttal_prompt,
    build_system_prompt,
)
from artemis.core.prompts.jury import (
    PERSPECTIVE_PROMPTS,
    build_reasoning_system_prompt,
    build_reasoning_user_prompt,
)

__all__ = [
    # H-L-DAG prompts
    "LEVEL_INSTRUCTIONS",
    "build_system_prompt",
    "build_context_prompt",
    "build_generation_prompt",
    "build_opening_prompt",
    "build_closing_prompt",
    "build_rebuttal_prompt",
    # Jury prompts
    "PERSPECTIVE_PROMPTS",
    "build_reasoning_system_prompt",
    "build_reasoning_user_prompt",
]
