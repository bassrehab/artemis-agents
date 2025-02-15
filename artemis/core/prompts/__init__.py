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

__all__ = [
    "LEVEL_INSTRUCTIONS",
    "build_system_prompt",
    "build_context_prompt",
    "build_generation_prompt",
    "build_opening_prompt",
    "build_closing_prompt",
    "build_rebuttal_prompt",
]
