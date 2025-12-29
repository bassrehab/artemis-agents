"""
ARTEMIS Prompt Templates

Prompt templates for argument generation, evaluation, and debate management.

Note: Raw prompts are now centralized in artemis.prompts. This module provides
prompt builder functions that use the centralized prompts.
"""

from artemis.core.prompts.hdag import (
    build_closing_prompt,
    build_context_prompt,
    build_generation_prompt,
    build_opening_prompt,
    build_rebuttal_prompt,
    build_system_prompt,
)
from artemis.core.prompts.jury import (
    build_reasoning_system_prompt,
    build_reasoning_user_prompt,
)

__all__ = [
    # H-L-DAG prompts
    "build_system_prompt",
    "build_context_prompt",
    "build_generation_prompt",
    "build_opening_prompt",
    "build_closing_prompt",
    "build_rebuttal_prompt",
    # Jury prompts
    "build_reasoning_system_prompt",
    "build_reasoning_user_prompt",
]
