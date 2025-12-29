"""
ARTEMIS Prompt Management System

Centralized, versioned prompts for all ARTEMIS components.
Prompts are organized by version (v1, v2, etc.) for A/B testing and rollback.

Usage:
    from artemis.prompts import get_prompt, list_prompts

    # Get a specific prompt
    prompt = get_prompt("hdag.strategic_instructions")

    # Get prompt from specific version
    prompt = get_prompt("hdag.strategic_instructions", version="v1")

    # List all available prompts
    prompts = list_prompts()
"""

from artemis.prompts.loader import get_prompt, get_prompt_version, list_prompts

__all__ = ["get_prompt", "list_prompts", "get_prompt_version"]
