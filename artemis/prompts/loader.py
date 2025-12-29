"""
Prompt loader for ARTEMIS.

Loads prompts from versioned modules with caching and validation.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

# Default prompt version
DEFAULT_VERSION = "v1"

# Registry of prompt modules per version
PROMPT_MODULES = {
    "v1": {
        "hdag": "artemis.prompts.v1.hdag",
        "evaluation": "artemis.prompts.v1.evaluation",
        "extraction": "artemis.prompts.v1.extraction",
        "jury": "artemis.prompts.v1.jury",
        "benchmark": "artemis.prompts.v1.benchmark",
    },
}


@lru_cache(maxsize=128)
def _load_module(module_path: str):
    """Load and cache a prompt module."""
    return importlib.import_module(module_path)


def get_prompt(
    prompt_key: str,
    version: str = DEFAULT_VERSION,
    **format_kwargs: Any,
) -> str:
    """
    Get a prompt by key with optional formatting.

    Args:
        prompt_key: Dot-separated key like "hdag.strategic_instructions"
        version: Prompt version (default: v1)
        **format_kwargs: Variables to format into the prompt

    Returns:
        The prompt string, optionally formatted

    Raises:
        KeyError: If prompt not found
        ValueError: If version not found

    Example:
        >>> get_prompt("hdag.strategic_instructions")
        "## Strategic Level Argument..."

        >>> get_prompt("evaluation.user", topic="AI Safety", level="strategic")
        "Evaluate the following argument..."
    """
    if version not in PROMPT_MODULES:
        raise ValueError(f"Unknown prompt version: {version}. Available: {list(PROMPT_MODULES.keys())}")

    parts = prompt_key.split(".", 1)
    if len(parts) != 2:
        raise KeyError(f"Invalid prompt key format: {prompt_key}. Use 'module.prompt_name'")

    module_name, prompt_name = parts

    if module_name not in PROMPT_MODULES[version]:
        raise KeyError(f"Unknown prompt module: {module_name}. Available: {list(PROMPT_MODULES[version].keys())}")

    module_path = PROMPT_MODULES[version][module_name]
    module = _load_module(module_path)

    # Get the prompt constant (uppercase)
    const_name = prompt_name.upper()
    if not hasattr(module, const_name):
        # Try original case
        if not hasattr(module, prompt_name):
            available = [n for n in dir(module) if not n.startswith("_") and n.isupper()]
            raise KeyError(f"Prompt '{prompt_name}' not found in {module_name}. Available: {available}")
        prompt = getattr(module, prompt_name)
    else:
        prompt = getattr(module, const_name)

    # Format if kwargs provided
    if format_kwargs:
        prompt = prompt.format(**format_kwargs)

    return prompt


def list_prompts(version: str = DEFAULT_VERSION) -> dict[str, list[str]]:
    """
    List all available prompts for a version.

    Args:
        version: Prompt version

    Returns:
        Dict mapping module names to list of prompt names
    """
    if version not in PROMPT_MODULES:
        raise ValueError(f"Unknown prompt version: {version}")

    result = {}
    for module_name, module_path in PROMPT_MODULES[version].items():
        try:
            module = _load_module(module_path)
            prompts = [n for n in dir(module) if n.isupper() and not n.startswith("_")]
            result[module_name] = prompts
        except ImportError:
            result[module_name] = []

    return result


def get_prompt_version() -> str:
    """Get the current default prompt version."""
    return DEFAULT_VERSION


def set_prompt_version(version: str) -> None:
    """
    Set the default prompt version globally.

    Args:
        version: Version string (e.g., "v1", "v2")
    """
    global DEFAULT_VERSION
    if version not in PROMPT_MODULES:
        raise ValueError(f"Unknown prompt version: {version}")
    DEFAULT_VERSION = version
