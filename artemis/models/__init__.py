"""
ARTEMIS Models Module

LLM provider integrations with unified interface.
Supports reasoning models (o1, R1, Gemini 2.5) with extended thinking.
"""

from typing import Any

from artemis.models.base import BaseModel, ModelRegistry
from artemis.models.openai import OpenAIModel

# Future providers (uncomment when implemented):
# from artemis.models.anthropic import AnthropicModel
# from artemis.models.google import GoogleModel
# from artemis.models.deepseek import DeepSeekModel


def create_model(
    model: str,
    provider: str | None = None,
    **kwargs: Any,
) -> BaseModel:
    """
    Factory function to create a model instance.

    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-3-opus").
        provider: Optional provider name. If not specified, will be inferred
                  from the model name.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        A model instance.

    Raises:
        ValueError: If provider cannot be determined.
        KeyError: If provider is not registered.

    Example:
        >>> model = create_model("gpt-4o")
        >>> response = await model.generate([Message(role="user", content="Hello")])
    """
    return ModelRegistry.create(model=model, provider=provider, **kwargs)


def list_providers() -> list[str]:
    """List all registered model providers."""
    return ModelRegistry.list_providers()


__all__ = [
    # Base
    "BaseModel",
    "ModelRegistry",
    # Providers
    "OpenAIModel",
    # Factory
    "create_model",
    "list_providers",
]
