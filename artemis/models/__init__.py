"""
ARTEMIS Models Module

LLM provider integrations with unified interface.
Supports reasoning models (o1, R1, Gemini 2.5) with extended thinking.
"""

from typing import Any

from artemis.models.base import BaseModel, ModelRegistry
from artemis.models.deepseek import DeepSeekModel
from artemis.models.openai import OpenAIModel
from artemis.models.reasoning import (
    MODEL_CAPABILITIES,
    ReasoningCapabilities,
    ReasoningConfig,
    ReasoningModel,
    ReasoningPromptBuilder,
    ReasoningStrategy,
    ThinkingBudget,
    calculate_thinking_budget,
    create_reasoning_config,
    get_default_thinking_budget,
    get_model_capabilities,
    is_reasoning_model,
)

# Lazy imports for providers with optional dependencies
_lazy_imports = {
    "AnthropicModel": "artemis.models.anthropic",
    "GoogleModel": "artemis.models.google",
}


def __getattr__(name: str):
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "DeepSeekModel",
    "AnthropicModel",
    "GoogleModel",
    # Reasoning Configuration
    "ReasoningModel",
    "ReasoningStrategy",
    "ReasoningConfig",
    "ReasoningCapabilities",
    "ThinkingBudget",
    "ReasoningPromptBuilder",
    "MODEL_CAPABILITIES",
    "get_model_capabilities",
    "is_reasoning_model",
    "get_default_thinking_budget",
    "calculate_thinking_budget",
    "create_reasoning_config",
    # Factory
    "create_model",
    "list_providers",
]
