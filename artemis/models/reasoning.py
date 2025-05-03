"""
ARTEMIS Reasoning Model Configuration

Configuration and utilities for reasoning models (o1, R1, Gemini 2.5 Pro)
that support extended thinking and chain-of-thought generation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from artemis.utils.logging import get_logger

logger = get_logger(__name__)


class ReasoningModel(str, Enum):
    """Enumeration of supported reasoning models."""

    # OpenAI o1 family
    O1 = "o1"
    O1_PREVIEW = "o1-preview"
    O1_MINI = "o1-mini"

    # DeepSeek R1
    DEEPSEEK_R1 = "deepseek-reasoner"
    DEEPSEEK_R1_DISTILL = "deepseek-r1-distill-llama-70b"

    # Google Gemini 2.5
    GEMINI_25_PRO = "gemini-2.5-pro"
    GEMINI_25_FLASH = "gemini-2.5-flash"

    # Anthropic Claude (extended thinking)
    CLAUDE_35_SONNET = "claude-3-5-sonnet-20241022"


class ReasoningStrategy(str, Enum):
    """Strategies for using reasoning capabilities."""

    ALWAYS = "always"
    """Always use extended thinking."""

    ADAPTIVE = "adaptive"
    """Use reasoning only for complex problems."""

    NEVER = "never"
    """Never use extended thinking, even if available."""


@dataclass
class ThinkingBudget:
    """Configuration for thinking token budgets."""

    min_tokens: int = 1000
    """Minimum thinking tokens."""

    max_tokens: int = 32000
    """Maximum thinking tokens."""

    default_tokens: int = 8000
    """Default thinking tokens when not specified."""

    scale_with_complexity: bool = True
    """Whether to scale budget based on problem complexity."""

    complexity_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "simple": 0.5,
            "moderate": 1.0,
            "complex": 2.0,
            "very_complex": 4.0,
        }
    )


class ReasoningConfig(BaseModel):
    """Configuration for reasoning model behavior."""

    model: str = Field(
        default="o1",
        description="The reasoning model to use.",
    )
    strategy: ReasoningStrategy = Field(
        default=ReasoningStrategy.ADAPTIVE,
        description="When to use extended thinking.",
    )
    thinking_budget: int = Field(
        default=8000,
        ge=1000,
        le=128000,
        description="Default token budget for thinking.",
    )
    show_thinking: bool = Field(
        default=False,
        description="Whether to return the thinking trace.",
    )
    thinking_style: str = Field(
        default="thorough",
        description="Style of thinking: 'thorough', 'concise', or 'analytical'.",
    )

    # Model-specific settings
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (some reasoning models require 1.0).",
    )
    use_system_prompt: bool = Field(
        default=True,
        description="Whether to use system prompts (not supported by all models).",
    )


@dataclass
class ReasoningCapabilities:
    """Describes the reasoning capabilities of a model."""

    supports_extended_thinking: bool = False
    """Whether the model supports extended thinking."""

    supports_thinking_budget: bool = False
    """Whether thinking budget can be configured."""

    supports_thinking_visibility: bool = False
    """Whether thinking trace can be shown."""

    supports_system_prompts: bool = True
    """Whether system prompts are supported."""

    requires_temperature_1: bool = False
    """Whether temperature must be 1.0."""

    max_thinking_tokens: int = 128000
    """Maximum supported thinking tokens."""

    default_thinking_tokens: int = 8000
    """Default thinking tokens."""


# Capability definitions for known reasoning models
MODEL_CAPABILITIES: dict[str, ReasoningCapabilities] = {
    # OpenAI o1
    "o1": ReasoningCapabilities(
        supports_extended_thinking=True,
        supports_thinking_budget=True,
        supports_thinking_visibility=False,  # o1 doesn't expose thinking
        supports_system_prompts=False,  # o1 uses developer messages
        requires_temperature_1=True,
        max_thinking_tokens=128000,
        default_thinking_tokens=16000,
    ),
    "o1-preview": ReasoningCapabilities(
        supports_extended_thinking=True,
        supports_thinking_budget=True,
        supports_thinking_visibility=False,
        supports_system_prompts=False,
        requires_temperature_1=True,
        max_thinking_tokens=32768,
        default_thinking_tokens=8000,
    ),
    "o1-mini": ReasoningCapabilities(
        supports_extended_thinking=True,
        supports_thinking_budget=True,
        supports_thinking_visibility=False,
        supports_system_prompts=False,
        requires_temperature_1=True,
        max_thinking_tokens=65536,
        default_thinking_tokens=4000,
    ),
    # DeepSeek R1
    "deepseek-reasoner": ReasoningCapabilities(
        supports_extended_thinking=True,
        supports_thinking_budget=True,
        supports_thinking_visibility=True,  # R1 shows thinking
        supports_system_prompts=True,
        requires_temperature_1=False,
        max_thinking_tokens=32768,
        default_thinking_tokens=8000,
    ),
    "deepseek-r1-distill-llama-70b": ReasoningCapabilities(
        supports_extended_thinking=True,
        supports_thinking_budget=False,  # Distilled version
        supports_thinking_visibility=True,
        supports_system_prompts=True,
        requires_temperature_1=False,
        max_thinking_tokens=16384,
        default_thinking_tokens=4000,
    ),
    # Gemini 2.5
    "gemini-2.5-pro": ReasoningCapabilities(
        supports_extended_thinking=True,
        supports_thinking_budget=True,
        supports_thinking_visibility=True,
        supports_system_prompts=True,
        requires_temperature_1=False,
        max_thinking_tokens=32768,
        default_thinking_tokens=8000,
    ),
    # Claude (extended thinking beta)
    "claude-3-5-sonnet-20241022": ReasoningCapabilities(
        supports_extended_thinking=True,
        supports_thinking_budget=True,
        supports_thinking_visibility=True,
        supports_system_prompts=True,
        requires_temperature_1=False,
        max_thinking_tokens=128000,
        default_thinking_tokens=8000,
    ),
}


def get_model_capabilities(model: str) -> ReasoningCapabilities:
    """
    Get reasoning capabilities for a model.

    Args:
        model: Model identifier.

    Returns:
        ReasoningCapabilities for the model.
    """
    # Check exact match first
    if model in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[model]

    # Check prefix matches
    for prefix, caps in MODEL_CAPABILITIES.items():
        if model.startswith(prefix):
            return caps

    # Default: no reasoning capabilities
    return ReasoningCapabilities()


def is_reasoning_model(model: str) -> bool:
    """
    Check if a model supports extended reasoning.

    Args:
        model: Model identifier.

    Returns:
        True if model supports reasoning.
    """
    return get_model_capabilities(model).supports_extended_thinking


def get_default_thinking_budget(model: str) -> int:
    """
    Get the default thinking budget for a model.

    Args:
        model: Model identifier.

    Returns:
        Default thinking token budget.
    """
    return get_model_capabilities(model).default_thinking_tokens


def calculate_thinking_budget(
    model: str,
    complexity: str = "moderate",
    base_budget: int | None = None,
) -> int:
    """
    Calculate thinking budget based on complexity.

    Args:
        model: Model identifier.
        complexity: Problem complexity level.
        base_budget: Optional base budget override.

    Returns:
        Calculated thinking budget.
    """
    caps = get_model_capabilities(model)
    base = base_budget or caps.default_thinking_tokens

    budget_config = ThinkingBudget()
    multiplier = budget_config.complexity_multipliers.get(complexity, 1.0)

    calculated = int(base * multiplier)
    return min(calculated, caps.max_thinking_tokens)


class ReasoningPromptBuilder:
    """
    Builds prompts optimized for reasoning models.

    Different reasoning models have different requirements for prompts.
    This class handles the variations.
    """

    @staticmethod
    def build_prompt(
        task: str,
        context: str | None = None,
        model: str = "o1",
        style: str = "thorough",
    ) -> str:
        """
        Build a prompt optimized for reasoning.

        Args:
            task: The main task or question.
            context: Optional context or background.
            model: Target reasoning model.
            style: Thinking style.

        Returns:
            Optimized prompt string.
        """
        caps = get_model_capabilities(model)

        parts = []

        # Add style guidance
        if style == "thorough":
            parts.append(
                "Take your time to think through this carefully, "
                "considering multiple perspectives and potential issues."
            )
        elif style == "analytical":
            parts.append(
                "Analyze this systematically, breaking down the problem "
                "into components and examining each thoroughly."
            )
        elif style == "concise":
            parts.append(
                "Think through this efficiently, focusing on the key "
                "aspects that lead to a clear conclusion."
            )

        # Add context if provided
        if context:
            parts.append(f"\nContext:\n{context}")

        # Add main task
        parts.append(f"\nTask:\n{task}")

        # Model-specific adjustments
        if not caps.supports_system_prompts:
            # For models like o1 that don't support system prompts,
            # embed all instructions in the user message
            parts.insert(0, "You are an expert analytical reasoner.")

        return "\n\n".join(parts)

    @staticmethod
    def build_debate_prompt(
        topic: str,
        position: str,
        opponent_arguments: list[str] | None = None,
        _model: str = "o1",
    ) -> str:
        """
        Build a prompt for debate reasoning.

        Args:
            topic: Debate topic.
            position: Position to argue.
            opponent_arguments: Previous opponent arguments to address.
            _model: Target reasoning model (reserved for future use).

        Returns:
            Debate-optimized prompt.
        """
        parts = [
            "You are participating in a structured debate.",
            f"\nTopic: {topic}",
            f"\nYour position: {position}",
        ]

        if opponent_arguments:
            parts.append("\nOpponent's arguments to address:")
            for i, arg in enumerate(opponent_arguments, 1):
                parts.append(f"  {i}. {arg[:200]}...")

        parts.append(
            "\nConstruct a well-reasoned argument supporting your position. "
            "Consider and address counterarguments. Use evidence and logic."
        )

        return "\n".join(parts)


def create_reasoning_config(
    model: str,
    **overrides: Any,
) -> ReasoningConfig:
    """
    Create a ReasoningConfig with model-appropriate defaults.

    Args:
        model: Model identifier.
        **overrides: Configuration overrides.

    Returns:
        ReasoningConfig with appropriate settings.
    """
    caps = get_model_capabilities(model)

    # Start with model-specific defaults
    config_dict: dict[str, Any] = {
        "model": model,
        "thinking_budget": caps.default_thinking_tokens,
        "show_thinking": caps.supports_thinking_visibility,
        "use_system_prompt": caps.supports_system_prompts,
    }

    if caps.requires_temperature_1:
        config_dict["temperature"] = 1.0

    # Apply overrides
    config_dict.update(overrides)

    return ReasoningConfig(**config_dict)
