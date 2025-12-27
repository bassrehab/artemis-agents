"""Config for reasoning models (o1, R1, Gemini 2.5 Pro)."""

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
    ALWAYS = "always"
    ADAPTIVE = "adaptive"  # use only for complex problems
    NEVER = "never"


@dataclass
class ThinkingBudget:
    min_tokens: int = 1000
    max_tokens: int = 32000
    default_tokens: int = 8000
    scale_with_complexity: bool = True
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
    """Reasoning capabilities for a model."""

    supports_extended_thinking: bool = False
    supports_thinking_budget: bool = False
    supports_thinking_visibility: bool = False
    supports_system_prompts: bool = True
    requires_temperature_1: bool = False
    max_thinking_tokens: int = 128000
    default_thinking_tokens: int = 8000


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


def get_model_capabilities(model: str):
    """Get reasoning capabilities for a model."""
    # Check exact match first
    if model in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[model]

    # Check prefix matches
    for prefix, caps in MODEL_CAPABILITIES.items():
        if model.startswith(prefix):
            return caps

    # Default: no reasoning capabilities
    return ReasoningCapabilities()


def is_reasoning_model(model: str):
    return get_model_capabilities(model).supports_extended_thinking


def get_default_thinking_budget(model: str):
    return get_model_capabilities(model).default_thinking_tokens


def calculate_thinking_budget(model: str, complexity: str = "moderate", base_budget=None):
    """Calculate thinking budget based on complexity."""
    caps = get_model_capabilities(model)
    base = base_budget or caps.default_thinking_tokens

    budget_config = ThinkingBudget()
    multiplier = budget_config.complexity_multipliers.get(complexity, 1.0)

    calculated = int(base * multiplier)
    return min(calculated, caps.max_thinking_tokens)


class ReasoningPromptBuilder:
    """Builds prompts optimized for reasoning models."""

    @staticmethod
    def build_prompt(task, context=None, model="o1", style="thorough"):
        """Build a prompt optimized for reasoning."""
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
    def build_debate_prompt(topic, position, opponent_arguments=None, _model="o1"):
        """Build a prompt for debate reasoning."""
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


def create_reasoning_config(model: str, **overrides):
    """Create ReasoningConfig with model-appropriate defaults."""
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
