"""
ARTEMIS Base Model Interface

Abstract base class for LLM provider integrations.
Provides a unified interface for text generation with optional reasoning support.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from artemis.core.types import Message, ModelResponse, ReasoningResponse, Usage


class BaseModel(ABC):
    """
    Abstract base class for LLM providers.

    All model implementations must inherit from this class and implement
    the required methods for text generation.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the model.

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3-opus").
            api_key: API key for authentication. If None, reads from environment.
            base_url: Optional custom base URL for API requests.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts for failed requests.
            **kwargs: Additional provider-specific configuration.
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._config = kwargs

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name (e.g., 'openai', 'anthropic')."""
        ...

    @property
    def supports_reasoning(self) -> bool:
        """Whether this model supports extended thinking/reasoning."""
        return False

    @property
    def supports_streaming(self) -> bool:
        """Whether this model supports streaming responses."""
        return True

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Generate a response from the model.

        Args:
            messages: List of conversation messages.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.
            stop: Optional stop sequences.
            **kwargs: Additional provider-specific parameters.

        Returns:
            ModelResponse with generated content and usage statistics.

        Raises:
            ModelError: If generation fails.
            RateLimitError: If rate limit is exceeded.
            TokenLimitError: If token limit is exceeded.
        """
        ...

    async def generate_with_reasoning(
        self,
        messages: list[Message],  # noqa: ARG002
        thinking_budget: int = 8000,  # noqa: ARG002
        temperature: float = 1.0,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ReasoningResponse:
        """
        Generate a response with extended thinking/reasoning.

        For models that support it (o1, R1, Gemini 2.5 Pro), this enables
        the extended thinking mode where the model reasons before responding.

        Args:
            messages: List of conversation messages.
            thinking_budget: Maximum tokens for thinking/reasoning.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens for the final response.
            **kwargs: Additional provider-specific parameters.

        Returns:
            ReasoningResponse with thinking trace and final content.

        Raises:
            NotImplementedError: If model doesn't support reasoning.
            ModelError: If generation fails.
        """
        if not self.supports_reasoning:
            raise NotImplementedError(
                f"Model {self.model} does not support extended reasoning. Use generate() instead."
            )
        # Subclasses that support reasoning should override this
        raise NotImplementedError

    async def generate_stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from the model.

        Args:
            messages: List of conversation messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stop: Optional stop sequences.
            **kwargs: Additional provider-specific parameters.

        Yields:
            String chunks of the generated response.

        Raises:
            NotImplementedError: If streaming is not supported.
            ModelError: If generation fails.
        """
        if not self.supports_streaming:
            raise NotImplementedError(f"Model {self.model} does not support streaming.")
        # Default implementation: call generate and yield full response
        response = await self.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs,
        )
        yield response.content

    @abstractmethod
    async def count_tokens(self, messages: list[Message]) -> int:
        """
        Count the number of tokens in the messages.

        Args:
            messages: List of messages to count tokens for.

        Returns:
            Token count.
        """
        ...

    def _messages_to_dicts(self, messages: list[Message]) -> list[dict[str, str]]:
        """Convert Message objects to dict format for API calls."""
        result = []
        for msg in messages:
            d = {"role": msg.role, "content": msg.content}
            if msg.name:
                d["name"] = msg.name
            result.append(d)
        return result

    def _create_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        reasoning_tokens: int | None = None,
    ) -> Usage:
        """Create a Usage object from token counts."""
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r}, provider={self.provider!r})"


class ModelRegistry:
    """
    Registry for model providers.

    Allows dynamic registration and lookup of model implementations.
    """

    _providers: dict[str, type[BaseModel]] = {}
    _model_mappings: dict[str, str] = {}

    @classmethod
    def register(
        cls,
        provider: str,
        model_class: type[BaseModel],
        model_prefixes: list[str] | None = None,
    ) -> None:
        """
        Register a model provider.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic').
            model_class: The model class to register.
            model_prefixes: Optional list of model name prefixes that map to this provider.
        """
        cls._providers[provider] = model_class
        if model_prefixes:
            for prefix in model_prefixes:
                cls._model_mappings[prefix] = provider

    @classmethod
    def get_provider(cls, provider: str) -> type[BaseModel]:
        """
        Get a provider class by name.

        Args:
            provider: Provider name.

        Returns:
            The provider class.

        Raises:
            KeyError: If provider not found.
        """
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise KeyError(f"Provider '{provider}' not found. Available: {available}")
        return cls._providers[provider]

    @classmethod
    def infer_provider(cls, model: str) -> str | None:
        """
        Infer the provider from a model name.

        Args:
            model: Model identifier.

        Returns:
            Provider name or None if cannot be inferred.
        """
        for prefix, provider in cls._model_mappings.items():
            if model.startswith(prefix):
                return provider
        return None

    @classmethod
    def create(
        cls,
        model: str,
        provider: str | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """
        Create a model instance.

        Args:
            model: Model identifier.
            provider: Optional provider name. If not specified, will be inferred.
            **kwargs: Additional arguments passed to the model constructor.

        Returns:
            A model instance.

        Raises:
            ValueError: If provider cannot be determined.
            KeyError: If provider not found.
        """
        if provider is None:
            provider = cls.infer_provider(model)
            if provider is None:
                raise ValueError(
                    f"Cannot infer provider for model '{model}'. "
                    f"Please specify provider explicitly."
                )

        model_class = cls.get_provider(provider)
        return model_class(model=model, **kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered providers."""
        return list(cls._providers.keys())
