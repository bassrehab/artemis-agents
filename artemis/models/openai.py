"""
ARTEMIS OpenAI Provider

OpenAI model implementation supporting GPT-4o and o1 reasoning models.
"""

import os
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any, TypeVar

import openai
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from artemis.core.types import Message, ModelResponse, ReasoningResponse
from artemis.exceptions import (
    ModelError,
    ProviderConnectionError,
    RateLimitError,
    TokenLimitError,
)
from artemis.models.base import BaseModel, ModelRegistry

T = TypeVar("T")

# Models that support extended thinking/reasoning
REASONING_MODELS = {"o1", "o1-preview", "o1-mini", "o1-2024-12-17"}

# Default token limits by model
MODEL_TOKEN_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "o1": 200000,
    "o1-preview": 128000,
    "o1-mini": 128000,
}


class OpenAIModel(BaseModel):
    """
    OpenAI model provider.

    Supports:
    - GPT-4o and GPT-4o-mini for standard generation
    - o1 and o1-mini for reasoning with extended thinking
    - Streaming responses
    - Token counting via tiktoken
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        Initialize OpenAI model.

        Args:
            model: Model identifier (e.g., "gpt-4o", "o1").
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            base_url: Custom API base URL (for Azure or proxies).
            organization: OpenAI organization ID.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
            **kwargs: Additional configuration.
        """
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
        self.organization = organization

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
            max_retries=0,  # We handle retries ourselves
        )

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def supports_reasoning(self) -> bool:
        return any(self.model.startswith(rm) for rm in REASONING_MODELS)

    @property
    def supports_streaming(self) -> bool:
        # o1 models don't support streaming yet
        return not self.supports_reasoning

    def _with_retry(
        self, func: Callable[[], Coroutine[Any, Any, T]]
    ) -> Callable[[], Coroutine[Any, Any, T]]:
        """Wrap an async function with retry logic."""
        decorated = retry(
            retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
            wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
            stop=stop_after_attempt(self.max_retries),
            reraise=True,
        )(func)
        return decorated

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response using OpenAI API."""

        async def _call() -> ModelResponse:
            try:
                # Build request parameters
                params: dict[str, Any] = {
                    "model": self.model,
                    "messages": self._messages_to_dicts(messages),
                }

                # o1 models have different parameter requirements
                if self.supports_reasoning:
                    # o1 models don't support temperature or system messages
                    if max_tokens:
                        params["max_completion_tokens"] = max_tokens
                else:
                    params["temperature"] = temperature
                    if max_tokens:
                        params["max_tokens"] = max_tokens
                    if stop:
                        params["stop"] = stop

                # Add any additional kwargs
                params.update(kwargs)

                response = await self._client.chat.completions.create(**params)

                # Extract response
                choice = response.choices[0]
                content = choice.message.content or ""

                # Build usage
                usage = self._create_usage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                )

                return ModelResponse(
                    content=content,
                    usage=usage,
                    model=response.model,
                    finish_reason=choice.finish_reason,
                )

            except openai.RateLimitError as e:
                raise RateLimitError(
                    message=str(e),
                    provider=self.provider,
                    retry_after=getattr(e, "retry_after", None),
                ) from e
            except openai.BadRequestError as e:
                if "maximum context length" in str(e).lower():
                    raise TokenLimitError(
                        message=str(e),
                        provider=self.provider,
                    ) from e
                raise ModelError(message=str(e), provider=self.provider) from e
            except openai.APIConnectionError as e:
                raise ProviderConnectionError(
                    message=f"Failed to connect to OpenAI: {e}",
                    provider=self.provider,
                ) from e
            except openai.APIError as e:
                raise ModelError(message=str(e), provider=self.provider) from e

        return await self._with_retry(_call)()

    async def generate_with_reasoning(
        self,
        messages: list[Message],
        thinking_budget: int = 8000,  # noqa: ARG002 - reserved for future use
        temperature: float = 1.0,  # noqa: ARG002 - o1 doesn't support temperature
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ReasoningResponse:
        """
        Generate a response with extended thinking using o1 models.

        Note: o1 models handle reasoning internally. The thinking_budget
        parameter influences the reasoning effort but the actual thinking
        tokens are determined by the model.
        """
        if not self.supports_reasoning:
            raise NotImplementedError(
                f"Model {self.model} does not support extended reasoning. "
                f"Use 'o1', 'o1-preview', or 'o1-mini' for reasoning."
            )

        async def _call() -> ReasoningResponse:
            try:
                params: dict[str, Any] = {
                    "model": self.model,
                    "messages": self._messages_to_dicts(messages),
                }

                # Set max completion tokens
                if max_tokens:
                    params["max_completion_tokens"] = max_tokens

                # o1 supports reasoning_effort parameter
                reasoning_effort = kwargs.pop("reasoning_effort", None)
                if reasoning_effort:
                    params["reasoning_effort"] = reasoning_effort

                params.update(kwargs)

                response = await self._client.chat.completions.create(**params)

                choice = response.choices[0]
                content = choice.message.content or ""

                # Extract reasoning tokens if available
                reasoning_tokens = None
                if response.usage:
                    # OpenAI provides completion_tokens_details with reasoning_tokens
                    details = getattr(response.usage, "completion_tokens_details", None)
                    if details:
                        reasoning_tokens = getattr(details, "reasoning_tokens", None)

                usage = self._create_usage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    reasoning_tokens=reasoning_tokens,
                )

                return ReasoningResponse(
                    content=content,
                    usage=usage,
                    model=response.model,
                    finish_reason=choice.finish_reason,
                    thinking=None,  # o1 doesn't expose the thinking trace
                    thinking_tokens=reasoning_tokens or 0,
                )

            except openai.RateLimitError as e:
                raise RateLimitError(
                    message=str(e),
                    provider=self.provider,
                    retry_after=getattr(e, "retry_after", None),
                ) from e
            except openai.BadRequestError as e:
                if "maximum context length" in str(e).lower():
                    raise TokenLimitError(
                        message=str(e),
                        provider=self.provider,
                    ) from e
                raise ModelError(message=str(e), provider=self.provider) from e
            except openai.APIConnectionError as e:
                raise ProviderConnectionError(
                    message=f"Failed to connect to OpenAI: {e}",
                    provider=self.provider,
                ) from e
            except openai.APIError as e:
                raise ModelError(message=str(e), provider=self.provider) from e

        return await self._with_retry(_call)()

    async def generate_stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        if not self.supports_streaming:
            raise NotImplementedError(f"Model {self.model} does not support streaming.")

        try:
            params: dict[str, Any] = {
                "model": self.model,
                "messages": self._messages_to_dicts(messages),
                "temperature": temperature,
                "stream": True,
            }

            if max_tokens:
                params["max_tokens"] = max_tokens
            if stop:
                params["stop"] = stop

            params.update(kwargs)

            stream = await self._client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except openai.RateLimitError as e:
            raise RateLimitError(
                message=str(e),
                provider=self.provider,
                retry_after=getattr(e, "retry_after", None),
            ) from e
        except openai.APIConnectionError as e:
            raise ProviderConnectionError(
                message=f"Failed to connect to OpenAI: {e}",
                provider=self.provider,
            ) from e
        except openai.APIError as e:
            raise ModelError(message=str(e), provider=self.provider) from e

    async def count_tokens(self, messages: list[Message]) -> int:
        """
        Count tokens in messages using tiktoken.

        Note: This is an approximation. For exact counts, use the API.
        """
        try:
            import tiktoken
        except ImportError:
            # Rough estimate: ~4 chars per token
            total_chars = sum(len(m.content) for m in messages)
            return total_chars // 4

        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        token_count = 0
        for message in messages:
            # Each message has overhead tokens
            token_count += 4  # <|start|>role<|sep|>content<|end|>
            token_count += len(encoding.encode(message.content))
            if message.name:
                token_count += len(encoding.encode(message.name))
                token_count += 1  # name field separator

        token_count += 2  # <|start|>assistant<|sep|>
        return token_count


# Register the provider
ModelRegistry.register(
    provider="openai",
    model_class=OpenAIModel,
    model_prefixes=["gpt-", "o1", "chatgpt-"],
)
