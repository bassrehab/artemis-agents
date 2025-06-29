"""ARTEMIS Anthropic/Claude Provider."""

import os
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from artemis.core.types import Message, ModelResponse, ReasoningResponse
from artemis.exceptions import (
    ModelError,
    ProviderConnectionError,
    RateLimitError,
    TokenLimitError,
)
from artemis.models.base import BaseModel, ModelRegistry

T = TypeVar("T")

# Models that support extended thinking
REASONING_MODELS = {"claude-3-5-sonnet-20241022", "claude-sonnet-4-20250514"}

# Token limits by model
MODEL_TOKEN_LIMITS = {
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-opus-4-20250514": 200000,
}


class AnthropicModel(BaseModel):
    """
    Anthropic/Claude model provider.

    Supports Claude 3 and 3.5 models including extended thinking.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        # Import here to allow graceful failure if not installed
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError(
                "anthropic is required for Anthropic provider. "
                "Install with: pip install anthropic"
            ) from e

        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,  # We handle retries ourselves
        )

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def supports_reasoning(self) -> bool:
        return any(rm in self.model for rm in REASONING_MODELS)

    @property
    def supports_streaming(self) -> bool:
        return True

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response using Anthropic API."""
        import anthropic

        # Extract system message
        system_prompt = None
        chat_messages = []
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})

        try:
            params: dict[str, Any] = {
                "model": self.model,
                "messages": chat_messages,
                "max_tokens": max_tokens or 4096,
                "temperature": temperature,
            }

            if system_prompt:
                params["system"] = system_prompt

            if stop:
                params["stop_sequences"] = stop

            params.update(kwargs)

            response = await self._client.messages.create(**params)

            # Extract content
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            usage = self._create_usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )

            return ModelResponse(
                content=content,
                usage=usage,
                model=response.model,
                finish_reason=response.stop_reason,
            )

        except anthropic.RateLimitError as e:
            raise RateLimitError(
                message=str(e),
                provider=self.provider,
            ) from e
        except anthropic.BadRequestError as e:
            if "token" in str(e).lower() or "length" in str(e).lower():
                raise TokenLimitError(
                    message=str(e),
                    provider=self.provider,
                ) from e
            raise ModelError(message=str(e), provider=self.provider) from e
        except anthropic.APIConnectionError as e:
            raise ProviderConnectionError(
                message=f"Failed to connect to Anthropic: {e}",
                provider=self.provider,
            ) from e
        except anthropic.APIError as e:
            raise ModelError(message=str(e), provider=self.provider) from e

    async def generate_with_reasoning(
        self,
        messages: list[Message],
        thinking_budget: int = 8000,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ReasoningResponse:
        """Generate with extended thinking for Claude 3.5 Sonnet."""
        import anthropic

        if not self.supports_reasoning:
            raise NotImplementedError(
                f"Model {self.model} does not support extended reasoning. "
                f"Use 'claude-3-5-sonnet-20241022' or newer."
            )

        # Extract system message
        system_prompt = None
        chat_messages = []
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})

        try:
            params: dict[str, Any] = {
                "model": self.model,
                "messages": chat_messages,
                "max_tokens": max_tokens or 16384,
                "temperature": temperature,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                },
            }

            if system_prompt:
                params["system"] = system_prompt

            params.update(kwargs)

            response = await self._client.messages.create(**params)

            # Extract thinking and content from response blocks
            thinking = None
            content = ""
            thinking_tokens = 0

            for block in response.content:
                if hasattr(block, "type"):
                    if block.type == "thinking":
                        thinking = block.thinking if hasattr(block, "thinking") else str(block)
                    elif block.type == "text":
                        content += block.text if hasattr(block, "text") else ""

            # Get thinking tokens from usage if available
            if hasattr(response.usage, "cache_read_input_tokens"):
                # Extended thinking uses cache tokens
                thinking_tokens = getattr(response.usage, "cache_read_input_tokens", 0)

            usage = self._create_usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                reasoning_tokens=thinking_tokens,
            )

            return ReasoningResponse(
                content=content,
                usage=usage,
                model=response.model,
                finish_reason=response.stop_reason,
                thinking=thinking,
                thinking_tokens=thinking_tokens,
            )

        except anthropic.RateLimitError as e:
            raise RateLimitError(
                message=str(e),
                provider=self.provider,
            ) from e
        except anthropic.BadRequestError as e:
            if "token" in str(e).lower() or "length" in str(e).lower():
                raise TokenLimitError(
                    message=str(e),
                    provider=self.provider,
                ) from e
            raise ModelError(message=str(e), provider=self.provider) from e
        except anthropic.APIConnectionError as e:
            raise ProviderConnectionError(
                message=f"Failed to connect to Anthropic: {e}",
                provider=self.provider,
            ) from e
        except anthropic.APIError as e:
            raise ModelError(message=str(e), provider=self.provider) from e

    async def generate_stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        import anthropic

        # Extract system message
        system_prompt = None
        chat_messages = []
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})

        try:
            params: dict[str, Any] = {
                "model": self.model,
                "messages": chat_messages,
                "max_tokens": max_tokens or 4096,
                "temperature": temperature,
            }

            if system_prompt:
                params["system"] = system_prompt

            if stop:
                params["stop_sequences"] = stop

            params.update(kwargs)

            async with self._client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text

        except anthropic.RateLimitError as e:
            raise RateLimitError(
                message=str(e),
                provider=self.provider,
            ) from e
        except anthropic.APIConnectionError as e:
            raise ProviderConnectionError(
                message=f"Failed to connect to Anthropic: {e}",
                provider=self.provider,
            ) from e
        except anthropic.APIError as e:
            raise ModelError(message=str(e), provider=self.provider) from e

    async def count_tokens(self, messages: list[Message]) -> int:
        """Count tokens using Anthropic's token counting."""
        try:
            # Anthropic provides count_tokens endpoint
            chat_messages = []
            for msg in messages:
                if msg.role != "system":
                    chat_messages.append({"role": msg.role, "content": msg.content})

            result = await self._client.messages.count_tokens(
                model=self.model,
                messages=chat_messages,
            )
            return result.input_tokens
        except Exception:
            # Fallback: estimate ~4 chars per token
            total_chars = sum(len(m.content) for m in messages)
            return total_chars // 4


# Register the provider
ModelRegistry.register(
    provider="anthropic",
    model_class=AnthropicModel,
    model_prefixes=["claude-"],
)
