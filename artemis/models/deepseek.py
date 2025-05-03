"""
ARTEMIS DeepSeek R1 Provider

Integration with DeepSeek R1 reasoning model.
Supports extended thinking with visible reasoning traces.
"""

import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from artemis.core.types import Message, ModelResponse, ReasoningResponse
from artemis.models.base import BaseModel, ModelRegistry
from artemis.models.reasoning import get_model_capabilities
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


class DeepSeekError(Exception):
    """Base exception for DeepSeek API errors."""

    pass


class DeepSeekRateLimitError(DeepSeekError):
    """Rate limit exceeded."""

    pass


class DeepSeekAPIError(DeepSeekError):
    """API error from DeepSeek."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class DeepSeekModel(BaseModel):
    """
    DeepSeek model provider with R1 reasoning support.

    Supports DeepSeek R1 and distilled variants with extended thinking.

    Example:
        >>> model = DeepSeekModel(model="deepseek-reasoner")
        >>> response = await model.generate_with_reasoning(
        ...     messages=[Message(role="user", content="Solve this puzzle...")],
        ...     thinking_budget=8000,
        ... )
        >>> print(response.thinking)  # Shows reasoning trace
        >>> print(response.content)   # Shows final answer

    Supported models:
        - deepseek-reasoner: Full R1 with extended thinking
        - deepseek-r1-distill-llama-70b: Distilled R1 variant
        - deepseek-chat: Standard chat model
        - deepseek-coder: Code-specialized model
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

    # Model identifiers
    REASONING_MODELS = {"deepseek-reasoner", "deepseek-r1-distill-llama-70b"}
    ALL_MODELS = REASONING_MODELS | {"deepseek-chat", "deepseek-coder"}

    def __init__(
        self,
        model: str = "deepseek-reasoner",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the DeepSeek model.

        Args:
            model: Model identifier.
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var).
            base_url: Custom API base URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
            **kwargs: Additional configuration.
        """
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY environment "
                "variable or pass api_key parameter."
            )

        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        # Get capabilities
        self._capabilities = get_model_capabilities(model)

        logger.info(
            "DeepSeekModel initialized",
            model=model,
            supports_reasoning=self.supports_reasoning,
        )

    @property
    def provider(self) -> str:
        return "deepseek"

    @property
    def supports_reasoning(self) -> bool:
        return self.model in self.REASONING_MODELS

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
        """
        Generate a response from DeepSeek.

        Args:
            messages: Conversation messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stop: Stop sequences.
            **kwargs: Additional parameters.

        Returns:
            ModelResponse with content and usage.
        """
        request_body = self._build_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs,
        )

        response_data = await self._make_request(request_body)

        return self._parse_response(response_data)

    async def generate_with_reasoning(
        self,
        messages: list[Message],
        thinking_budget: int = 8000,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ReasoningResponse:
        """
        Generate a response with extended thinking.

        DeepSeek R1 provides visible reasoning traces that show
        the model's chain-of-thought process.

        Args:
            messages: Conversation messages.
            thinking_budget: Token budget for thinking.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens for final response.
            **kwargs: Additional parameters.

        Returns:
            ReasoningResponse with thinking trace and content.
        """
        if not self.supports_reasoning:
            raise NotImplementedError(
                f"Model {self.model} does not support extended reasoning. "
                f"Use deepseek-reasoner or deepseek-r1-distill-llama-70b."
            )

        # Add thinking configuration
        request_body = self._build_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            **kwargs,
        )

        response_data = await self._make_request(request_body)

        return self._parse_reasoning_response(response_data)

    async def generate_stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response.

        Args:
            messages: Conversation messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            stop: Stop sequences.
            **kwargs: Additional parameters.

        Yields:
            Response chunks.
        """
        request_body = self._build_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            stream=True,
            **kwargs,
        )

        async with self._client.stream(
            "POST",
            "/chat/completions",
            json=request_body,
        ) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise DeepSeekAPIError(
                    f"API error: {text.decode()}",
                    status_code=response.status_code,
                )

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    import json
                    chunk = json.loads(data)
                    if chunk.get("choices"):
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content

    async def count_tokens(self, messages: list[Message]) -> int:
        """
        Estimate token count for messages.

        DeepSeek doesn't provide a token counting endpoint,
        so we estimate based on character count.

        Args:
            messages: Messages to count.

        Returns:
            Estimated token count.
        """
        total_chars = sum(
            len(msg.content) + len(msg.role) + 4  # Role tokens + format
            for msg in messages
        )
        # Rough estimate: ~4 chars per token for English
        return total_chars // 4

    def _build_request(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        stream: bool = False,
        thinking_budget: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the API request body."""
        request: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages_to_dicts(messages),
            "temperature": temperature,
            "stream": stream,
        }

        if max_tokens:
            request["max_tokens"] = max_tokens

        if stop:
            request["stop"] = stop

        # Add thinking configuration for R1 models
        if thinking_budget and self.supports_reasoning:
            request["reasoning_effort"] = self._thinking_budget_to_effort(
                thinking_budget
            )

        # Add any extra parameters
        request.update(kwargs)

        return request

    def _thinking_budget_to_effort(self, budget: int) -> str:
        """Convert thinking budget to effort level."""
        if budget < 2000:
            return "low"
        elif budget < 8000:
            return "medium"
        else:
            return "high"

    async def _make_request(self, request_body: dict[str, Any]) -> dict[str, Any]:
        """Make an API request with retries."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await self._client.post(
                    "/chat/completions",
                    json=request_body,
                )

                if response.status_code == 429:
                    raise DeepSeekRateLimitError("Rate limit exceeded")

                if response.status_code != 200:
                    raise DeepSeekAPIError(
                        f"API error: {response.text}",
                        status_code=response.status_code,
                    )

                return response.json()

            except DeepSeekRateLimitError:
                # Exponential backoff for rate limits
                import asyncio
                wait_time = 2 ** attempt
                logger.warning(
                    "Rate limit hit, waiting",
                    wait_time=wait_time,
                    attempt=attempt + 1,
                )
                await asyncio.sleep(wait_time)
                last_error = DeepSeekRateLimitError("Rate limit exceeded")

            except httpx.RequestError as e:
                logger.warning(
                    "Request error",
                    error=str(e),
                    attempt=attempt + 1,
                )
                last_error = e

        raise DeepSeekError(f"Request failed after {self.max_retries} attempts") from last_error

    def _parse_response(self, data: dict[str, Any]) -> ModelResponse:
        """Parse API response into ModelResponse."""
        choice = data["choices"][0]
        message = choice["message"]

        usage_data = data.get("usage", {})
        usage = self._create_usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
        )

        return ModelResponse(
            content=message.get("content", ""),
            usage=usage,
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason"),
        )

    def _parse_reasoning_response(self, data: dict[str, Any]) -> ReasoningResponse:
        """Parse API response into ReasoningResponse."""
        choice = data["choices"][0]
        message = choice["message"]

        # DeepSeek R1 includes thinking in the response
        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", "")

        usage_data = data.get("usage", {})
        reasoning_tokens = usage_data.get("reasoning_tokens", 0)

        usage = self._create_usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            reasoning_tokens=reasoning_tokens,
        )

        return ReasoningResponse(
            content=content,
            thinking=reasoning_content,
            usage=usage,
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason"),
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "DeepSeekModel":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


# Register the provider
ModelRegistry.register(
    provider="deepseek",
    model_class=DeepSeekModel,
    model_prefixes=["deepseek"],
)
