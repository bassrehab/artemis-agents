"""ARTEMIS Google/Gemini Provider with Vertex AI support."""

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
REASONING_MODELS = {"gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash-thinking-exp"}

# Token limits by model
MODEL_TOKEN_LIMITS = {
    "gemini-2.0-flash": 1048576,
    "gemini-2.0-flash-exp": 1048576,
    "gemini-1.5-pro": 2097152,
    "gemini-1.5-flash": 1048576,
    "gemini-2.5-pro": 1048576,
    "gemini-2.5-flash": 1048576,
}


class GoogleModel(BaseModel):
    """
    Google/Gemini model provider.

    Supports both Google AI Studio and Vertex AI backends.
    Use Vertex AI for higher rate limits and enterprise features.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        project: str | None = None,
        location: str = "us-central1",
        use_vertex_ai: bool | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """
        Initialize Google model.

        For Vertex AI, set use_vertex_ai=True and provide project ID.
        For AI Studio, provide api_key or set GOOGLE_API_KEY env var.
        """
        # Determine which backend to use
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

        # Auto-detect: use Vertex AI if project is set, otherwise AI Studio
        if use_vertex_ai is None:
            use_vertex_ai = bool(self.project)

        self.use_vertex_ai = use_vertex_ai

        if use_vertex_ai:
            api_key = None  # Vertex AI uses ADC, not API key
        else:
            api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

        super().__init__(
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        if use_vertex_ai:
            self._init_vertex_ai()
        else:
            self._init_ai_studio()

    def _init_vertex_ai(self):
        """Initialize Vertex AI client."""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError as e:
            raise ImportError(
                "google-cloud-aiplatform is required for Vertex AI. "
                "Install with: pip install google-cloud-aiplatform"
            ) from e

        if not self.project:
            raise ValueError(
                "project is required for Vertex AI. "
                "Set GOOGLE_CLOUD_PROJECT env var or pass project parameter."
            )

        vertexai.init(project=self.project, location=self.location)
        self._vertex_model_class = GenerativeModel
        self._genai = None

    def _init_ai_studio(self):
        """Initialize Google AI Studio client."""
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai is required for Google AI Studio. "
                "Install with: pip install google-generativeai"
            ) from e

        if not self.api_key:
            raise ValueError(
                "api_key is required for Google AI Studio. "
                "Set GOOGLE_API_KEY env var or pass api_key parameter."
            )

        genai.configure(api_key=self.api_key)
        self._genai = genai
        self._vertex_model_class = None

    def _get_model(self, system_instruction: str | None = None):
        """Get model instance."""
        if self.use_vertex_ai:
            from vertexai.generative_models import GenerativeModel
            return GenerativeModel(
                model_name=self.model,
                system_instruction=system_instruction,
            )
        else:
            import google.generativeai as genai
            return genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_instruction,
            )

    @property
    def provider(self) -> str:
        return "vertex_ai" if self.use_vertex_ai else "google"

    @property
    def supports_reasoning(self) -> bool:
        return any(self.model.startswith(rm) for rm in REASONING_MODELS)

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
        """Generate a response using Gemini API."""
        # Extract system message if present
        system_instruction = None
        chat_messages = []
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                chat_messages.append(msg)

        try:
            model = self._get_model(system_instruction)

            # Build generation config
            generation_config = self._build_generation_config(
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )

            # Convert messages to Gemini format
            contents = self._convert_messages(chat_messages)

            # Generate response
            response = await model.generate_content_async(
                contents=contents,
                generation_config=generation_config,
                **kwargs,
            )

            # Extract content
            content = response.text if response.text else ""

            # Build usage info
            usage_metadata = getattr(response, "usage_metadata", None)
            usage = self._create_usage(
                prompt_tokens=getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0,
                completion_tokens=getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0,
            )

            return ModelResponse(
                content=content,
                usage=usage,
                model=self.model,
                finish_reason=self._get_finish_reason(response),
            )

        except Exception as e:
            self._handle_error(e)

    async def generate_with_reasoning(
        self,
        messages: list[Message],
        thinking_budget: int = 8000,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ReasoningResponse:
        """Generate with extended thinking for Gemini 2.5 models."""
        if not self.supports_reasoning:
            raise NotImplementedError(
                f"Model {self.model} does not support extended reasoning. "
                f"Use 'gemini-2.5-pro' or 'gemini-2.5-flash'."
            )

        # Extract system message
        system_instruction = None
        chat_messages = []
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                chat_messages.append(msg)

        try:
            model = self._get_model(system_instruction)

            # Build config with thinking budget
            generation_config = self._build_generation_config(
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
            )

            contents = self._convert_messages(chat_messages)

            response = await model.generate_content_async(
                contents=contents,
                generation_config=generation_config,
                **kwargs,
            )

            # Extract thinking and content from response
            thinking = None
            content = ""
            thinking_tokens = 0

            # Gemini 2.5 returns thinking in parts
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, "thought") and part.thought:
                            thinking = part.text if hasattr(part, "text") else str(part)
                        elif hasattr(part, "text"):
                            content += part.text

            # Fallback to simple text extraction
            if not content and response.text:
                content = response.text

            usage_metadata = getattr(response, "usage_metadata", None)
            if usage_metadata:
                thinking_tokens = getattr(usage_metadata, "thoughts_token_count", 0)

            usage = self._create_usage(
                prompt_tokens=getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0,
                completion_tokens=getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0,
                reasoning_tokens=thinking_tokens,
            )

            return ReasoningResponse(
                content=content,
                usage=usage,
                model=self.model,
                finish_reason=self._get_finish_reason(response),
                thinking=thinking,
                thinking_tokens=thinking_tokens,
            )

        except Exception as e:
            self._handle_error(e)

    async def generate_stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        # Extract system message
        system_instruction = None
        chat_messages = []
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                chat_messages.append(msg)

        try:
            model = self._get_model(system_instruction)

            generation_config = self._build_generation_config(
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )

            contents = self._convert_messages(chat_messages)

            response = await model.generate_content_async(
                contents=contents,
                generation_config=generation_config,
                stream=True,
                **kwargs,
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            self._handle_error(e)

    async def count_tokens(self, messages: list[Message]) -> int:
        """Count tokens using Gemini's token counting API."""
        try:
            model = self._get_model()
            contents = self._convert_messages(messages)
            result = await model.count_tokens_async(contents)
            return result.total_tokens
        except Exception:
            # Fallback: estimate ~4 chars per token
            total_chars = sum(len(m.content) for m in messages)
            return total_chars // 4

    def _build_generation_config(
        self,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        thinking_budget: int | None = None,
    ):
        """Build generation config for either backend."""
        if self.use_vertex_ai:
            from vertexai.generative_models import GenerationConfig
            config_cls = GenerationConfig
        else:
            import google.generativeai as genai
            config_cls = genai.GenerationConfig

        config_kwargs = {
            "temperature": temperature,
        }

        if max_tokens:
            config_kwargs["max_output_tokens"] = max_tokens
        if stop:
            config_kwargs["stop_sequences"] = stop
        if thinking_budget:
            config_kwargs["thinking_config"] = {"thinking_budget": thinking_budget}

        return config_cls(**config_kwargs)

    def _convert_messages(self, messages: list[Message]) -> list:
        """Convert ARTEMIS messages to Gemini format."""
        if self.use_vertex_ai:
            from vertexai.generative_models import Content, Part

            contents = []
            for msg in messages:
                role = "model" if msg.role == "assistant" else "user"
                contents.append(Content(role=role, parts=[Part.from_text(msg.content)]))
            return contents
        else:
            # AI Studio format
            contents = []
            for msg in messages:
                role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": role, "parts": [msg.content]})
            return contents

    def _get_finish_reason(self, response) -> str | None:
        """Extract finish reason from response."""
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "finish_reason"):
                return str(candidate.finish_reason)
        return None

    def _handle_error(self, e: Exception):
        """Handle and re-raise errors with appropriate types."""
        from google.api_core import exceptions as google_exceptions

        if isinstance(e, google_exceptions.ResourceExhausted):
            raise RateLimitError(
                message=str(e),
                provider=self.provider,
            ) from e
        elif isinstance(e, google_exceptions.InvalidArgument):
            if "token" in str(e).lower():
                raise TokenLimitError(
                    message=str(e),
                    provider=self.provider,
                ) from e
            raise ModelError(message=str(e), provider=self.provider) from e
        elif isinstance(e, google_exceptions.GoogleAPIError):
            raise ProviderConnectionError(
                message=f"Failed to connect to {self.provider}: {e}",
                provider=self.provider,
            ) from e
        else:
            raise ModelError(message=str(e), provider=self.provider) from e


# Register the provider
ModelRegistry.register(
    provider="google",
    model_class=GoogleModel,
    model_prefixes=["gemini-"],
)
