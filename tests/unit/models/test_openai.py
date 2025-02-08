"""
Unit tests for OpenAI model provider.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from artemis.core.types import Message, ModelResponse, ReasoningResponse
from artemis.exceptions import RateLimitError, TokenLimitError
from artemis.models import OpenAIModel, create_model


class TestOpenAIModel:
    """Tests for OpenAIModel class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        model = OpenAIModel(api_key="test-key")
        assert model.model == "gpt-4o"
        assert model.provider == "openai"
        assert model.timeout == 60.0
        assert model.max_retries == 3

    def test_init_custom_model(self) -> None:
        """Test initialization with custom model."""
        model = OpenAIModel(model="gpt-4-turbo", api_key="test-key")
        assert model.model == "gpt-4-turbo"

    def test_supports_reasoning_gpt4o(self) -> None:
        """Test that GPT-4o doesn't support reasoning."""
        model = OpenAIModel(model="gpt-4o", api_key="test-key")
        assert model.supports_reasoning is False

    def test_supports_reasoning_o1(self) -> None:
        """Test that o1 supports reasoning."""
        model = OpenAIModel(model="o1", api_key="test-key")
        assert model.supports_reasoning is True

    def test_supports_reasoning_o1_mini(self) -> None:
        """Test that o1-mini supports reasoning."""
        model = OpenAIModel(model="o1-mini", api_key="test-key")
        assert model.supports_reasoning is True

    def test_supports_streaming_gpt4o(self) -> None:
        """Test that GPT-4o supports streaming."""
        model = OpenAIModel(model="gpt-4o", api_key="test-key")
        assert model.supports_streaming is True

    def test_supports_streaming_o1(self) -> None:
        """Test that o1 doesn't support streaming."""
        model = OpenAIModel(model="o1", api_key="test-key")
        assert model.supports_streaming is False

    def test_repr(self) -> None:
        """Test string representation."""
        model = OpenAIModel(model="gpt-4o", api_key="test-key")
        assert "OpenAIModel" in repr(model)
        assert "gpt-4o" in repr(model)


class TestOpenAIModelGenerate:
    """Tests for OpenAIModel.generate method."""

    @pytest.fixture
    def model(self) -> OpenAIModel:
        """Create a model instance for testing."""
        return OpenAIModel(model="gpt-4o", api_key="test-key")

    @pytest.fixture
    def mock_response(self) -> MagicMock:
        """Create a mock OpenAI response."""
        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(content="Test response"),
                finish_reason="stop",
            )
        ]
        response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
        )
        response.model = "gpt-4o"
        return response

    @pytest.mark.asyncio
    async def test_generate_success(self, model: OpenAIModel, mock_response: MagicMock) -> None:
        """Test successful generation."""
        model._client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [Message(role="user", content="Hello")]
        response = await model.generate(messages)

        assert isinstance(response, ModelResponse)
        assert response.content == "Test response"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.model == "gpt-4o"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_with_parameters(
        self, model: OpenAIModel, mock_response: MagicMock
    ) -> None:
        """Test generation with custom parameters."""
        model._client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [Message(role="user", content="Hello")]
        await model.generate(
            messages,
            temperature=0.5,
            max_tokens=100,
            stop=["\n"],
        )

        # Verify the call was made with correct parameters
        call_kwargs = model._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["stop"] == ["\n"]

    @pytest.mark.asyncio
    async def test_generate_rate_limit_error(self, model: OpenAIModel) -> None:
        """Test handling of rate limit errors."""
        import openai

        model._client.chat.completions.create = AsyncMock(
            side_effect=openai.RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )
        )

        messages = [Message(role="user", content="Hello")]
        with pytest.raises(RateLimitError) as exc_info:
            await model.generate(messages)

        assert "Rate limit" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_generate_token_limit_error(self, model: OpenAIModel) -> None:
        """Test handling of token limit errors."""
        import openai

        model._client.chat.completions.create = AsyncMock(
            side_effect=openai.BadRequestError(
                message="maximum context length exceeded",
                response=MagicMock(status_code=400),
                body=None,
            )
        )

        messages = [Message(role="user", content="Hello")]
        with pytest.raises(TokenLimitError) as exc_info:
            await model.generate(messages)

        assert "context length" in exc_info.value.message


class TestOpenAIModelReasoning:
    """Tests for OpenAIModel reasoning capabilities."""

    @pytest.fixture
    def o1_model(self) -> OpenAIModel:
        """Create an o1 model instance for testing."""
        return OpenAIModel(model="o1", api_key="test-key")

    @pytest.fixture
    def mock_reasoning_response(self) -> MagicMock:
        """Create a mock OpenAI response with reasoning tokens."""
        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(content="Reasoned response"),
                finish_reason="stop",
            )
        ]
        response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=100,
        )
        response.usage.completion_tokens_details = MagicMock(reasoning_tokens=80)
        response.model = "o1"
        return response

    @pytest.mark.asyncio
    async def test_generate_with_reasoning_success(
        self, o1_model: OpenAIModel, mock_reasoning_response: MagicMock
    ) -> None:
        """Test successful reasoning generation."""
        o1_model._client.chat.completions.create = AsyncMock(return_value=mock_reasoning_response)

        messages = [Message(role="user", content="Solve this problem")]
        response = await o1_model.generate_with_reasoning(messages)

        assert isinstance(response, ReasoningResponse)
        assert response.content == "Reasoned response"
        assert response.thinking_tokens == 80
        assert response.usage.reasoning_tokens == 80

    @pytest.mark.asyncio
    async def test_generate_with_reasoning_not_supported(self) -> None:
        """Test that reasoning raises error for non-reasoning models."""
        model = OpenAIModel(model="gpt-4o", api_key="test-key")
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(NotImplementedError) as exc_info:
            await model.generate_with_reasoning(messages)

        assert "gpt-4o" in str(exc_info.value)
        assert "does not support" in str(exc_info.value)


class TestModelFactory:
    """Tests for model factory functions."""

    def test_create_model_openai(self) -> None:
        """Test creating OpenAI model via factory."""
        model = create_model("gpt-4o", api_key="test-key")
        assert isinstance(model, OpenAIModel)
        assert model.model == "gpt-4o"

    def test_create_model_infer_provider(self) -> None:
        """Test that provider is inferred from model name."""
        model = create_model("gpt-4-turbo", api_key="test-key")
        assert isinstance(model, OpenAIModel)

    def test_create_model_explicit_provider(self) -> None:
        """Test creating model with explicit provider."""
        model = create_model("custom-model", provider="openai", api_key="test-key")
        assert isinstance(model, OpenAIModel)
        assert model.model == "custom-model"

    def test_create_model_unknown_provider(self) -> None:
        """Test error when provider cannot be determined."""
        with pytest.raises(ValueError) as exc_info:
            create_model("unknown-model")

        assert "Cannot infer provider" in str(exc_info.value)


class TestTokenCounting:
    """Tests for token counting functionality."""

    @pytest.fixture
    def model(self) -> OpenAIModel:
        """Create a model instance for testing."""
        return OpenAIModel(model="gpt-4o", api_key="test-key")

    @pytest.mark.asyncio
    async def test_count_tokens_with_tiktoken(self, model: OpenAIModel) -> None:
        """Test token counting with tiktoken installed."""
        messages = [
            Message(role="user", content="Hello, how are you?"),
        ]
        count = await model.count_tokens(messages)
        assert count > 0
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_count_tokens_multiple_messages(self, model: OpenAIModel) -> None:
        """Test token counting with multiple messages."""
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        count = await model.count_tokens(messages)
        assert count >= 10  # Should have reasonable token count
