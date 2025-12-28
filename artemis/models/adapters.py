"""Multimodal content adapters for LLM providers.

Provides adapters for formatting multimodal content (images, documents)
for specific LLM provider APIs.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from typing import Any

from artemis.core.types import ContentPart, ContentType, Message
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


class ContentAdapter(ABC):
    """Abstract base class for multimodal content adapters.

    Adapters format multimodal content for specific LLM providers.

    Example:
        ```python
        adapter = OpenAIContentAdapter()
        if adapter.supports_type(ContentType.IMAGE):
            formatted = adapter.format_content(message.parts)
        ```
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider this adapter supports."""
        pass

    @abstractmethod
    def supports_type(self, content_type: ContentType) -> bool:
        """Check if this adapter supports the given content type.

        Args:
            content_type: The content type to check.

        Returns:
            True if supported, False otherwise.
        """
        pass

    @abstractmethod
    def format_content(self, parts: list[ContentPart]) -> list[dict[str, Any]]:
        """Format content parts for the provider's API.

        Args:
            parts: List of content parts to format.

        Returns:
            List of formatted content dictionaries.
        """
        pass

    def format_message(self, message: Message) -> dict[str, Any]:
        """Format a complete message for the provider.

        Args:
            message: The message to format.

        Returns:
            Formatted message dictionary.
        """
        if not message.parts or not message.is_multimodal:
            # Simple text message
            return {"role": message.role, "content": message.content}

        # Multimodal message
        formatted_content = self.format_content(message.parts)
        return {"role": message.role, "content": formatted_content}


class OpenAIContentAdapter(ContentAdapter):
    """Content adapter for OpenAI's vision API.

    Formats content for GPT-4 Vision and similar models.

    Example:
        ```python
        adapter = OpenAIContentAdapter()
        messages = [adapter.format_message(msg) for msg in messages]
        ```
    """

    @property
    def provider_name(self) -> str:
        return "openai"

    def supports_type(self, content_type: ContentType) -> bool:
        """OpenAI supports text and images."""
        return content_type in (ContentType.TEXT, ContentType.IMAGE)

    def format_content(self, parts: list[ContentPart]) -> list[dict[str, Any]]:
        """Format content for OpenAI API."""
        formatted = []

        for part in parts:
            if part.type == ContentType.TEXT:
                formatted.append({"type": "text", "text": part.text or ""})

            elif part.type == ContentType.IMAGE:
                if part.url:
                    formatted.append({
                        "type": "image_url",
                        "image_url": {"url": part.url},
                    })
                elif part.data:
                    # Base64 encoded data
                    b64_data = base64.b64encode(part.data).decode("utf-8")
                    media_type = part.media_type or "image/png"
                    data_url = f"data:{media_type};base64,{b64_data}"
                    formatted.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })

            elif part.type == ContentType.DOCUMENT:
                # OpenAI doesn't natively support documents
                # Convert to text description
                desc = f"[Document: {part.filename or 'unnamed'}]"
                if part.alt_text:
                    desc = f"{desc} - {part.alt_text}"
                formatted.append({"type": "text", "text": desc})
                logger.warning(
                    "OpenAI does not support document content, converted to text",
                    filename=part.filename,
                )

        return formatted


class AnthropicContentAdapter(ContentAdapter):
    """Content adapter for Anthropic's Claude vision API.

    Formats content for Claude 3 and similar models.

    Example:
        ```python
        adapter = AnthropicContentAdapter()
        messages = [adapter.format_message(msg) for msg in messages]
        ```
    """

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def supports_type(self, content_type: ContentType) -> bool:
        """Anthropic supports text, images, and PDFs."""
        return content_type in (ContentType.TEXT, ContentType.IMAGE, ContentType.DOCUMENT)

    def format_content(self, parts: list[ContentPart]) -> list[dict[str, Any]]:
        """Format content for Anthropic API."""
        formatted = []

        for part in parts:
            if part.type == ContentType.TEXT:
                formatted.append({"type": "text", "text": part.text or ""})

            elif part.type == ContentType.IMAGE:
                if part.data:
                    b64_data = base64.b64encode(part.data).decode("utf-8")
                    formatted.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.media_type or "image/png",
                            "data": b64_data,
                        },
                    })
                elif part.url:
                    # Anthropic prefers base64, but URLs can work
                    formatted.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": part.url,
                        },
                    })

            elif part.type == ContentType.DOCUMENT:
                if part.data and part.media_type == "application/pdf":
                    # Claude 3 supports PDF natively
                    b64_data = base64.b64encode(part.data).decode("utf-8")
                    formatted.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": b64_data,
                        },
                    })
                else:
                    # Non-PDF documents - convert to text description
                    desc = f"[Document: {part.filename or 'unnamed'}]"
                    formatted.append({"type": "text", "text": desc})

        return formatted


class GoogleContentAdapter(ContentAdapter):
    """Content adapter for Google's Gemini multimodal API.

    Formats content for Gemini Pro Vision and similar models.

    Example:
        ```python
        adapter = GoogleContentAdapter()
        messages = [adapter.format_message(msg) for msg in messages]
        ```
    """

    @property
    def provider_name(self) -> str:
        return "google"

    def supports_type(self, content_type: ContentType) -> bool:
        """Google Gemini supports text, images, and documents."""
        return content_type in (ContentType.TEXT, ContentType.IMAGE, ContentType.DOCUMENT)

    def format_content(self, parts: list[ContentPart]) -> list[dict[str, Any]]:
        """Format content for Gemini API."""
        formatted = []

        for part in parts:
            if part.type == ContentType.TEXT:
                formatted.append({"text": part.text or ""})

            elif part.type == ContentType.IMAGE:
                if part.data:
                    b64_data = base64.b64encode(part.data).decode("utf-8")
                    formatted.append({
                        "inline_data": {
                            "mime_type": part.media_type or "image/png",
                            "data": b64_data,
                        },
                    })
                elif part.url:
                    # Gemini supports file URIs
                    formatted.append({
                        "file_data": {
                            "mime_type": part.media_type or "image/png",
                            "file_uri": part.url,
                        },
                    })

            elif part.type == ContentType.DOCUMENT:
                if part.data:
                    b64_data = base64.b64encode(part.data).decode("utf-8")
                    formatted.append({
                        "inline_data": {
                            "mime_type": part.media_type or "application/pdf",
                            "data": b64_data,
                        },
                    })
                elif part.url:
                    formatted.append({
                        "file_data": {
                            "mime_type": part.media_type or "application/pdf",
                            "file_uri": part.url,
                        },
                    })

        return formatted


class TextOnlyAdapter(ContentAdapter):
    """Fallback adapter that converts all content to text.

    Use for models that don't support multimodal content.

    Example:
        ```python
        adapter = TextOnlyAdapter()
        # All content converted to text descriptions
        formatted = adapter.format_content(parts)
        ```
    """

    @property
    def provider_name(self) -> str:
        return "text_only"

    def supports_type(self, content_type: ContentType) -> bool:
        """Only text is fully supported, others are converted."""
        return content_type == ContentType.TEXT

    def format_content(self, parts: list[ContentPart]) -> list[dict[str, Any]]:
        """Convert all content to text."""
        formatted = []

        for part in parts:
            text = part.get_text()
            formatted.append({"type": "text", "text": text})

        return formatted

    def format_message(self, message: Message) -> dict[str, Any]:
        """Format message as text only."""
        if not message.parts or not message.is_multimodal:
            return {"role": message.role, "content": message.content}

        # Combine all parts into single text
        texts = [message.content] if message.content else []
        for part in message.parts:
            texts.append(part.get_text())

        return {"role": message.role, "content": " ".join(texts)}


def get_adapter(provider: str) -> ContentAdapter:
    """Get the appropriate content adapter for a provider.

    Args:
        provider: Provider name (openai, anthropic, google, etc.)

    Returns:
        Appropriate ContentAdapter instance.

    Example:
        ```python
        adapter = get_adapter("openai")
        formatted = adapter.format_message(message)
        ```
    """
    adapters = {
        "openai": OpenAIContentAdapter,
        "anthropic": AnthropicContentAdapter,
        "google": GoogleContentAdapter,
        "gemini": GoogleContentAdapter,  # Alias
        "text": TextOnlyAdapter,
    }

    adapter_class = adapters.get(provider.lower())
    if adapter_class:
        return adapter_class()

    # Default to text-only for unknown providers
    logger.warning(
        "Unknown provider, using text-only adapter",
        provider=provider,
    )
    return TextOnlyAdapter()
