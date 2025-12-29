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
    """Base class for multimodal content adapters."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider this adapter supports."""
        pass

    @abstractmethod
    def supports_type(self, content_type: ContentType) -> bool:
        """Check if this adapter supports the given content type."""
        pass

    @abstractmethod
    def format_content(self, parts: list[ContentPart]) -> list[dict[str, Any]]:
        """Format content parts for the provider's API."""
        pass

    def format_message(self, message: Message) -> dict[str, Any]:
        """Format a complete message for the provider."""
        if not message.parts or not message.is_multimodal:
            # Simple text message
            return {"role": message.role, "content": message.content}

        # Multimodal message
        formatted_content = self.format_content(message.parts)
        return {"role": message.role, "content": formatted_content}


class OpenAIContentAdapter(ContentAdapter):
    """Content adapter for OpenAI's vision API."""

    @property
    def provider_name(self) -> str:
        return "openai"

    def supports_type(self, content_type: ContentType) -> bool:
        return content_type in (ContentType.TEXT, ContentType.IMAGE)

    def format_content(self, parts: list[ContentPart]) -> list[dict[str, Any]]:
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
    """Content adapter for Anthropic's Claude vision API."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def supports_type(self, content_type: ContentType) -> bool:
        return content_type in (ContentType.TEXT, ContentType.IMAGE, ContentType.DOCUMENT)

    def format_content(self, parts: list[ContentPart]) -> list[dict[str, Any]]:
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
    """Content adapter for Google's Gemini multimodal API."""

    @property
    def provider_name(self) -> str:
        return "google"

    def supports_type(self, content_type: ContentType) -> bool:
        return content_type in (ContentType.TEXT, ContentType.IMAGE, ContentType.DOCUMENT)

    def format_content(self, parts: list[ContentPart]) -> list[dict[str, Any]]:
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
    """Fallback adapter that converts all content to text."""

    @property
    def provider_name(self) -> str:
        return "text_only"

    def supports_type(self, content_type: ContentType) -> bool:
        return content_type == ContentType.TEXT

    def format_content(self, parts: list[ContentPart]) -> list[dict[str, Any]]:
        formatted = []

        for part in parts:
            text = part.get_text()
            formatted.append({"type": "text", "text": text})

        return formatted

    def format_message(self, message: Message) -> dict[str, Any]:
        # works but could be cleaner - just concatenates everything
        if not message.parts or not message.is_multimodal:
            return {"role": message.role, "content": message.content}

        # Combine all parts into single text
        texts = [message.content] if message.content else []
        for part in message.parts:
            texts.append(part.get_text())

        return {"role": message.role, "content": " ".join(texts)}


def get_adapter(provider: str) -> ContentAdapter:
    """Get the appropriate content adapter for a provider."""
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
