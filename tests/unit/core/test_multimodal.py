"""Tests for multimodal content types and extraction."""

import pytest

from artemis.core.types import ContentPart, ContentType, Message


class TestContentType:
    """Tests for ContentType enum."""

    def test_content_type_values(self):
        """Should have expected values."""
        assert ContentType.TEXT == "text"
        assert ContentType.IMAGE == "image"
        assert ContentType.DOCUMENT == "document"


class TestContentPart:
    """Tests for ContentPart model."""

    def test_text_content_part(self):
        """Should create text content part."""
        part = ContentPart(type=ContentType.TEXT, text="Hello world")
        assert part.type == ContentType.TEXT
        assert part.text == "Hello world"
        assert part.is_text
        assert not part.is_image
        assert not part.is_document

    def test_image_content_part_url(self):
        """Should create image content part with URL."""
        part = ContentPart(
            type=ContentType.IMAGE,
            url="https://example.com/image.png",
            media_type="image/png",
        )
        assert part.type == ContentType.IMAGE
        assert part.url == "https://example.com/image.png"
        assert part.is_image
        assert not part.is_text

    def test_image_content_part_data(self):
        """Should create image content part with data."""
        part = ContentPart(
            type=ContentType.IMAGE,
            data=b"fake image data",
            media_type="image/jpeg",
        )
        assert part.data == b"fake image data"
        assert part.media_type == "image/jpeg"

    def test_document_content_part(self):
        """Should create document content part."""
        part = ContentPart(
            type=ContentType.DOCUMENT,
            data=b"PDF content",
            media_type="application/pdf",
            filename="evidence.pdf",
        )
        assert part.type == ContentType.DOCUMENT
        assert part.filename == "evidence.pdf"
        assert part.is_document

    def test_get_text_from_text(self):
        """Should return text for text content."""
        part = ContentPart(type=ContentType.TEXT, text="Test text")
        assert part.get_text() == "Test text"

    def test_get_text_from_image_with_alt(self):
        """Should return alt text for image."""
        part = ContentPart(
            type=ContentType.IMAGE,
            url="https://example.com/image.png",
            alt_text="A chart showing data",
        )
        assert part.get_text() == "[Image: A chart showing data]"

    def test_get_text_from_document(self):
        """Should return filename for document."""
        part = ContentPart(
            type=ContentType.DOCUMENT,
            filename="report.pdf",
        )
        assert part.get_text() == "[Document: report.pdf]"

    def test_get_text_fallback(self):
        """Should return type as fallback."""
        part = ContentPart(type=ContentType.IMAGE)
        assert part.get_text() == "[image]"


class TestMessageMultimodal:
    """Tests for Message with multimodal content."""

    def test_simple_text_message(self):
        """Should create simple text message."""
        msg = Message(role="user", content="Hello")
        assert msg.content == "Hello"
        assert not msg.is_multimodal
        assert msg.text_content == "Hello"

    def test_multimodal_message_with_image(self):
        """Should create multimodal message with image."""
        msg = Message(
            role="user",
            content="Analyze this chart",
            parts=[
                ContentPart(type=ContentType.TEXT, text="Analyze this chart"),
                ContentPart(
                    type=ContentType.IMAGE,
                    url="https://example.com/chart.png",
                    media_type="image/png",
                ),
            ],
        )
        assert msg.is_multimodal
        assert len(msg.parts) == 2
        assert len(msg.images) == 1

    def test_message_images_property(self):
        """Should return only image parts."""
        msg = Message(
            role="user",
            content="Multiple images",
            parts=[
                ContentPart(type=ContentType.TEXT, text="Text"),
                ContentPart(type=ContentType.IMAGE, url="https://example.com/1.png"),
                ContentPart(type=ContentType.IMAGE, url="https://example.com/2.png"),
            ],
        )
        assert len(msg.images) == 2

    def test_message_documents_property(self):
        """Should return only document parts."""
        msg = Message(
            role="user",
            content="Documents",
            parts=[
                ContentPart(type=ContentType.TEXT, text="Text"),
                ContentPart(
                    type=ContentType.DOCUMENT,
                    filename="doc1.pdf",
                ),
                ContentPart(
                    type=ContentType.DOCUMENT,
                    filename="doc2.pdf",
                ),
            ],
        )
        assert len(msg.documents) == 2

    def test_text_content_combines_parts(self):
        """Should combine text from all parts."""
        msg = Message(
            role="user",
            content="Initial text",
            parts=[
                ContentPart(type=ContentType.TEXT, text="Part 1"),
                ContentPart(type=ContentType.IMAGE, url="https://example.com/img.png"),
                ContentPart(type=ContentType.TEXT, text="Part 2"),
            ],
        )
        text = msg.text_content
        assert "Initial text" in text
        assert "Part 1" in text
        assert "Part 2" in text

    def test_non_multimodal_with_only_text_parts(self):
        """Should not be multimodal with only text parts."""
        msg = Message(
            role="user",
            content="Text message",
            parts=[
                ContentPart(type=ContentType.TEXT, text="Only text"),
            ],
        )
        assert not msg.is_multimodal

    def test_empty_images_for_non_multimodal(self):
        """Should return empty list for non-multimodal message."""
        msg = Message(role="user", content="Hello")
        assert msg.images == []
        assert msg.documents == []


class TestContentAdapters:
    """Tests for content adapters."""

    def test_openai_adapter_text(self):
        """Should format text for OpenAI."""
        from artemis.models.adapters import OpenAIContentAdapter

        adapter = OpenAIContentAdapter()
        parts = [ContentPart(type=ContentType.TEXT, text="Hello")]
        formatted = adapter.format_content(parts)

        assert len(formatted) == 1
        assert formatted[0]["type"] == "text"
        assert formatted[0]["text"] == "Hello"

    def test_openai_adapter_image_url(self):
        """Should format image URL for OpenAI."""
        from artemis.models.adapters import OpenAIContentAdapter

        adapter = OpenAIContentAdapter()
        parts = [
            ContentPart(
                type=ContentType.IMAGE,
                url="https://example.com/image.png",
            ),
        ]
        formatted = adapter.format_content(parts)

        assert len(formatted) == 1
        assert formatted[0]["type"] == "image_url"
        assert formatted[0]["image_url"]["url"] == "https://example.com/image.png"

    def test_openai_adapter_supports_type(self):
        """Should support text and images."""
        from artemis.models.adapters import OpenAIContentAdapter

        adapter = OpenAIContentAdapter()
        assert adapter.supports_type(ContentType.TEXT)
        assert adapter.supports_type(ContentType.IMAGE)
        assert not adapter.supports_type(ContentType.DOCUMENT)

    def test_anthropic_adapter_text(self):
        """Should format text for Anthropic."""
        from artemis.models.adapters import AnthropicContentAdapter

        adapter = AnthropicContentAdapter()
        parts = [ContentPart(type=ContentType.TEXT, text="Hello")]
        formatted = adapter.format_content(parts)

        assert len(formatted) == 1
        assert formatted[0]["type"] == "text"

    def test_anthropic_adapter_supports_documents(self):
        """Should support documents."""
        from artemis.models.adapters import AnthropicContentAdapter

        adapter = AnthropicContentAdapter()
        assert adapter.supports_type(ContentType.DOCUMENT)

    def test_google_adapter_image(self):
        """Should format image for Google."""
        from artemis.models.adapters import GoogleContentAdapter

        adapter = GoogleContentAdapter()
        parts = [
            ContentPart(
                type=ContentType.IMAGE,
                data=b"image data",
                media_type="image/png",
            ),
        ]
        formatted = adapter.format_content(parts)

        assert len(formatted) == 1
        assert "inline_data" in formatted[0]

    def test_text_only_adapter(self):
        """Should convert all content to text."""
        from artemis.models.adapters import TextOnlyAdapter

        adapter = TextOnlyAdapter()
        parts = [
            ContentPart(type=ContentType.TEXT, text="Hello"),
            ContentPart(
                type=ContentType.IMAGE,
                url="https://example.com/img.png",
                alt_text="An image",
            ),
        ]
        formatted = adapter.format_content(parts)

        assert len(formatted) == 2
        assert all(p["type"] == "text" for p in formatted)

    def test_get_adapter_openai(self):
        """Should return OpenAI adapter."""
        from artemis.models.adapters import OpenAIContentAdapter, get_adapter

        adapter = get_adapter("openai")
        assert isinstance(adapter, OpenAIContentAdapter)

    def test_get_adapter_unknown(self):
        """Should return text-only adapter for unknown provider."""
        from artemis.models.adapters import TextOnlyAdapter, get_adapter

        adapter = get_adapter("unknown_provider")
        assert isinstance(adapter, TextOnlyAdapter)


class TestMultimodalEvidence:
    """Tests for multimodal evidence extraction."""

    def test_extraction_type_values(self):
        """Should have expected extraction types."""
        from artemis.core.multimodal_evidence import ExtractionType

        assert ExtractionType.TEXT.value == "text"
        assert ExtractionType.DESCRIPTION.value == "description"
        assert ExtractionType.DATA.value == "data"
        assert ExtractionType.CLAIMS.value == "claims"
        assert ExtractionType.SUMMARY.value == "summary"

    def test_extracted_content_creation(self):
        """Should create extracted content."""
        from artemis.core.multimodal_evidence import ExtractedContent, ExtractionType

        content = ExtractedContent(
            source_type=ContentType.IMAGE,
            extraction_type=ExtractionType.DESCRIPTION,
            text="A bar chart showing data",
            confidence=0.9,
        )

        assert content.source_type == ContentType.IMAGE
        assert content.text == "A bar chart showing data"
        assert content.is_reliable

    def test_extracted_content_unreliable(self):
        """Should detect unreliable extraction."""
        from artemis.core.multimodal_evidence import ExtractedContent, ExtractionType

        content = ExtractedContent(
            source_type=ContentType.IMAGE,
            extraction_type=ExtractionType.DESCRIPTION,
            text="Low confidence result",
            confidence=0.5,
        )

        assert not content.is_reliable

    def test_document_processor_init(self):
        """Should initialize document processor."""
        from artemis.core.multimodal_evidence import DocumentProcessor

        processor = DocumentProcessor(max_pages=5, chunk_size=2000)
        assert processor.max_pages == 5
        assert processor.chunk_size == 2000

    def test_document_processor_chunk_text(self):
        """Should chunk text content."""
        from artemis.core.multimodal_evidence import DocumentProcessor

        processor = DocumentProcessor(chunk_size=50)
        chunks = processor._chunk_text("Line 1\nLine 2\nLine 3\n" * 10)

        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
