"""Multimodal evidence extraction for debates.

Provides tools for extracting evidence from images and documents
to support debate arguments with visual content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from artemis.core.types import ContentPart, ContentType
from artemis.utils.logging import get_logger

if TYPE_CHECKING:
    from artemis.models.base import BaseModel

logger = get_logger(__name__)


class ExtractionType(str, Enum):
    """Type of evidence extraction."""

    TEXT = "text"  # Extract text from document
    DESCRIPTION = "description"  # Describe image content
    DATA = "data"  # Extract data/statistics
    CLAIMS = "claims"  # Extract factual claims
    SUMMARY = "summary"  # Summarize content


@dataclass
class ExtractedContent:
    """Content extracted from multimodal source.

    Example:
        ```python
        content = ExtractedContent(
            source_type=ContentType.IMAGE,
            extraction_type=ExtractionType.DESCRIPTION,
            text="A bar chart showing increasing AI adoption...",
            confidence=0.85,
            metadata={"model": "gpt-4o"},
        )
        ```
    """

    source_type: ContentType
    extraction_type: ExtractionType
    text: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_reliable(self) -> bool:
        """Check if extraction is reliable (confidence > 0.7)."""
        return self.confidence >= 0.7


class MultimodalEvidenceExtractor:
    """Extracts evidence from images and documents.

    Uses LLM vision capabilities to analyze visual content
    and extract relevant evidence for debates.

    Example:
        ```python
        extractor = MultimodalEvidenceExtractor(model=vision_model)

        # Extract from image
        content = ContentPart(
            type=ContentType.IMAGE,
            url="https://example.com/chart.png",
        )
        result = await extractor.extract(content)
        print(result.text)
        ```
    """

    EXTRACTION_PROMPTS = {
        ExtractionType.DESCRIPTION: """Analyze this image and provide a detailed description.
Focus on:
1. What the image shows
2. Key data or information visible
3. Any text or labels
4. Overall message or implication

Provide a clear, factual description.""",

        ExtractionType.DATA: """Extract all data, statistics, and numerical information from this image.
Format as:
- List each data point
- Include units and context
- Note any trends or patterns
- Identify the source if visible

Be precise and accurate.""",

        ExtractionType.CLAIMS: """Identify all factual claims supported by this image.
For each claim:
1. State the claim clearly
2. Note the supporting evidence in the image
3. Rate confidence (high/medium/low)

Focus on verifiable claims.""",

        ExtractionType.SUMMARY: """Provide a concise summary of this content.
Include:
- Main topic or subject
- Key findings or information
- Relevance to debates or arguments
- Any limitations or caveats

Keep it brief but comprehensive.""",

        ExtractionType.TEXT: """Extract all text visible in this image or document.
Include:
- Headers and titles
- Body text
- Labels and captions
- Any footnotes or citations

Maintain the original structure where possible.""",
    }

    def __init__(
        self,
        model: "BaseModel | None" = None,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
    ) -> None:
        """Initialize the extractor.

        Args:
            model: Pre-configured vision model.
            model_name: Model name if creating new instance.
            api_key: API key for model.
        """
        self._model = model
        self._model_name = model_name
        self._api_key = api_key

    async def _get_model(self) -> "BaseModel":
        """Get or create the model instance."""
        if self._model:
            return self._model

        from artemis.models.base import ModelRegistry

        self._model = ModelRegistry.create(
            self._model_name,
            api_key=self._api_key,
        )
        return self._model

    async def extract(
        self,
        content: ContentPart,
        extraction_type: ExtractionType = ExtractionType.DESCRIPTION,
    ) -> ExtractedContent:
        """Extract evidence from multimodal content.

        Args:
            content: The content part to analyze.
            extraction_type: Type of extraction to perform.

        Returns:
            ExtractedContent with extracted text and metadata.
        """
        if content.type == ContentType.TEXT:
            # Text content - just return as-is
            return ExtractedContent(
                source_type=ContentType.TEXT,
                extraction_type=extraction_type,
                text=content.text or "",
                confidence=1.0,
            )

        # Get the model
        model = await self._get_model()

        # Build the prompt
        prompt = self.EXTRACTION_PROMPTS.get(
            extraction_type,
            self.EXTRACTION_PROMPTS[ExtractionType.DESCRIPTION],
        )

        # Create multimodal message
        from artemis.core.types import Message

        # Build message with content part
        message = Message(
            role="user",
            content=prompt,
            parts=[
                ContentPart(type=ContentType.TEXT, text=prompt),
                content,
            ],
        )

        try:
            # Generate response (model should handle multimodal)
            response = await model.generate([message])

            logger.info(
                "Extracted content from multimodal source",
                source_type=content.type.value,
                extraction_type=extraction_type.value,
            )

            return ExtractedContent(
                source_type=content.type,
                extraction_type=extraction_type,
                text=response.content,
                confidence=0.85,  # LLM extractions have inherent uncertainty
                metadata={
                    "model": model.model,
                    "media_type": content.media_type,
                    "filename": content.filename,
                },
            )

        except Exception as e:
            logger.error(
                "Failed to extract content",
                source_type=content.type.value,
                error=str(e),
            )
            return ExtractedContent(
                source_type=content.type,
                extraction_type=extraction_type,
                text=f"[Extraction failed: {content.type.value}]",
                confidence=0.0,
                metadata={"error": str(e)},
            )

    async def extract_all(
        self,
        content: ContentPart,
    ) -> list[ExtractedContent]:
        """Perform all extraction types on content.

        Args:
            content: The content part to analyze.

        Returns:
            List of ExtractedContent for each extraction type.
        """
        results = []
        for extraction_type in ExtractionType:
            result = await self.extract(content, extraction_type)
            results.append(result)
        return results

    async def describe_for_debate(
        self,
        content: ContentPart,
        debate_topic: str,
    ) -> ExtractedContent:
        """Extract content relevant to a specific debate topic.

        Args:
            content: The content part to analyze.
            debate_topic: The topic of the debate.

        Returns:
            ExtractedContent focused on debate relevance.
        """
        model = await self._get_model()

        prompt = f"""Analyze this content in the context of the following debate topic:

Debate Topic: {debate_topic}

Provide:
1. Description of the content
2. Key data or claims relevant to the debate
3. How this evidence could support arguments (for or against)
4. Any limitations or caveats

Be objective and analytical."""

        from artemis.core.types import Message

        message = Message(
            role="user",
            content=prompt,
            parts=[
                ContentPart(type=ContentType.TEXT, text=prompt),
                content,
            ],
        )

        try:
            response = await model.generate([message])

            return ExtractedContent(
                source_type=content.type,
                extraction_type=ExtractionType.SUMMARY,
                text=response.content,
                confidence=0.85,
                metadata={
                    "model": model.model,
                    "debate_topic": debate_topic,
                },
            )

        except Exception as e:
            logger.error(
                "Failed to extract debate-relevant content",
                error=str(e),
            )
            return ExtractedContent(
                source_type=content.type,
                extraction_type=ExtractionType.SUMMARY,
                text=f"[Extraction failed]",
                confidence=0.0,
                metadata={"error": str(e)},
            )


class DocumentProcessor:
    """Processes documents for evidence extraction.

    Handles PDF, text, and other document formats.

    Example:
        ```python
        processor = DocumentProcessor()
        pages = await processor.process_document(content)
        for page in pages:
            print(f"Page {page['number']}: {page['text'][:100]}...")
        ```
    """

    def __init__(
        self,
        max_pages: int = 10,
        chunk_size: int = 4000,
    ) -> None:
        """Initialize the processor.

        Args:
            max_pages: Maximum pages to process.
            chunk_size: Maximum characters per chunk.
        """
        self.max_pages = max_pages
        self.chunk_size = chunk_size

    async def process_document(
        self,
        content: ContentPart,
    ) -> list[dict[str, Any]]:
        """Process a document into extractable chunks.

        Args:
            content: Document content part.

        Returns:
            List of page/chunk dictionaries.
        """
        if content.type != ContentType.DOCUMENT:
            return []

        if not content.data:
            logger.warning("Document has no data")
            return []

        # Basic PDF text extraction (simplified)
        # In production, would use PyPDF2, pdfplumber, etc.
        if content.media_type == "application/pdf":
            return self._process_pdf(content.data)

        # Text-based documents
        if content.media_type in ("text/plain", "text/markdown", "text/html"):
            text = content.data.decode("utf-8", errors="ignore")
            return self._chunk_text(text)

        # Unknown format - return metadata only
        return [{
            "number": 1,
            "text": f"[Binary document: {content.filename or 'unnamed'}]",
            "type": content.media_type,
        }]

    def _process_pdf(self, data: bytes) -> list[dict[str, Any]]:
        """Process PDF data (simplified stub)."""
        # In production, use PyPDF2 or similar
        # This is a placeholder that returns metadata
        return [{
            "number": 1,
            "text": "[PDF content - requires PDF library for extraction]",
            "type": "application/pdf",
            "size": len(data),
        }]

    def _chunk_text(self, text: str) -> list[dict[str, Any]]:
        """Split text into manageable chunks."""
        chunks = []
        lines = text.split("\n")
        current_chunk = ""
        chunk_num = 1

        for line in lines:
            if len(current_chunk) + len(line) > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        "number": chunk_num,
                        "text": current_chunk.strip(),
                        "type": "text/plain",
                    })
                    chunk_num += 1
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"

        if current_chunk:
            chunks.append({
                "number": chunk_num,
                "text": current_chunk.strip(),
                "type": "text/plain",
            })

        return chunks[:self.max_pages]


class ImageAnalyzer:
    """Analyzes images for debate-relevant content.

    Provides specialized analysis for charts, graphs,
    and other visual evidence.

    Example:
        ```python
        analyzer = ImageAnalyzer(model=vision_model)
        analysis = await analyzer.analyze_chart(image_content)
        print(f"Chart type: {analysis['chart_type']}")
        print(f"Data: {analysis['data_points']}")
        ```
    """

    CHART_ANALYSIS_PROMPT = """Analyze this chart/graph image:

1. Chart Type: (bar, line, pie, scatter, etc.)
2. Title/Subject: What does it represent?
3. Axes: What are the X and Y axes? What units?
4. Data Points: List the key data values visible
5. Trends: What trends or patterns are shown?
6. Source: Is a source cited?
7. Limitations: Any caveats or limitations?

Be precise with numbers and labels."""

    def __init__(
        self,
        model: "BaseModel | None" = None,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
    ) -> None:
        """Initialize the analyzer."""
        self._extractor = MultimodalEvidenceExtractor(
            model=model,
            model_name=model_name,
            api_key=api_key,
        )

    async def analyze_chart(
        self,
        content: ContentPart,
    ) -> dict[str, Any]:
        """Analyze a chart or graph image.

        Args:
            content: Image content part.

        Returns:
            Dictionary with chart analysis.
        """
        model = await self._extractor._get_model()

        from artemis.core.types import Message

        message = Message(
            role="user",
            content=self.CHART_ANALYSIS_PROMPT,
            parts=[
                ContentPart(type=ContentType.TEXT, text=self.CHART_ANALYSIS_PROMPT),
                content,
            ],
        )

        try:
            response = await model.generate([message])

            return {
                "raw_analysis": response.content,
                "source_type": "chart",
                "confidence": 0.85,
            }

        except Exception as e:
            logger.error("Chart analysis failed", error=str(e))
            return {
                "raw_analysis": f"[Analysis failed: {e}]",
                "source_type": "chart",
                "confidence": 0.0,
                "error": str(e),
            }

    async def compare_images(
        self,
        image1: ContentPart,
        image2: ContentPart,
    ) -> dict[str, Any]:
        """Compare two images for debate evidence.

        Args:
            image1: First image content.
            image2: Second image content.

        Returns:
            Dictionary with comparison analysis.
        """
        model = await self._extractor._get_model()

        prompt = """Compare these two images:

1. Similarities: What do they have in common?
2. Differences: Key differences between them?
3. Contradictions: Do they contradict each other?
4. Context: How might they relate to each other?
5. Evidence Value: What do they prove or suggest together?

Be analytical and objective."""

        from artemis.core.types import Message

        message = Message(
            role="user",
            content=prompt,
            parts=[
                ContentPart(type=ContentType.TEXT, text=prompt),
                image1,
                image2,
            ],
        )

        try:
            response = await model.generate([message])

            return {
                "comparison": response.content,
                "confidence": 0.8,
            }

        except Exception as e:
            logger.error("Image comparison failed", error=str(e))
            return {
                "comparison": f"[Comparison failed: {e}]",
                "confidence": 0.0,
                "error": str(e),
            }
