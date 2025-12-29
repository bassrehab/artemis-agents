"""LLM-based extraction for causal links and evidence.

Replaces brittle regex-based extraction with robust LLM parsing.
Works across all providers (OpenAI, Anthropic, Gemini).
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from artemis.core.types import CausalLink, Evidence
from artemis.prompts import get_prompt
from artemis.utils.logging import get_logger

if TYPE_CHECKING:
    from artemis.models.base import BaseModel

logger = get_logger(__name__)

# Cache for extraction results (content_hash -> result)
_extraction_cache: dict[str, Any] = {}


def _hash_content(content: str) -> str:
    """Create a hash of content for caching."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class LLMCausalExtractor:
    """Extract causal relationships using LLM analysis."""

    def __init__(
        self,
        model: BaseModel | None = None,
        model_name: str = "gpt-4o-mini",
        use_cache: bool = True,
    ):
        self._model = model
        self._model_name = model_name
        self._use_cache = use_cache

    async def _get_model(self) -> BaseModel:
        """Lazy-load model if not provided."""
        if self._model is None:
            from artemis.models.base import ModelRegistry

            self._model = ModelRegistry.create(self._model_name)
        return self._model

    async def extract(self, content: str) -> list[CausalLink]:
        """Extract causal links from content."""
        if not content or len(content.strip()) < 20:
            return []

        # Check cache
        cache_key = f"causal_{_hash_content(content)}"
        if self._use_cache and cache_key in _extraction_cache:
            logger.debug("Causal extraction cache hit", key=cache_key)
            return _extraction_cache[cache_key]

        try:
            model = await self._get_model()
            prompt = get_prompt("extraction.causal_extraction", content=content)

            from artemis.core.types import Message

            response = await model.generate([Message(role="user", content=prompt)])

            # Parse JSON response
            result = self._parse_response(response.content)
            links = self._convert_to_causal_links(result.get("causal_links", []))

            # Cache result
            if self._use_cache:
                _extraction_cache[cache_key] = links

            logger.debug(
                "Causal extraction complete",
                links_found=len(links),
            )
            return links

        except Exception as e:
            logger.warning("Causal extraction failed", error=str(e))
            return []

    def _parse_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        # Handle potential markdown code blocks
        content = response.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (``` markers)
            content = "\n".join(lines[1:-1])
            if content.startswith("json"):
                content = content[4:].strip()

        return json.loads(content)

    def _convert_to_causal_links(self, raw_links: list[dict]) -> list[CausalLink]:
        """Convert raw extraction to CausalLink objects."""
        links = []
        for item in raw_links:
            try:
                links.append(
                    CausalLink(
                        cause=item.get("cause", "")[:100],
                        effect=item.get("effect", "")[:100],
                        strength=float(item.get("strength", 0.5)),
                    )
                )
            except (KeyError, ValueError) as e:
                logger.debug("Skipping invalid causal link", error=str(e))
                continue
        return links


class LLMEvidenceExtractor:
    """Extract evidence using LLM analysis."""

    def __init__(
        self,
        model: BaseModel | None = None,
        model_name: str = "gpt-4o-mini",
        use_cache: bool = True,
    ):
        self._model = model
        self._model_name = model_name
        self._use_cache = use_cache

    async def _get_model(self) -> BaseModel:
        """Lazy-load model if not provided."""
        if self._model is None:
            from artemis.models.base import ModelRegistry

            self._model = ModelRegistry.create(self._model_name)
        return self._model

    async def extract(self, content: str) -> list[Evidence]:
        """Extract evidence from content."""
        if not content or len(content.strip()) < 20:
            return []

        # Check cache
        cache_key = f"evidence_{_hash_content(content)}"
        if self._use_cache and cache_key in _extraction_cache:
            logger.debug("Evidence extraction cache hit", key=cache_key)
            return _extraction_cache[cache_key]

        try:
            model = await self._get_model()
            prompt = get_prompt("extraction.evidence_extraction", content=content)

            from artemis.core.types import Message

            response = await model.generate([Message(role="user", content=prompt)])

            # Parse JSON response
            result = self._parse_response(response.content)
            evidence = self._convert_to_evidence(result.get("evidence", []))

            # Cache result
            if self._use_cache:
                _extraction_cache[cache_key] = evidence

            logger.debug(
                "Evidence extraction complete",
                evidence_found=len(evidence),
            )
            return evidence

        except Exception as e:
            logger.warning("Evidence extraction failed", error=str(e))
            return []

    def _parse_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        content = response.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
            if content.startswith("json"):
                content = content[4:].strip()

        return json.loads(content)

    def _convert_to_evidence(self, raw_evidence: list[dict]) -> list[Evidence]:
        """Convert raw extraction to Evidence objects."""
        evidence_list = []
        for item in raw_evidence:
            try:
                evidence_list.append(
                    Evidence(
                        type=item.get("type", "quote"),
                        content=item.get("content", "")[:500],
                        source=item.get("source"),
                        verified=False,
                    )
                )
            except (KeyError, ValueError) as e:
                logger.debug("Skipping invalid evidence", error=str(e))
                continue
        return evidence_list


class HybridCausalExtractor:
    """Hybrid extractor: tries regex first, falls back to LLM."""

    def __init__(
        self,
        model: BaseModel | None = None,
        model_name: str = "gpt-4o-mini",
        use_cache: bool = True,
    ):
        from artemis.core.causal import CausalExtractor as RegexExtractor

        self._regex_extractor = RegexExtractor()
        self._llm_extractor = LLMCausalExtractor(
            model=model,
            model_name=model_name,
            use_cache=use_cache,
        )

    async def extract(self, content: str) -> list[CausalLink]:
        """Extract causal links, trying regex first then LLM."""
        # Try regex first (fast, cheap)
        regex_results = self._regex_extractor.extract(content)
        regex_links = [link for link, _ in regex_results]

        if regex_links:
            logger.debug("Regex extraction succeeded", links=len(regex_links))
            return regex_links

        # Fall back to LLM
        return await self._llm_extractor.extract(content)


class HybridEvidenceExtractor:
    """Hybrid extractor: tries regex first, falls back to LLM."""

    def __init__(
        self,
        model: BaseModel | None = None,
        model_name: str = "gpt-4o-mini",
        use_cache: bool = True,
    ):
        from artemis.core.evidence import EvidenceExtractor as RegexExtractor

        self._regex_extractor = RegexExtractor()
        self._llm_extractor = LLMEvidenceExtractor(
            model=model,
            model_name=model_name,
            use_cache=use_cache,
        )

    async def extract(self, content: str) -> list[Evidence]:
        """Extract evidence, trying regex first then LLM."""
        # Try regex first (fast, cheap)
        regex_results = self._regex_extractor.extract(content)

        if regex_results:
            logger.debug("Regex extraction succeeded", evidence=len(regex_results))
            # Convert ExtractedEvidence to Evidence
            return [
                Evidence(
                    type=e.evidence_type.value,
                    content=e.content,
                    source=e.source,
                    verified=False,
                )
                for e in regex_results
            ]

        # Fall back to LLM
        return await self._llm_extractor.extract(content)


def clear_extraction_cache() -> None:
    """Clear the extraction cache."""
    global _extraction_cache
    _extraction_cache = {}
    logger.debug("Extraction cache cleared")
