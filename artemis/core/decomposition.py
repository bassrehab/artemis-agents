"""Topic decomposition for hierarchical debates.

Provides strategies for breaking down complex topics into
sub-topics suitable for focused sub-debates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from artemis.core.types import (
    DecompositionStrategy,
    HierarchicalContext,
    SubDebateSpec,
)
from artemis.utils.logging import get_logger

if TYPE_CHECKING:
    from artemis.models.base import BaseModel

logger = get_logger(__name__)


class TopicDecomposer(ABC):
    """Abstract base class for topic decomposition.

    Implementations determine how a complex topic is broken down
    into sub-topics for hierarchical debates.
    """

    @property
    @abstractmethod
    def strategy(self) -> DecompositionStrategy:
        """The decomposition strategy used."""
        pass

    @abstractmethod
    async def decompose(
        self,
        topic: str,
        context: HierarchicalContext | None = None,
        max_subtopics: int = 5,
    ) -> list[SubDebateSpec]:
        """Decompose a topic into sub-debate specifications.

        Args:
            topic: The topic to decompose.
            context: Optional hierarchical context.
            max_subtopics: Maximum number of sub-topics.

        Returns:
            List of SubDebateSpec for sub-debates.
        """
        pass


class ManualDecomposer(TopicDecomposer):
    """Decomposer that uses manually specified sub-topics.

    Example:
        ```python
        decomposer = ManualDecomposer([
            SubDebateSpec(aspect="Economic impact", weight=0.4),
            SubDebateSpec(aspect="Social implications", weight=0.3),
            SubDebateSpec(aspect="Technical feasibility", weight=0.3),
        ])
        specs = await decomposer.decompose("Should we implement UBI?")
        ```
    """

    def __init__(self, specs: list[SubDebateSpec]) -> None:
        """Initialize with predefined specs.

        Args:
            specs: Pre-defined sub-debate specifications.
        """
        self._specs = specs

    @property
    def strategy(self) -> DecompositionStrategy:
        return DecompositionStrategy.MANUAL

    async def decompose(
        self,
        topic: str,
        context: HierarchicalContext | None = None,
        max_subtopics: int = 5,
    ) -> list[SubDebateSpec]:
        """Return the predefined specs (up to max)."""
        return self._specs[:max_subtopics]


class RuleBasedDecomposer(TopicDecomposer):
    """Decomposer using rule-based patterns.

    Identifies common debate dimensions and creates sub-topics
    based on known patterns.

    Example:
        ```python
        decomposer = RuleBasedDecomposer()
        specs = await decomposer.decompose("Should AI be regulated?")
        # Returns specs for economic, ethical, practical dimensions
        ```
    """

    # Common debate dimensions
    DEFAULT_DIMENSIONS = [
        ("Economic implications", 0.25, "Analyze costs, benefits, and economic impact"),
        ("Ethical considerations", 0.25, "Examine moral and ethical aspects"),
        ("Practical feasibility", 0.25, "Assess implementation challenges"),
        ("Social impact", 0.25, "Consider effects on society and communities"),
    ]

    # Topic patterns that trigger specific dimensions
    TOPIC_PATTERNS = {
        "technology": [
            ("Technical viability", 0.3, "Assess technical requirements and challenges"),
            ("Security implications", 0.2, "Evaluate security risks and safeguards"),
            ("Privacy concerns", 0.2, "Examine data privacy aspects"),
            ("Innovation potential", 0.15, "Consider future possibilities"),
            ("Adoption barriers", 0.15, "Identify implementation obstacles"),
        ],
        "policy": [
            ("Economic impact", 0.25, "Analyze economic effects"),
            ("Legal framework", 0.2, "Examine legal requirements"),
            ("Political feasibility", 0.2, "Assess political landscape"),
            ("Public opinion", 0.15, "Consider public sentiment"),
            ("Implementation", 0.2, "Evaluate practical execution"),
        ],
        "environment": [
            ("Environmental impact", 0.3, "Assess ecological effects"),
            ("Economic cost-benefit", 0.25, "Analyze financial implications"),
            ("Long-term sustainability", 0.2, "Consider long-term viability"),
            ("Behavioral change", 0.15, "Examine required changes"),
            ("Technological solutions", 0.1, "Evaluate tech approaches"),
        ],
    }

    def __init__(
        self,
        custom_dimensions: list[tuple[str, float, str]] | None = None,
    ) -> None:
        """Initialize the decomposer.

        Args:
            custom_dimensions: Optional custom dimensions as (name, weight, desc).
        """
        self._custom_dimensions = custom_dimensions

    @property
    def strategy(self) -> DecompositionStrategy:
        return DecompositionStrategy.RULE_BASED

    async def decompose(
        self,
        topic: str,
        context: HierarchicalContext | None = None,
        max_subtopics: int = 5,
    ) -> list[SubDebateSpec]:
        """Decompose using rule-based pattern matching."""
        dimensions = self._select_dimensions(topic)

        specs = []
        for name, weight, desc in dimensions[:max_subtopics]:
            # Construct sub-topic
            aspect = f"{name} of: {topic}"

            specs.append(
                SubDebateSpec(
                    aspect=aspect,
                    weight=weight,
                    description=desc,
                )
            )

        # Normalize weights
        total_weight = sum(s.weight for s in specs)
        if total_weight > 0:
            for spec in specs:
                spec.weight = spec.weight / total_weight

        return specs

    def _select_dimensions(
        self, topic: str
    ) -> list[tuple[str, float, str]]:
        """Select appropriate dimensions for the topic."""
        if self._custom_dimensions:
            return self._custom_dimensions

        topic_lower = topic.lower()

        # Check for pattern matches
        for pattern, dimensions in self.TOPIC_PATTERNS.items():
            if pattern in topic_lower:
                return dimensions

        # Check for specific keywords
        if any(kw in topic_lower for kw in ["ai", "technology", "software", "digital"]):
            return self.TOPIC_PATTERNS["technology"]

        if any(kw in topic_lower for kw in ["law", "regulation", "policy", "government"]):
            return self.TOPIC_PATTERNS["policy"]

        if any(kw in topic_lower for kw in ["climate", "environment", "sustainable"]):
            return self.TOPIC_PATTERNS["environment"]

        return self.DEFAULT_DIMENSIONS


class LLMTopicDecomposer(TopicDecomposer):
    """Decomposer that uses an LLM to identify sub-topics.

    Uses an LLM to intelligently decompose topics based on
    the specific content and context.

    Example:
        ```python
        decomposer = LLMTopicDecomposer(model=my_model)
        specs = await decomposer.decompose(
            "Should genetic engineering be used to prevent diseases?",
            max_subtopics=4,
        )
        ```
    """

    DECOMPOSITION_PROMPT = """Analyze the following debate topic and identify {max_topics} key aspects or sub-topics that should be debated separately.

Topic: {topic}

{context_section}

For each sub-topic, provide:
1. A clear, focused aspect to debate
2. A weight (0.0-1.0) indicating its importance to the main topic
3. A brief description of what this sub-debate should address

Output as a numbered list in this format:
1. [Aspect]: [Weight] - [Description]

Ensure:
- Aspects are distinct and don't overlap significantly
- Weights sum to approximately 1.0
- Each aspect is debatable and relevant to the main topic
"""

    def __init__(
        self,
        model: "BaseModel | None" = None,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        """Initialize the LLM decomposer.

        Args:
            model: Pre-configured model instance.
            model_name: Model name if creating new instance.
            api_key: API key for model.
        """
        self._model = model
        self._model_name = model_name
        self._api_key = api_key

    @property
    def strategy(self) -> DecompositionStrategy:
        return DecompositionStrategy.LLM

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

    async def decompose(
        self,
        topic: str,
        context: HierarchicalContext | None = None,
        max_subtopics: int = 5,
    ) -> list[SubDebateSpec]:
        """Decompose using LLM analysis."""
        model = await self._get_model()

        # Build context section
        context_section = ""
        if context:
            parts = []
            if context.parent_topic:
                parts.append(f"Parent topic: {context.parent_topic}")
            if context.sibling_topics:
                parts.append(f"Already covered: {', '.join(context.sibling_topics)}")
            if context.depth > 0:
                parts.append(f"Depth: {context.depth}/{context.max_depth}")
            if parts:
                context_section = "Context:\n" + "\n".join(parts)

        prompt = self.DECOMPOSITION_PROMPT.format(
            topic=topic,
            max_topics=max_subtopics,
            context_section=context_section,
        )

        from artemis.core.types import Message

        response = await model.generate([Message(role="user", content=prompt)])

        # Parse response
        specs = self._parse_response(response.content, topic)

        return specs[:max_subtopics]

    def _parse_response(self, response: str, topic: str) -> list[SubDebateSpec]:
        """Parse LLM response into SubDebateSpecs."""
        import re

        specs = []
        lines = response.strip().split("\n")

        for line in lines:
            # Look for numbered items
            match = re.match(r"^\d+\.\s*(.+?):\s*([\d.]+)\s*-\s*(.+)$", line.strip())
            if match:
                aspect = match.group(1).strip()
                try:
                    weight = float(match.group(2))
                except ValueError:
                    weight = 0.2
                description = match.group(3).strip()

                specs.append(
                    SubDebateSpec(
                        aspect=f"{aspect}: {topic}",
                        weight=min(1.0, max(0.0, weight)),
                        description=description,
                    )
                )

        # If parsing failed, create default specs synchronously
        if not specs:
            logger.warning("Failed to parse LLM decomposition, using defaults")
            # Return simple default specs (can't call async decomposer here)
            return [
                SubDebateSpec(
                    aspect=f"Key considerations of: {topic}",
                    weight=0.5,
                    description="Main aspects to consider",
                ),
                SubDebateSpec(
                    aspect=f"Implications of: {topic}",
                    weight=0.5,
                    description="Potential consequences and effects",
                ),
            ]

        # Normalize weights
        total_weight = sum(s.weight for s in specs)
        if total_weight > 0:
            for spec in specs:
                spec.weight = spec.weight / total_weight

        return specs


class HybridDecomposer(TopicDecomposer):
    """Decomposer that combines LLM and rule-based approaches.

    Uses rules for known patterns and LLM for novel topics.

    Example:
        ```python
        decomposer = HybridDecomposer(model=my_model)
        specs = await decomposer.decompose("A complex novel topic")
        ```
    """

    def __init__(
        self,
        model: "BaseModel | None" = None,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        """Initialize the hybrid decomposer."""
        self._rule_based = RuleBasedDecomposer()
        self._llm_based = LLMTopicDecomposer(
            model=model,
            model_name=model_name,
            api_key=api_key,
        )

    @property
    def strategy(self) -> DecompositionStrategy:
        return DecompositionStrategy.HYBRID

    async def decompose(
        self,
        topic: str,
        context: HierarchicalContext | None = None,
        max_subtopics: int = 5,
    ) -> list[SubDebateSpec]:
        """Decompose using hybrid approach."""
        # Get rule-based suggestions
        rule_specs = await self._rule_based.decompose(topic, context, max_subtopics)

        # Use LLM to refine or expand
        try:
            llm_specs = await self._llm_based.decompose(topic, context, max_subtopics)

            # Merge: prefer LLM but fall back to rules
            if len(llm_specs) >= 2:
                return llm_specs
            else:
                return rule_specs
        except Exception as e:
            logger.warning(f"LLM decomposition failed: {e}, using rule-based")
            return rule_specs
