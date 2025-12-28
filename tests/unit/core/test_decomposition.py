"""Tests for topic decomposition strategies."""

import pytest

from artemis.core.decomposition import (
    HybridDecomposer,
    LLMTopicDecomposer,
    ManualDecomposer,
    RuleBasedDecomposer,
    TopicDecomposer,
)
from artemis.core.types import (
    DecompositionStrategy,
    HierarchicalContext,
    SubDebateSpec,
)


class TestManualDecomposer:
    """Tests for ManualDecomposer."""

    @pytest.fixture
    def specs(self) -> list[SubDebateSpec]:
        """Create test specs."""
        return [
            SubDebateSpec(aspect="Economic impact", weight=0.4),
            SubDebateSpec(aspect="Social implications", weight=0.3),
            SubDebateSpec(aspect="Technical feasibility", weight=0.3),
        ]

    def test_strategy_property(self, specs: list[SubDebateSpec]):
        """Should return MANUAL strategy."""
        decomposer = ManualDecomposer(specs)
        assert decomposer.strategy == DecompositionStrategy.MANUAL

    @pytest.mark.asyncio
    async def test_decompose_returns_specs(self, specs: list[SubDebateSpec]):
        """Should return the predefined specs."""
        decomposer = ManualDecomposer(specs)
        result = await decomposer.decompose("Any topic")
        assert result == specs

    @pytest.mark.asyncio
    async def test_decompose_respects_max_subtopics(self, specs: list[SubDebateSpec]):
        """Should limit to max_subtopics."""
        decomposer = ManualDecomposer(specs)
        result = await decomposer.decompose("Any topic", max_subtopics=2)
        assert len(result) == 2
        assert result == specs[:2]

    @pytest.mark.asyncio
    async def test_decompose_ignores_context(self, specs: list[SubDebateSpec]):
        """Context should not affect manual decomposition."""
        decomposer = ManualDecomposer(specs)
        context = HierarchicalContext(
            parent_topic="Parent",
            depth=1,
            max_depth=3,
            path=["Parent"],
        )
        result = await decomposer.decompose("Topic", context=context)
        assert result == specs


class TestRuleBasedDecomposer:
    """Tests for RuleBasedDecomposer."""

    def test_strategy_property(self):
        """Should return RULE_BASED strategy."""
        decomposer = RuleBasedDecomposer()
        assert decomposer.strategy == DecompositionStrategy.RULE_BASED

    @pytest.mark.asyncio
    async def test_decompose_default_dimensions(self):
        """Should use default dimensions for generic topics."""
        decomposer = RuleBasedDecomposer()
        result = await decomposer.decompose("Should we do this thing?")

        assert len(result) == 4
        # Default dimensions
        aspects = [s.aspect for s in result]
        assert any("Economic" in a for a in aspects)
        assert any("Ethical" in a for a in aspects)

    @pytest.mark.asyncio
    async def test_decompose_technology_topic(self):
        """Should use technology dimensions for tech topics."""
        decomposer = RuleBasedDecomposer()
        result = await decomposer.decompose("Should AI be regulated?")

        aspects = [s.aspect for s in result]
        assert any("Technical" in a or "Security" in a for a in aspects)

    @pytest.mark.asyncio
    async def test_decompose_policy_topic(self):
        """Should use policy dimensions for policy topics."""
        decomposer = RuleBasedDecomposer()
        result = await decomposer.decompose("Should we change the law?")

        aspects = [s.aspect for s in result]
        assert any("Legal" in a or "Political" in a for a in aspects)

    @pytest.mark.asyncio
    async def test_decompose_environment_topic(self):
        """Should use environment dimensions for climate topics."""
        decomposer = RuleBasedDecomposer()
        result = await decomposer.decompose("Is climate change action needed?")

        aspects = [s.aspect for s in result]
        assert any("Environmental" in a for a in aspects)

    @pytest.mark.asyncio
    async def test_decompose_respects_max_subtopics(self):
        """Should limit to max_subtopics."""
        decomposer = RuleBasedDecomposer()
        result = await decomposer.decompose("Generic topic", max_subtopics=2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_decompose_normalizes_weights(self):
        """Weights should sum to 1.0."""
        decomposer = RuleBasedDecomposer()
        result = await decomposer.decompose("Some topic")

        total_weight = sum(s.weight for s in result)
        assert abs(total_weight - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_custom_dimensions(self):
        """Should use custom dimensions when provided."""
        custom = [
            ("Custom dimension 1", 0.5, "Description 1"),
            ("Custom dimension 2", 0.5, "Description 2"),
        ]
        decomposer = RuleBasedDecomposer(custom_dimensions=custom)
        result = await decomposer.decompose("Any topic")

        assert len(result) == 2
        assert "Custom dimension 1" in result[0].aspect
        assert "Custom dimension 2" in result[1].aspect


class TestLLMTopicDecomposer:
    """Tests for LLMTopicDecomposer."""

    def test_strategy_property(self):
        """Should return LLM strategy."""
        decomposer = LLMTopicDecomposer()
        assert decomposer.strategy == DecompositionStrategy.LLM

    def test_prompt_template(self):
        """Should have decomposition prompt template."""
        assert "{topic}" in LLMTopicDecomposer.DECOMPOSITION_PROMPT
        assert "{max_topics}" in LLMTopicDecomposer.DECOMPOSITION_PROMPT

    def test_parse_response_valid(self):
        """Should parse valid LLM response."""
        decomposer = LLMTopicDecomposer()
        response = """
1. Economic considerations: 0.3 - Analyze the economic effects
2. Social impact: 0.35 - Consider societal implications
3. Technical aspects: 0.35 - Evaluate technical requirements
"""
        topic = "Test topic"
        specs = decomposer._parse_response(response, topic)

        assert len(specs) == 3
        assert "Economic considerations" in specs[0].aspect
        assert abs(specs[0].weight + specs[1].weight + specs[2].weight - 1.0) < 0.01

    def test_parse_response_invalid_format(self):
        """Should handle invalid response format gracefully."""
        decomposer = LLMTopicDecomposer()
        response = "This is not a valid decomposition format"
        # Should fall back to rule-based or return empty
        # The current implementation returns empty and falls back
        specs = decomposer._parse_response(response, "Test topic")
        # May return empty or fall back to rule-based
        assert isinstance(specs, list)

    def test_parse_response_normalizes_weights(self):
        """Should normalize weights in parsed response."""
        decomposer = LLMTopicDecomposer()
        response = """
1. Aspect A: 0.5 - Description A
2. Aspect B: 0.5 - Description B
"""
        specs = decomposer._parse_response(response, "Topic")
        total = sum(s.weight for s in specs)
        assert abs(total - 1.0) < 0.01


class TestHybridDecomposer:
    """Tests for HybridDecomposer."""

    def test_strategy_property(self):
        """Should return HYBRID strategy."""
        decomposer = HybridDecomposer()
        assert decomposer.strategy == DecompositionStrategy.HYBRID

    def test_has_both_decomposers(self):
        """Should have both rule-based and LLM decomposers."""
        decomposer = HybridDecomposer()
        assert hasattr(decomposer, "_rule_based")
        assert hasattr(decomposer, "_llm_based")
        assert isinstance(decomposer._rule_based, RuleBasedDecomposer)
        assert isinstance(decomposer._llm_based, LLMTopicDecomposer)


class TestTopicDecomposerInterface:
    """Tests for TopicDecomposer ABC."""

    def test_cannot_instantiate_abc(self):
        """Should not be able to instantiate abstract class."""
        with pytest.raises(TypeError):
            TopicDecomposer()

    def test_subclass_requires_methods(self):
        """Subclass must implement required methods."""
        class IncompleteDecomposer(TopicDecomposer):
            pass

        with pytest.raises(TypeError):
            IncompleteDecomposer()
