"""
Unit tests for Causal Reasoning module.
"""

import pytest

from artemis.core.causal import (
    CausalEdge,
    CausalExtractor,
    CausalGraph,
    CausalNode,
    LinkType,
)
from artemis.core.types import CausalLink


class TestCausalNode:
    """Tests for CausalNode dataclass."""

    def test_node_creation(self) -> None:
        """Test creating a causal node."""
        node = CausalNode(id="test_node", label="Test Node")
        assert node.id == "test_node"
        assert node.label == "Test Node"
        assert node.argument_ids == []

    def test_node_with_arguments(self) -> None:
        """Test node with argument IDs."""
        node = CausalNode(
            id="node1",
            label="Node One",
            argument_ids=["arg-1", "arg-2"],
        )
        assert len(node.argument_ids) == 2

    def test_node_equality(self) -> None:
        """Test node equality based on ID."""
        node1 = CausalNode(id="same_id", label="Label 1")
        node2 = CausalNode(id="same_id", label="Label 2")
        node3 = CausalNode(id="different_id", label="Label 1")

        assert node1 == node2
        assert node1 != node3

    def test_node_hash(self) -> None:
        """Test node hashing."""
        node1 = CausalNode(id="test", label="Test")
        node2 = CausalNode(id="test", label="Test")

        assert hash(node1) == hash(node2)


class TestCausalEdge:
    """Tests for CausalEdge dataclass."""

    def test_edge_creation(self) -> None:
        """Test creating a causal edge."""
        edge = CausalEdge(
            source_id="cause",
            target_id="effect",
            link_type=LinkType.CAUSES,
            strength=0.8,
        )
        assert edge.source_id == "cause"
        assert edge.target_id == "effect"
        assert edge.link_type == LinkType.CAUSES
        assert edge.strength == 0.8
        assert edge.evidence_count == 1

    def test_edge_hash(self) -> None:
        """Test edge hashing."""
        edge1 = CausalEdge(
            source_id="a", target_id="b", link_type=LinkType.CAUSES, strength=0.5
        )
        edge2 = CausalEdge(
            source_id="a", target_id="b", link_type=LinkType.CAUSES, strength=0.8
        )
        assert hash(edge1) == hash(edge2)


class TestCausalGraph:
    """Tests for CausalGraph class."""

    @pytest.fixture
    def graph(self) -> CausalGraph:
        return CausalGraph()

    def test_add_node(self, graph: CausalGraph) -> None:
        """Test adding a node to the graph."""
        node_id = graph.add_node("AI Technology")
        assert node_id == "ai_technology"
        assert len(graph) == 1

    def test_add_node_with_argument(self, graph: CausalGraph) -> None:
        """Test adding a node with argument ID."""
        graph.add_node("AI Technology", argument_id="arg-1")
        node = graph.get_node("AI Technology")
        assert node is not None
        assert "arg-1" in node.argument_ids

    def test_add_duplicate_node(self, graph: CausalGraph) -> None:
        """Test adding duplicate node updates argument IDs."""
        graph.add_node("Test", argument_id="arg-1")
        graph.add_node("Test", argument_id="arg-2")

        node = graph.get_node("Test")
        assert node is not None
        assert len(node.argument_ids) == 2

    def test_add_link(self, graph: CausalGraph) -> None:
        """Test adding a causal link."""
        link = CausalLink(cause="AI", effect="Automation", strength=0.7)
        graph.add_link(link)

        assert len(graph) == 2  # Two nodes created
        edge = graph.get_edge("AI", "Automation")
        assert edge is not None
        assert edge.strength == 0.7

    def test_add_link_with_type(self, graph: CausalGraph) -> None:
        """Test adding a link with specific type."""
        link = CausalLink(cause="Barrier", effect="Progress", strength=0.8)
        graph.add_link(link, link_type=LinkType.PREVENTS)

        edge = graph.get_edge("Barrier", "Progress")
        assert edge is not None
        assert edge.link_type == LinkType.PREVENTS

    def test_add_duplicate_link(self, graph: CausalGraph) -> None:
        """Test that duplicate links update strength."""
        link1 = CausalLink(cause="A", effect="B", strength=0.6)
        link2 = CausalLink(cause="A", effect="B", strength=0.8)

        graph.add_link(link1)
        graph.add_link(link2)

        edge = graph.get_edge("A", "B")
        assert edge is not None
        assert edge.evidence_count == 2
        # Strength should be average: (0.6 * 1 + 0.8) / 2 = 0.7
        assert abs(edge.strength - 0.7) < 0.01

    def test_get_effects(self, graph: CausalGraph) -> None:
        """Test getting direct effects of a cause."""
        graph.add_link(CausalLink(cause="AI", effect="Automation", strength=0.7))
        graph.add_link(CausalLink(cause="AI", effect="Job Loss", strength=0.5))
        graph.add_link(CausalLink(cause="Automation", effect="Efficiency", strength=0.8))

        effects = graph.get_effects("AI")
        effect_labels = [e.label for e in effects]
        assert len(effects) == 2
        assert "Automation" in effect_labels
        assert "Job Loss" in effect_labels

    def test_get_causes(self, graph: CausalGraph) -> None:
        """Test getting direct causes of an effect."""
        graph.add_link(CausalLink(cause="AI", effect="Productivity", strength=0.7))
        graph.add_link(CausalLink(cause="Automation", effect="Productivity", strength=0.8))

        causes = graph.get_causes("Productivity")
        cause_labels = [c.label for c in causes]
        assert len(causes) == 2
        assert "AI" in cause_labels
        assert "Automation" in cause_labels

    def test_find_paths(self, graph: CausalGraph) -> None:
        """Test finding paths between nodes."""
        graph.add_link(CausalLink(cause="AI", effect="Automation", strength=0.8))
        graph.add_link(CausalLink(cause="Automation", effect="Efficiency", strength=0.9))
        graph.add_link(CausalLink(cause="Efficiency", effect="Profit", strength=0.7))

        paths = graph.find_paths("AI", "Profit")
        assert len(paths) == 1
        assert len(paths[0]) == 4  # AI -> Automation -> Efficiency -> Profit

    def test_find_multiple_paths(self, graph: CausalGraph) -> None:
        """Test finding multiple paths."""
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.8))
        graph.add_link(CausalLink(cause="A", effect="C", strength=0.7))
        graph.add_link(CausalLink(cause="B", effect="D", strength=0.9))
        graph.add_link(CausalLink(cause="C", effect="D", strength=0.6))

        paths = graph.find_paths("A", "D")
        assert len(paths) == 2

    def test_find_no_path(self, graph: CausalGraph) -> None:
        """Test when no path exists."""
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.8))
        graph.add_link(CausalLink(cause="C", effect="D", strength=0.8))

        paths = graph.find_paths("A", "D")
        assert len(paths) == 0

    def test_compute_path_strength(self, graph: CausalGraph) -> None:
        """Test computing path strength."""
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.8))
        graph.add_link(CausalLink(cause="B", effect="C", strength=0.5))

        paths = graph.find_paths("A", "C")
        assert len(paths) == 1

        strength = graph.compute_path_strength(paths[0])
        # 0.8 * 0.5 = 0.4
        assert abs(strength - 0.4) < 0.01

    def test_get_strongest_path(self, graph: CausalGraph) -> None:
        """Test getting the strongest path."""
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.9))
        graph.add_link(CausalLink(cause="A", effect="C", strength=0.6))
        graph.add_link(CausalLink(cause="B", effect="D", strength=0.8))
        graph.add_link(CausalLink(cause="C", effect="D", strength=0.7))

        path, strength = graph.get_strongest_path("A", "D")
        # A->B->D: 0.9 * 0.8 = 0.72
        # A->C->D: 0.6 * 0.7 = 0.42
        assert abs(strength - 0.72) < 0.01

    def test_get_root_causes(self, graph: CausalGraph) -> None:
        """Test getting root causes."""
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.8))
        graph.add_link(CausalLink(cause="B", effect="C", strength=0.7))
        graph.add_link(CausalLink(cause="D", effect="C", strength=0.6))

        roots = graph.get_root_causes()
        root_ids = [r.id for r in roots]
        assert "a" in root_ids
        assert "d" in root_ids
        assert "b" not in root_ids
        assert "c" not in root_ids

    def test_get_terminal_effects(self, graph: CausalGraph) -> None:
        """Test getting terminal effects."""
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.8))
        graph.add_link(CausalLink(cause="B", effect="C", strength=0.7))

        terminals = graph.get_terminal_effects()
        terminal_ids = [t.id for t in terminals]
        assert "c" in terminal_ids
        assert "a" not in terminal_ids
        assert "b" not in terminal_ids

    def test_has_no_cycle(self, graph: CausalGraph) -> None:
        """Test cycle detection when no cycle exists."""
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.8))
        graph.add_link(CausalLink(cause="B", effect="C", strength=0.7))
        graph.add_link(CausalLink(cause="A", effect="C", strength=0.6))

        assert not graph.has_cycle()

    def test_has_cycle(self, graph: CausalGraph) -> None:
        """Test cycle detection when cycle exists."""
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.8))
        graph.add_link(CausalLink(cause="B", effect="C", strength=0.7))
        graph.add_link(CausalLink(cause="C", effect="A", strength=0.6))

        assert graph.has_cycle()

    def test_merge_nodes(self, graph: CausalGraph) -> None:
        """Test merging multiple nodes."""
        graph.add_link(CausalLink(cause="AI", effect="Result", strength=0.8))
        graph.add_link(CausalLink(cause="Artificial Intelligence", effect="Outcome", strength=0.7))

        graph.merge_nodes(["AI", "Artificial Intelligence"], "AI Tech")

        # Should have merged node
        merged = graph.get_node("AI Tech")
        assert merged is not None

    def test_graph_iteration(self, graph: CausalGraph) -> None:
        """Test iterating over graph nodes."""
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.8))
        graph.add_link(CausalLink(cause="B", effect="C", strength=0.7))

        nodes = list(graph)
        assert len(nodes) == 3


class TestCausalExtractor:
    """Tests for CausalExtractor class."""

    @pytest.fixture
    def extractor(self) -> CausalExtractor:
        return CausalExtractor()

    def test_extract_causes_pattern(self, extractor: CausalExtractor) -> None:
        """Test extracting 'causes' pattern."""
        text = "Climate change causes rising sea levels."
        results = extractor.extract(text)

        assert len(results) >= 1
        link, link_type = results[0]
        assert link_type == LinkType.CAUSES

    def test_extract_leads_to_pattern(self, extractor: CausalExtractor) -> None:
        """Test extracting 'leads to' pattern."""
        text = "Economic instability leads to social unrest."
        results = extractor.extract(text)

        assert len(results) >= 1
        assert any(lt == LinkType.CAUSES for _, lt in results)

    def test_extract_therefore_pattern(self, extractor: CausalExtractor) -> None:
        """Test extracting 'therefore' pattern."""
        text = "All humans are mortal. Socrates is human, therefore Socrates is mortal."
        results = extractor.extract(text)

        assert len(results) >= 1

    def test_extract_prevents_pattern(self, extractor: CausalExtractor) -> None:
        """Test extracting 'prevents' pattern."""
        text = "Vaccination prevents the spread of disease."
        results = extractor.extract(text)

        assert len(results) >= 1
        assert any(lt == LinkType.PREVENTS for _, lt in results)

    def test_extract_enables_pattern(self, extractor: CausalExtractor) -> None:
        """Test extracting 'enables' pattern."""
        text = "Technology enables faster communication."
        results = extractor.extract(text)

        assert len(results) >= 1
        assert any(lt == LinkType.ENABLES for _, lt in results)

    def test_extract_if_then_pattern(self, extractor: CausalExtractor) -> None:
        """Test extracting 'if...then' pattern."""
        text = "If we increase taxes, then public services will improve."
        results = extractor.extract(text)

        assert len(results) >= 1
        assert any(lt == LinkType.IMPLIES for _, lt in results)

    def test_extract_correlates_pattern(self, extractor: CausalExtractor) -> None:
        """Test extracting correlation pattern."""
        text = "Income level is correlated with education."
        results = extractor.extract(text)

        assert len(results) >= 1
        assert any(lt == LinkType.CORRELATES for _, lt in results)

    def test_extract_multiple_links(self, extractor: CausalExtractor) -> None:
        """Test extracting multiple causal links."""
        text = (
            "Climate change causes rising temperatures. "
            "Higher temperatures lead to ice melt. "
            "Ice melt results in sea level rise."
        )
        results = extractor.extract(text)

        assert len(results) >= 2

    def test_strength_estimation(self, extractor: CausalExtractor) -> None:
        """Test that strength is estimated based on certainty markers."""
        text_strong = "AI definitely causes job displacement."
        text_weak = "AI might cause job displacement."

        results_strong = extractor.extract(text_strong)
        results_weak = extractor.extract(text_weak)

        if results_strong and results_weak:
            # Strong certainty should have higher strength
            assert results_strong[0][0].strength >= results_weak[0][0].strength

    def test_build_graph(self, extractor: CausalExtractor) -> None:
        """Test building a graph from text."""
        text = "AI causes automation. Automation leads to efficiency."
        graph = extractor.build_graph(text)

        # Should find at least one causal relationship
        assert len(graph) >= 2
        assert len(graph.edges) >= 1

    def test_empty_text(self, extractor: CausalExtractor) -> None:
        """Test extraction from empty text."""
        results = extractor.extract("")
        assert results == []

    def test_no_causal_text(self, extractor: CausalExtractor) -> None:
        """Test extraction from text without causal relationships."""
        text = "The sky is blue. Water is wet."
        results = extractor.extract(text)
        # Should return empty or minimal results
        assert isinstance(results, list)


class TestLinkType:
    """Tests for LinkType enum."""

    def test_all_types_defined(self) -> None:
        """Test that all expected link types are defined."""
        expected_types = ["causes", "enables", "prevents", "correlates", "implies"]
        for type_name in expected_types:
            assert hasattr(LinkType, type_name.upper())

    def test_type_values(self) -> None:
        """Test that type values match expected strings."""
        assert LinkType.CAUSES.value == "causes"
        assert LinkType.ENABLES.value == "enables"
        assert LinkType.PREVENTS.value == "prevents"
        assert LinkType.CORRELATES.value == "correlates"
        assert LinkType.IMPLIES.value == "implies"
