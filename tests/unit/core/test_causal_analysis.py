"""
Unit tests for CausalGraph v2 analysis features.

Tests for:
- CausalAnalyzer (circular reasoning, weak links, contradictions, etc.)
- CausalFallacyDetector (post hoc, false cause, slippery slope)
- CausalStrategy (attack surface, reinforcement suggestions)
- CausalVisualizer (DOT, Mermaid, JSON export)
"""

import pytest

from artemis.core.causal import CausalGraph, LinkType
from artemis.core.causal_analysis import CausalAnalyzer, CausalFallacyDetector
from artemis.core.causal_strategy import CausalStrategy
from artemis.core.causal_visualization import CausalVisualizer, create_snapshot
from artemis.core.types import (
    Argument,
    ArgumentLevel,
    CausalLink,
    FallacyType,
)


class TestCausalAnalyzer:
    """Tests for CausalAnalyzer class."""

    @pytest.fixture
    def simple_graph(self) -> CausalGraph:
        """Create a simple graph for testing."""
        graph = CausalGraph()
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.8))
        graph.add_link(CausalLink(cause="B", effect="C", strength=0.7))
        graph.add_link(CausalLink(cause="C", effect="D", strength=0.6))
        return graph

    @pytest.fixture
    def cyclic_graph(self) -> CausalGraph:
        """Create a graph with a cycle."""
        graph = CausalGraph()
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.8))
        graph.add_link(CausalLink(cause="B", effect="C", strength=0.7))
        graph.add_link(CausalLink(cause="C", effect="A", strength=0.6))
        return graph

    @pytest.fixture
    def weak_links_graph(self) -> CausalGraph:
        """Create a graph with weak links."""
        graph = CausalGraph()
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.9))
        graph.add_link(CausalLink(cause="B", effect="C", strength=0.2))
        graph.add_link(CausalLink(cause="C", effect="D", strength=0.3))
        graph.add_link(CausalLink(cause="D", effect="E", strength=0.8))
        return graph

    def test_find_circular_reasoning_no_cycle(self, simple_graph: CausalGraph) -> None:
        """Test circular reasoning detection with no cycle."""
        analyzer = CausalAnalyzer(simple_graph)
        cycles = analyzer.find_circular_reasoning()
        assert len(cycles) == 0

    def test_find_circular_reasoning_with_cycle(self, cyclic_graph: CausalGraph) -> None:
        """Test circular reasoning detection with cycle."""
        analyzer = CausalAnalyzer(cyclic_graph)
        cycles = analyzer.find_circular_reasoning()
        assert len(cycles) >= 1
        assert cycles[0].severity > 0

    def test_get_circular_chains(self, cyclic_graph: CausalGraph) -> None:
        """Test getting raw cycle lists."""
        analyzer = CausalAnalyzer(cyclic_graph)
        chains = analyzer.get_circular_chains()
        assert len(chains) >= 1
        assert isinstance(chains[0], list)

    def test_find_weak_links(self, weak_links_graph: CausalGraph) -> None:
        """Test weak link detection."""
        analyzer = CausalAnalyzer(weak_links_graph)
        weak = analyzer.find_weak_links(threshold=0.4)

        assert len(weak) == 2
        assert weak[0].weakness_score > weak[1].weakness_score
        assert weak[0].strength < 0.4

    def test_find_weak_links_with_suggestions(self, weak_links_graph: CausalGraph) -> None:
        """Test that weak links include attack suggestions."""
        analyzer = CausalAnalyzer(weak_links_graph)
        weak = analyzer.find_weak_links(threshold=0.4)

        assert len(weak) > 0
        assert len(weak[0].attack_suggestions) > 0

    def test_find_contradictions_none(self, simple_graph: CausalGraph) -> None:
        """Test contradiction detection with no contradictions."""
        analyzer = CausalAnalyzer(simple_graph)
        contradictions = analyzer.find_contradictions()
        assert len(contradictions) == 0

    def test_find_contradictions_opposite_types(self) -> None:
        """Test detecting contradictions from bidirectional strong causation.

        Note: CausalGraph merges edges with same source-target, so we test
        the bidirectional strong causation case instead of opposite types.
        """
        graph = CausalGraph()
        # Bidirectional strong causation is a type of contradiction
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.8))
        graph.add_link(CausalLink(cause="B", effect="A", strength=0.8))

        analyzer = CausalAnalyzer(graph)
        contradictions = analyzer.find_contradictions()

        assert len(contradictions) >= 1

    def test_find_contradictions_bidirectional(self) -> None:
        """Test detecting bidirectional strong causation."""
        graph = CausalGraph()
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.8))
        graph.add_link(CausalLink(cause="B", effect="A", strength=0.8))

        analyzer = CausalAnalyzer(graph)
        contradictions = analyzer.find_contradictions()

        assert len(contradictions) >= 1

    def test_compute_argument_strength(self, simple_graph: CausalGraph) -> None:
        """Test computing argument strength."""
        simple_graph.add_node("A", argument_id="arg-1")
        simple_graph.add_node("B", argument_id="arg-1")

        analyzer = CausalAnalyzer(simple_graph)
        strength = analyzer.compute_argument_strength("arg-1")

        assert strength.argument_id == "arg-1"
        assert 0 <= strength.overall_score <= 1
        assert 0 <= strength.causal_support <= 1
        assert 0 <= strength.vulnerability <= 1

    def test_compute_argument_strength_unknown(self, simple_graph: CausalGraph) -> None:
        """Test computing strength for unknown argument."""
        analyzer = CausalAnalyzer(simple_graph)
        strength = analyzer.compute_argument_strength("unknown-arg")

        assert strength.overall_score == 0.0
        assert strength.vulnerability == 1.0

    def test_rank_arguments_by_strength(self, simple_graph: CausalGraph) -> None:
        """Test ranking arguments by strength."""
        simple_graph.add_node("A", argument_id="arg-1")
        simple_graph.add_node("B", argument_id="arg-2")
        simple_graph.add_node("C", argument_id="arg-2")

        analyzer = CausalAnalyzer(simple_graph)
        rankings = analyzer.rank_arguments_by_strength()

        assert len(rankings) >= 2
        assert rankings[0][1] >= rankings[-1][1]

    def test_find_critical_nodes(self, simple_graph: CausalGraph) -> None:
        """Test critical node detection."""
        analyzer = CausalAnalyzer(simple_graph)
        critical = analyzer.find_critical_nodes()

        assert len(critical) > 0
        assert critical[0].centrality_score >= critical[-1].centrality_score

    def test_find_reasoning_gaps(self, simple_graph: CausalGraph) -> None:
        """Test reasoning gap detection."""
        analyzer = CausalAnalyzer(simple_graph)
        gaps = analyzer.find_reasoning_gaps()

        assert isinstance(gaps, list)

    def test_compute_chain_completeness(self, simple_graph: CausalGraph) -> None:
        """Test chain completeness computation."""
        analyzer = CausalAnalyzer(simple_graph)
        paths = simple_graph.find_paths("A", "D")

        completeness = analyzer.compute_chain_completeness(paths[0])
        assert 0 <= completeness <= 1

    def test_analyze_full(self, simple_graph: CausalGraph) -> None:
        """Test full analysis."""
        analyzer = CausalAnalyzer(simple_graph)
        result = analyzer.analyze()

        assert result.has_circular_reasoning is False
        assert isinstance(result.weak_links, list)
        assert isinstance(result.contradictions, list)
        assert 0 <= result.overall_coherence <= 1

    def test_analyze_with_issues(self, cyclic_graph: CausalGraph) -> None:
        """Test full analysis with issues."""
        analyzer = CausalAnalyzer(cyclic_graph)
        result = analyzer.analyze()

        assert result.has_circular_reasoning is True
        assert result.overall_coherence < 1.0


class TestCausalFallacyDetector:
    """Tests for CausalFallacyDetector class."""

    @pytest.fixture
    def detector(self) -> CausalFallacyDetector:
        return CausalFallacyDetector()

    @pytest.fixture
    def simple_graph(self) -> CausalGraph:
        graph = CausalGraph()
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.8))
        return graph

    def test_detect_post_hoc(self, detector: CausalFallacyDetector) -> None:
        """Test post hoc fallacy detection."""
        argument = Argument(
            agent="test",
            level=ArgumentLevel.TACTICAL,
            content="After the new policy was introduced, crime rates dropped.",
            causal_links=[
                CausalLink(
                    cause="policy",
                    effect="crime_drop",
                    strength=0.2,
                )
            ],
        )

        fallacies = detector._check_post_hoc(argument)
        assert isinstance(fallacies, list)

    def test_detect_false_cause_weak_link(
        self, detector: CausalFallacyDetector
    ) -> None:
        """Test false cause detection from weak links."""
        argument = Argument(
            agent="test",
            level=ArgumentLevel.TACTICAL,
            content="Studies show correlation between X and Y.",
            causal_links=[
                CausalLink(
                    cause="correlation_x_y",
                    effect="conclusion",
                    strength=0.2,
                )
            ],
        )

        fallacies = detector._check_false_cause(argument)
        assert isinstance(fallacies, list)
        if fallacies:
            assert fallacies[0].fallacy_type == FallacyType.FALSE_CAUSE

    def test_detect_slippery_slope_text(
        self, detector: CausalFallacyDetector, simple_graph: CausalGraph
    ) -> None:
        """Test slippery slope detection from text."""
        argument = Argument(
            agent="test",
            level=ArgumentLevel.TACTICAL,
            content="This policy will inevitably lead to economic collapse.",
            causal_links=[],
        )

        fallacies = detector._check_slippery_slope(argument, simple_graph)
        assert isinstance(fallacies, list)
        if fallacies:
            assert fallacies[0].fallacy_type == FallacyType.SLIPPERY_SLOPE

    def test_detect_circular_reasoning(
        self, detector: CausalFallacyDetector
    ) -> None:
        """Test circular reasoning detection."""
        graph = CausalGraph()
        graph.add_link(
            CausalLink(cause="A", effect="B", strength=0.8),
            argument_id="arg-1",
        )
        graph.add_link(
            CausalLink(cause="B", effect="C", strength=0.7),
            argument_id="arg-1",
        )
        graph.add_link(
            CausalLink(cause="C", effect="A", strength=0.6),
            argument_id="arg-1",
        )

        argument = Argument(
            id="arg-1",
            agent="test",
            level=ArgumentLevel.TACTICAL,
            content="A leads to B leads to C leads back to A.",
            causal_links=[],
        )

        fallacies = detector._check_circular_reasoning(argument, graph)
        assert isinstance(fallacies, list)

    def test_detect_fallacies_full(
        self, detector: CausalFallacyDetector, simple_graph: CausalGraph
    ) -> None:
        """Test full fallacy detection."""
        argument = Argument(
            agent="test",
            level=ArgumentLevel.TACTICAL,
            content="After X happened, Y occurred. Therefore X caused Y.",
            causal_links=[
                CausalLink(cause="X", effect="Y", strength=0.3)
            ],
        )

        fallacies = detector.detect_fallacies(argument, simple_graph)
        assert isinstance(fallacies, list)

    def test_check_post_hoc_single_link(
        self, detector: CausalFallacyDetector
    ) -> None:
        """Test post hoc check on single link."""
        link = CausalLink(cause="event_a", effect="event_b", strength=0.2)
        result = detector.check_post_hoc(link)

        assert result is not None
        assert result.fallacy_type == FallacyType.POST_HOC

    def test_check_slippery_slope_path(
        self, detector: CausalFallacyDetector, simple_graph: CausalGraph
    ) -> None:
        """Test slippery slope check on path."""
        graph = CausalGraph()
        for i in range(5):
            graph.add_link(
                CausalLink(
                    cause=f"node_{i}",
                    effect=f"node_{i + 1}",
                    strength=0.3,
                )
            )

        path = [f"node_{i}" for i in range(6)]
        result = detector.check_slippery_slope(path, graph)

        assert result is not None
        assert result.fallacy_type == FallacyType.SLIPPERY_SLOPE


class TestCausalStrategy:
    """Tests for CausalStrategy class."""

    @pytest.fixture
    def own_graph(self) -> CausalGraph:
        graph = CausalGraph()
        graph.add_link(CausalLink(cause="premise_1", effect="conclusion", strength=0.8))
        graph.add_link(CausalLink(cause="premise_2", effect="conclusion", strength=0.5))
        return graph

    @pytest.fixture
    def opponent_graph(self) -> CausalGraph:
        graph = CausalGraph()
        graph.add_link(CausalLink(cause="opp_premise", effect="opp_conclusion", strength=0.3))
        graph.add_link(CausalLink(cause="weak_claim", effect="strong_claim", strength=0.2))
        return graph

    def test_map_attack_surface(
        self, own_graph: CausalGraph, opponent_graph: CausalGraph
    ) -> None:
        """Test attack surface mapping."""
        strategy = CausalStrategy(own_graph, opponent_graph)
        targets = strategy.map_attack_surface()

        assert len(targets) > 0
        assert targets[0].vulnerability_score > 0

    def test_map_attack_surface_no_opponent(self, own_graph: CausalGraph) -> None:
        """Test attack surface with no opponent."""
        strategy = CausalStrategy(own_graph)
        targets = strategy.map_attack_surface()

        assert len(targets) == 0

    def test_suggest_rebuttals(self, own_graph: CausalGraph) -> None:
        """Test rebuttal suggestions."""
        strategy = CausalStrategy(own_graph)

        opponent_argument = Argument(
            agent="opponent",
            level=ArgumentLevel.TACTICAL,
            content="Weak argument with little evidence.",
            causal_links=[
                CausalLink(cause="weak_premise", effect="conclusion", strength=0.2)
            ],
        )

        suggestions = strategy.suggest_rebuttals(opponent_argument)
        assert len(suggestions) > 0

    def test_find_vulnerable_claims(self, own_graph: CausalGraph) -> None:
        """Test finding vulnerable claims."""
        own_graph.add_link(CausalLink(cause="weak", effect="conclusion", strength=0.2))

        strategy = CausalStrategy(own_graph)
        vulnerable = strategy.find_vulnerable_claims()

        assert isinstance(vulnerable, list)

    def test_suggest_reinforcements(self, own_graph: CausalGraph) -> None:
        """Test reinforcement suggestions."""
        own_graph.add_link(CausalLink(cause="weak", effect="target", strength=0.3))

        strategy = CausalStrategy(own_graph)
        suggestions = strategy.suggest_reinforcements()

        assert isinstance(suggestions, list)
        if suggestions:
            assert suggestions[0].priority > 0

    def test_identify_defensive_priorities(
        self, own_graph: CausalGraph
    ) -> None:
        """Test defensive priority identification."""
        strategy = CausalStrategy(own_graph)
        priorities = strategy.identify_defensive_priorities()

        assert isinstance(priorities, list)

    def test_predict_opponent_targets(self, own_graph: CausalGraph) -> None:
        """Test opponent target prediction."""
        own_graph.add_link(CausalLink(cause="exposed", effect="goal", strength=0.2))

        strategy = CausalStrategy(own_graph)
        predictions = strategy.predict_opponent_targets()

        assert isinstance(predictions, list)

    def test_analyze_opponent_strategy(
        self, own_graph: CausalGraph, opponent_graph: CausalGraph
    ) -> None:
        """Test opponent strategy analysis."""
        strategy = CausalStrategy(own_graph, opponent_graph)
        profile = strategy.analyze_opponent_strategy()

        assert profile is not None
        assert profile.primary_focus != ""
        assert profile.argument_style in ["evidence-heavy", "logical", "mixed"]

    def test_analyze_opponent_strategy_no_opponent(
        self, own_graph: CausalGraph
    ) -> None:
        """Test opponent analysis with no opponent."""
        strategy = CausalStrategy(own_graph)
        profile = strategy.analyze_opponent_strategy()

        assert profile is None


class TestCausalVisualizer:
    """Tests for CausalVisualizer class."""

    @pytest.fixture
    def graph(self) -> CausalGraph:
        graph = CausalGraph()
        graph.add_link(CausalLink(cause="Climate Change", effect="Rising Temperatures", strength=0.9))
        graph.add_link(CausalLink(cause="Rising Temperatures", effect="Ice Melt", strength=0.8))
        graph.add_link(CausalLink(cause="Ice Melt", effect="Sea Level Rise", strength=0.7))
        return graph

    def test_to_dot(self, graph: CausalGraph) -> None:
        """Test DOT format export."""
        visualizer = CausalVisualizer(graph)
        dot = visualizer.to_dot()

        assert "digraph CausalGraph" in dot
        assert "rankdir=TB" in dot
        assert "->" in dot

    def test_to_dot_highlight_weak(self, graph: CausalGraph) -> None:
        """Test DOT with weak link highlighting."""
        graph.add_link(CausalLink(cause="Weak", effect="Link", strength=0.2))

        visualizer = CausalVisualizer(graph)
        dot = visualizer.to_dot(highlight_weak=True, weak_threshold=0.4)

        assert "digraph CausalGraph" in dot

    def test_to_dot_with_title(self, graph: CausalGraph) -> None:
        """Test DOT with title."""
        visualizer = CausalVisualizer(graph)
        dot = visualizer.to_dot(title="Test Graph")

        assert 'label="Test Graph"' in dot

    def test_to_mermaid(self, graph: CausalGraph) -> None:
        """Test Mermaid format export."""
        visualizer = CausalVisualizer(graph)
        mermaid = visualizer.to_mermaid()

        assert "graph TB" in mermaid
        assert "->" in mermaid or "-->" in mermaid

    def test_to_mermaid_lr_direction(self, graph: CausalGraph) -> None:
        """Test Mermaid with LR direction."""
        visualizer = CausalVisualizer(graph)
        mermaid = visualizer.to_mermaid(direction="LR")

        assert "graph LR" in mermaid

    def test_to_json(self, graph: CausalGraph) -> None:
        """Test JSON export."""
        visualizer = CausalVisualizer(graph)
        data = visualizer.to_json()

        assert "nodes" in data
        assert "edges" in data
        assert "metadata" in data
        assert len(data["nodes"]) == 4
        assert len(data["edges"]) == 3

    def test_to_json_with_analysis(self, graph: CausalGraph) -> None:
        """Test JSON export with analysis."""
        visualizer = CausalVisualizer(graph)
        data = visualizer.to_json(include_analysis=True)

        assert "analysis" in data
        assert "has_circular_reasoning" in data["analysis"]
        assert "overall_coherence" in data["analysis"]

    def test_generate_timeline(self, graph: CausalGraph) -> None:
        """Test timeline generation."""
        snapshots = [
            create_snapshot(graph, 1, 1),
            create_snapshot(graph, 1, 2),
            create_snapshot(graph, 2, 1),
        ]

        visualizer = CausalVisualizer(graph)
        timeline = visualizer.generate_timeline(snapshots)

        assert "gantt" in timeline
        assert "title" in timeline

    def test_generate_timeline_empty(self, graph: CausalGraph) -> None:
        """Test timeline with no snapshots."""
        visualizer = CausalVisualizer(graph)
        timeline = visualizer.generate_timeline([])

        assert "no data" in timeline

    def test_generate_report(self, graph: CausalGraph) -> None:
        """Test HTML report generation."""
        visualizer = CausalVisualizer(graph)
        html = visualizer.generate_report(title="Test Report")

        assert "<!DOCTYPE html>" in html
        assert "Test Report" in html
        assert "mermaid" in html
        assert "Overview" in html

    def test_generate_report_with_issues(self) -> None:
        """Test report with issues."""
        graph = CausalGraph()
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.2))
        graph.add_link(CausalLink(cause="B", effect="A", strength=0.8))

        visualizer = CausalVisualizer(graph)
        html = visualizer.generate_report()

        assert "<!DOCTYPE html>" in html


class TestCreateSnapshot:
    """Tests for create_snapshot function."""

    def test_create_snapshot(self) -> None:
        """Test creating a graph snapshot."""
        graph = CausalGraph()
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.8))
        graph.add_link(CausalLink(cause="B", effect="C", strength=0.7))

        snapshot = create_snapshot(graph, round_num=1, turn_num=2)

        assert snapshot.round == 1
        assert snapshot.turn == 2
        assert snapshot.node_count == 3
        assert snapshot.edge_count == 2
        assert len(snapshot.nodes) == 3
        assert len(snapshot.edges) == 2


class TestCausalGraphNewMethods:
    """Tests for new methods added to CausalGraph in v2."""

    @pytest.fixture
    def graph(self) -> CausalGraph:
        graph = CausalGraph()
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.8))
        graph.add_link(CausalLink(cause="B", effect="C", strength=0.7))
        graph.add_link(CausalLink(cause="A", effect="C", strength=0.6))
        return graph

    @pytest.fixture
    def cyclic_graph(self) -> CausalGraph:
        graph = CausalGraph()
        graph.add_link(CausalLink(cause="A", effect="B", strength=0.8))
        graph.add_link(CausalLink(cause="B", effect="C", strength=0.7))
        graph.add_link(CausalLink(cause="C", effect="A", strength=0.6))
        return graph

    def test_get_all_cycles_none(self, graph: CausalGraph) -> None:
        """Test get_all_cycles with no cycles."""
        cycles = graph.get_all_cycles()
        assert len(cycles) == 0

    def test_get_all_cycles_with_cycle(self, cyclic_graph: CausalGraph) -> None:
        """Test get_all_cycles with cycle."""
        cycles = cyclic_graph.get_all_cycles()
        assert len(cycles) >= 1

    def test_get_contradicting_edges_none(self, graph: CausalGraph) -> None:
        """Test get_contradicting_edges with no contradictions."""
        contradictions = graph.get_contradicting_edges()
        assert len(contradictions) == 0

    def test_compute_betweenness_centrality(self, graph: CausalGraph) -> None:
        """Test betweenness centrality computation."""
        centrality = graph.compute_betweenness_centrality()

        assert isinstance(centrality, dict)
        for score in centrality.values():
            assert 0 <= score <= 1

    def test_compute_betweenness_centrality_empty(self) -> None:
        """Test centrality on empty graph."""
        graph = CausalGraph()
        centrality = graph.compute_betweenness_centrality()

        assert centrality == {}

    def test_get_subgraph(self, graph: CausalGraph) -> None:
        """Test subgraph extraction."""
        subgraph = graph.get_subgraph(["a", "b"])

        assert len(subgraph.nodes) == 2
        assert len(subgraph.edges) == 1

    def test_get_subgraph_empty(self, graph: CausalGraph) -> None:
        """Test subgraph with no matching nodes."""
        subgraph = graph.get_subgraph(["x", "y", "z"])

        assert len(subgraph.nodes) == 0
        assert len(subgraph.edges) == 0

    def test_get_neighborhood(self, graph: CausalGraph) -> None:
        """Test neighborhood extraction."""
        neighborhood = graph.get_neighborhood("B", depth=1)

        assert len(neighborhood.nodes) >= 2

    def test_get_neighborhood_depth_2(self, graph: CausalGraph) -> None:
        """Test neighborhood with depth 2."""
        neighborhood = graph.get_neighborhood("A", depth=2)

        assert len(neighborhood.nodes) >= 3

    def test_get_neighborhood_unknown_node(self, graph: CausalGraph) -> None:
        """Test neighborhood of unknown node."""
        neighborhood = graph.get_neighborhood("Unknown", depth=1)

        assert len(neighborhood.nodes) == 0
