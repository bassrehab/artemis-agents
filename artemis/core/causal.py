"""Causal graph construction and analysis for argument evaluation."""

import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

from artemis.core.types import CausalLink


class LinkType(str, Enum):
    """Types of causal relationships."""

    CAUSES = "causes"  # Direct causation
    ENABLES = "enables"  # Necessary but not sufficient
    PREVENTS = "prevents"  # Negative causation
    CORRELATES = "correlates"  # Association without clear causation
    IMPLIES = "implies"  # Logical implication


@dataclass
class CausalNode:
    """A node in the causal graph representing a concept or event."""

    id: str
    label: str
    argument_ids: list[str] = field(default_factory=list)  # Arguments mentioning this
    metadata: dict[str, str] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CausalNode):
            return self.id == other.id
        return False


@dataclass
class CausalEdge:
    """An edge in the causal graph representing a causal relationship."""

    source_id: str
    target_id: str
    link_type: LinkType
    strength: float  # 0.0 to 1.0
    evidence_count: int = 1  # Number of arguments supporting this link
    argument_ids: list[str] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash((self.source_id, self.target_id, self.link_type))


class CausalGraph:
    """Directed graph for causal relationships between concepts."""

    def __init__(self):
        self._nodes: dict[str, CausalNode] = {}
        self._edges: dict[tuple[str, str], CausalEdge] = {}
        self._outgoing: dict[str, set[str]] = defaultdict(set)
        self._incoming: dict[str, set[str]] = defaultdict(set)

    def add_node(self, label: str, argument_id: str | None = None) -> str:
        """Add a node to the graph or update existing."""
        node_id = self._normalize_label(label)

        if node_id in self._nodes:
            if argument_id:
                self._nodes[node_id].argument_ids.append(argument_id)
        else:
            self._nodes[node_id] = CausalNode(
                id=node_id,
                label=label,
                argument_ids=[argument_id] if argument_id else [],
            )

        return node_id

    def add_link(
        self,
        link: CausalLink,
        link_type: LinkType = LinkType.CAUSES,
        argument_id: str | None = None,
    ):
        """Add a causal link to the graph."""
        source_id = self.add_node(link.cause, argument_id)
        target_id = self.add_node(link.effect, argument_id)

        edge_key = (source_id, target_id)

        if edge_key in self._edges:
            # Update existing edge
            existing = self._edges[edge_key]
            # Weighted average of strengths
            total_evidence = existing.evidence_count + 1
            existing.strength = (
                existing.strength * existing.evidence_count + link.strength
            ) / total_evidence
            existing.evidence_count = total_evidence
            if argument_id:
                existing.argument_ids.append(argument_id)
        else:
            # Create new edge
            self._edges[edge_key] = CausalEdge(
                source_id=source_id,
                target_id=target_id,
                link_type=link_type,
                strength=link.strength,
                argument_ids=[argument_id] if argument_id else [],
            )
            self._outgoing[source_id].add(target_id)
            self._incoming[target_id].add(source_id)

    def get_node(self, label: str) -> CausalNode | None:
        """Get a node by its label."""
        node_id = self._normalize_label(label)
        return self._nodes.get(node_id)

    def get_edge(self, cause: str, effect: str) -> CausalEdge | None:
        """Get an edge by cause and effect labels."""
        source_id = self._normalize_label(cause)
        target_id = self._normalize_label(effect)
        return self._edges.get((source_id, target_id))

    def get_effects(self, cause: str) -> list[CausalNode]:
        """Get all direct effects of a cause."""
        source_id = self._normalize_label(cause)
        target_ids = self._outgoing.get(source_id, set())
        return [self._nodes[tid] for tid in target_ids if tid in self._nodes]

    def get_causes(self, effect: str) -> list[CausalNode]:
        """Get all direct causes of an effect."""
        target_id = self._normalize_label(effect)
        source_ids = self._incoming.get(target_id, set())
        return [self._nodes[sid] for sid in source_ids if sid in self._nodes]

    def find_paths(self, start: str, end: str, max_length: int = 5):
        """Find all causal paths from start to end."""
        start_id = self._normalize_label(start)
        end_id = self._normalize_label(end)

        if start_id not in self._nodes or end_id not in self._nodes:
            return []

        paths: list[list[str]] = []
        self._find_paths_dfs(start_id, end_id, [start_id], set(), paths, max_length)
        return paths

    def _find_paths_dfs(self, current, end, path, visited, paths, max_length):
        # DFS helper for path finding
        if len(path) > max_length:
            return

        if current == end:
            paths.append(path.copy())
            return

        visited.add(current)

        for neighbor in self._outgoing.get(current, set()):
            if neighbor not in visited:
                path.append(neighbor)
                self._find_paths_dfs(neighbor, end, path, visited, paths, max_length)
                path.pop()

        visited.remove(current)

    def compute_path_strength(self, path):
        """Compute combined strength of a causal path (multiplicative)."""
        if len(path) < 2:
            return 1.0

        strength = 1.0
        for i in range(len(path) - 1):
            edge = self._edges.get((path[i], path[i + 1]))
            if edge:
                strength *= edge.strength
            else:
                return 0.0  # No edge means broken path

        return strength

    def get_strongest_path(self, start: str, end: str):
        """Find strongest causal path between two concepts."""
        paths = self.find_paths(start, end)

        if not paths:
            return [], 0.0

        best_path: list[str] = []
        best_strength = 0.0

        for path in paths:
            strength = self.compute_path_strength(path)
            if strength > best_strength:
                best_strength = strength
                best_path = path

        return best_path, best_strength

    def get_root_causes(self) -> list[CausalNode]:
        """Get all nodes with no incoming edges (root causes)."""
        roots: list[CausalNode] = []
        for node_id, node in self._nodes.items():
            if not self._incoming.get(node_id):
                roots.append(node)
        return roots

    def get_terminal_effects(self) -> list[CausalNode]:
        """Get all nodes with no outgoing edges (terminal effects)."""
        terminals: list[CausalNode] = []
        for node_id, node in self._nodes.items():
            if not self._outgoing.get(node_id):
                terminals.append(node)
        return terminals

    def has_cycle(self) -> bool:
        """Check if the graph contains any cycles."""
        visited: set[str] = set()
        rec_stack: set[str] = set()

        return any(self._has_cycle_dfs(node_id, visited, rec_stack) for node_id in self._nodes)

    def _has_cycle_dfs(self, node, visited, rec_stack):
        visited.add(node)
        rec_stack.add(node)

        for neighbor in self._outgoing.get(node, set()):
            if neighbor not in visited:
                if self._has_cycle_dfs(neighbor, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    def get_all_cycles(self) -> list[list[str]]:
        """Get all cycles in the graph.

        Returns:
            List of cycles, each as a list of node IDs.
        """
        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._outgoing.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:].copy()
                    if len(cycle) > 1:
                        cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        for node_id in self._nodes:
            if node_id not in visited:
                dfs(node_id)

        return cycles

    def get_contradicting_edges(self) -> list[tuple[CausalEdge, CausalEdge]]:
        """Find pairs of edges that contradict each other.

        Contradictions include:
        - Same source and target but opposite link types (CAUSES vs PREVENTS)
        - Bidirectional strong causal claims

        Returns:
            List of (edge1, edge2) tuples representing contradictions.
        """
        contradictions: list[tuple[CausalEdge, CausalEdge]] = []
        edges = list(self._edges.values())

        for i, e1 in enumerate(edges):
            for e2 in edges[i + 1 :]:
                # Same cause-effect pair with opposite types
                if e1.source_id == e2.source_id and e1.target_id == e2.target_id:
                    if self._are_contradictory_types(e1.link_type, e2.link_type):
                        contradictions.append((e1, e2))

                # Bidirectional strong causation
                if (
                    e1.source_id == e2.target_id
                    and e1.target_id == e2.source_id
                    and e1.link_type == LinkType.CAUSES
                    and e2.link_type == LinkType.CAUSES
                    and e1.strength > 0.6
                    and e2.strength > 0.6
                ):
                    contradictions.append((e1, e2))

        return contradictions

    def _are_contradictory_types(self, t1: LinkType, t2: LinkType) -> bool:
        """Check if two link types are contradictory."""
        contradictions = {
            (LinkType.CAUSES, LinkType.PREVENTS),
            (LinkType.ENABLES, LinkType.PREVENTS),
        }
        return (t1, t2) in contradictions or (t2, t1) in contradictions

    def compute_betweenness_centrality(self) -> dict[str, float]:
        """Compute betweenness centrality for all nodes.

        Betweenness centrality measures how often a node lies on
        shortest paths between other nodes.

        Returns:
            Dictionary mapping node_id to centrality score (0-1).
        """
        centrality: dict[str, int] = dict.fromkeys(self._nodes, 0)
        nodes = list(self._nodes.keys())

        for source in nodes:
            for target in nodes:
                if source != target:
                    paths = self.find_paths(source, target, max_length=5)
                    for path in paths:
                        # Exclude endpoints
                        for node in path[1:-1]:
                            centrality[node] += 1

        max_val = max(centrality.values()) if centrality.values() else 1
        if max_val == 0:
            max_val = 1

        return {k: v / max_val for k, v in centrality.items()}

    def get_subgraph(self, node_ids: list[str]) -> "CausalGraph":
        """Extract a subgraph containing only specified nodes.

        Args:
            node_ids: List of node IDs to include.

        Returns:
            New CausalGraph containing only the specified nodes and edges between them.
        """
        subgraph = CausalGraph()
        node_set = set(node_ids)

        # Add nodes
        for node_id in node_ids:
            if node_id in self._nodes:
                node = self._nodes[node_id]
                subgraph._nodes[node_id] = CausalNode(
                    id=node.id,
                    label=node.label,
                    argument_ids=node.argument_ids.copy(),
                    metadata=node.metadata.copy(),
                )

        # Add edges between included nodes
        for (source, target), edge in self._edges.items():
            if source in node_set and target in node_set:
                subgraph._edges[(source, target)] = CausalEdge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    link_type=edge.link_type,
                    strength=edge.strength,
                    evidence_count=edge.evidence_count,
                    argument_ids=edge.argument_ids.copy(),
                )
                subgraph._outgoing[source].add(target)
                subgraph._incoming[target].add(source)

        return subgraph

    def get_neighborhood(self, node_label: str, depth: int = 1) -> "CausalGraph":
        """Get subgraph of nodes within a certain depth from a node.

        Args:
            node_label: Label of the center node.
            depth: Maximum distance from center node.

        Returns:
            CausalGraph containing the neighborhood.
        """
        node_id = self._normalize_label(node_label)
        if node_id not in self._nodes:
            return CausalGraph()

        to_include = {node_id}
        frontier = {node_id}

        for _ in range(depth):
            new_frontier: set[str] = set()
            for current in frontier:
                # Add outgoing neighbors
                new_frontier.update(self._outgoing.get(current, set()))
                # Add incoming neighbors
                new_frontier.update(self._incoming.get(current, set()))
            frontier = new_frontier - to_include
            to_include.update(frontier)

        return self.get_subgraph(list(to_include))

    def merge_nodes(self, labels: list[str], merged_label: str):
        """Merge multiple nodes into one (for synonymous concepts)."""
        node_ids = [self._normalize_label(label) for label in labels]
        existing_ids = [nid for nid in node_ids if nid in self._nodes]

        if not existing_ids:
            return

        merged_id = self.add_node(merged_label)

        # Collect all argument IDs
        all_arg_ids: list[str] = []
        for nid in existing_ids:
            if nid in self._nodes:
                all_arg_ids.extend(self._nodes[nid].argument_ids)

        self._nodes[merged_id].argument_ids = list(set(all_arg_ids))

        # Redirect edges
        for nid in existing_ids:
            if nid == merged_id:
                continue

            # Redirect outgoing edges
            for target in list(self._outgoing.get(nid, set())):
                old_edge = self._edges.pop((nid, target), None)
                if old_edge:
                    new_edge_key = (merged_id, target)
                    if new_edge_key not in self._edges:
                        self._edges[new_edge_key] = CausalEdge(
                            source_id=merged_id,
                            target_id=target,
                            link_type=old_edge.link_type,
                            strength=old_edge.strength,
                            evidence_count=old_edge.evidence_count,
                            argument_ids=old_edge.argument_ids,
                        )
                        self._outgoing[merged_id].add(target)

            # Redirect incoming edges
            for source in list(self._incoming.get(nid, set())):
                old_edge = self._edges.pop((source, nid), None)
                if old_edge:
                    new_edge_key = (source, merged_id)
                    if new_edge_key not in self._edges:
                        self._edges[new_edge_key] = CausalEdge(
                            source_id=source,
                            target_id=merged_id,
                            link_type=old_edge.link_type,
                            strength=old_edge.strength,
                            evidence_count=old_edge.evidence_count,
                            argument_ids=old_edge.argument_ids,
                        )
                        self._incoming[merged_id].add(source)

            # Remove old node
            if nid in self._nodes and nid != merged_id:
                del self._nodes[nid]
            self._outgoing.pop(nid, None)
            self._incoming.pop(nid, None)

    def _normalize_label(self, label: str) -> str:
        """Normalize a label to create a consistent node ID."""
        return label.lower().strip().replace(" ", "_")

    @property
    def nodes(self) -> list[CausalNode]:
        """Get all nodes in the graph."""
        return list(self._nodes.values())

    @property
    def edges(self) -> list[CausalEdge]:
        """Get all edges in the graph."""
        return list(self._edges.values())

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[CausalNode]:
        return iter(self._nodes.values())


class CausalExtractor:
    """Extract causal relationships from text using linguistic patterns."""

    # Causal indicators with their link types
    CAUSAL_PATTERNS = [
        # Direct causation
        (r"(.+?)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(.+?)(?:\.|,|$)", LinkType.CAUSES),
        (
            r"(.+?)\s+(?:is\s+the\s+cause\s+of|is\s+responsible\s+for)\s+(.+?)(?:\.|,|$)",
            LinkType.CAUSES,
        ),
        (r"because\s+(?:of\s+)?(.+?),?\s+(.+?)(?:\.|$)", LinkType.CAUSES),
        (r"(.+?)\s+therefore\s+(.+?)(?:\.|$)", LinkType.CAUSES),
        (r"(.+?)\s+consequently\s+(.+?)(?:\.|$)", LinkType.CAUSES),
        (r"(.+?)\s+thus\s+(.+?)(?:\.|$)", LinkType.CAUSES),
        (r"(.+?)\s+hence\s+(.+?)(?:\.|$)", LinkType.CAUSES),
        (r"as\s+a\s+result\s+of\s+(.+?),?\s+(.+?)(?:\.|$)", LinkType.CAUSES),
        # Enabling
        (r"(.+?)\s+(?:enables?|allows?|permits?)\s+(.+?)(?:\.|,|$)", LinkType.ENABLES),
        (r"(.+?)\s+makes?\s+(?:it\s+)?possible\s+(?:for\s+)?(.+?)(?:\.|,|$)", LinkType.ENABLES),
        # Prevention
        (r"(.+?)\s+(?:prevents?|stops?|blocks?)\s+(.+?)(?:\.|,|$)", LinkType.PREVENTS),
        (r"without\s+(.+?),?\s+(.+?)\s+would(?:n't|\s+not)\s+(.+?)(?:\.|$)", LinkType.PREVENTS),
        # Correlation
        (
            r"(.+?)\s+(?:is\s+)?(?:correlated|associated)\s+with\s+(.+?)(?:\.|,|$)",
            LinkType.CORRELATES,
        ),
        # Implication
        (r"if\s+(.+?),?\s+then\s+(.+?)(?:\.|$)", LinkType.IMPLIES),
        (r"(.+?)\s+implies\s+(?:that\s+)?(.+?)(?:\.|$)", LinkType.IMPLIES),
    ]

    def __init__(self, min_length: int = 3):
        self.min_length = min_length
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), link_type)
            for pattern, link_type in self.CAUSAL_PATTERNS
        ]

    def extract(self, text: str):
        """Extract causal links from text."""
        results = []

        for pattern, link_type in self._compiled_patterns:
            for match in pattern.finditer(text):
                cause = self._clean_phrase(match.group(1))
                effect = self._clean_phrase(match.group(2))

                # Validate phrases
                if not self._is_valid_phrase(cause) or not self._is_valid_phrase(effect):
                    continue

                # Estimate strength based on language certainty
                strength = self._estimate_strength(text, match.start(), match.end())

                link = CausalLink(
                    cause=cause,
                    effect=effect,
                    strength=strength,
                )

                results.append((link, link_type))

        return results

    def _clean_phrase(self, phrase):
        # Remove leading/trailing whitespace and punctuation
        phrase = phrase.strip().strip(".,;:")

        # Remove common filler words at the start
        filler_patterns = [
            r"^(?:the|a|an)\s+",
            r"^(?:this|that|these|those)\s+",
            r"^(?:it|they|we)\s+",
        ]

        for pattern in filler_patterns:
            phrase = re.sub(pattern, "", phrase, flags=re.IGNORECASE)

        return phrase.strip()

    def _is_valid_phrase(self, phrase):
        words = phrase.split()
        return len(words) >= self.min_length // 2 and len(phrase) >= self.min_length

    def _estimate_strength(self, text, start, end):
        # HACK: just looking for certainty markers in surrounding context
        context = text[max(0, start - 30) : min(len(text), end + 30)].lower()

        # Strong certainty markers
        strong_markers = ["always", "definitely", "certainly", "inevitably", "directly"]
        for marker in strong_markers:
            if marker in context:
                return 0.9

        # Moderate certainty markers
        moderate_markers = ["often", "usually", "typically", "likely", "probably"]
        for marker in moderate_markers:
            if marker in context:
                return 0.7

        # Weak certainty markers
        weak_markers = ["sometimes", "might", "could", "possibly", "perhaps"]
        for marker in weak_markers:
            if marker in context:
                return 0.4

        # Default moderate strength
        return 0.6

    def build_graph(self, text: str):
        """Build a causal graph from text."""
        graph = CausalGraph()

        for link, link_type in self.extract(text):
            graph.add_link(link, link_type)

        return graph
