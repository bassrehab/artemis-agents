"""Causal graph analysis - cycle detection, weak links, fallacies, etc."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

from artemis.core.types import (
    Argument,
    ArgumentStrengthScore,
    CausalAnalysisResult,
    CausalLink,
    CircularReasoningResult,
    ContradictionResult,
    CriticalNodeResult,
    FallacyResult,
    FallacyType,
    ReasoningGap,
    WeakLinkResult,
)

if TYPE_CHECKING:
    from artemis.core.causal import CausalEdge, CausalGraph, LinkType


class CausalAnalyzer:
    """Analyzes CausalGraph for reasoning issues and vulnerabilities."""

    def __init__(self, graph: CausalGraph):
        self.graph = graph

    def find_circular_reasoning(self) -> list[CircularReasoningResult]:
        """Find all cycles in the graph (circular reasoning)."""
        # uses a DFS-based cycle detection - not exactly Tarjan's but works
        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.graph._outgoing.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:].copy()
                    if len(cycle) > 1:
                        cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        for node_id in self.graph._nodes:
            if node_id not in visited:
                dfs(node_id)

        return [self._cycle_to_result(cycle) for cycle in cycles]

    def _cycle_to_result(self, cycle: list[str]) -> CircularReasoningResult:
        """Convert a cycle to a CircularReasoningResult."""
        argument_ids: set[str] = set()
        for node_id in cycle:
            node = self.graph._nodes.get(node_id)
            if node:
                argument_ids.update(node.argument_ids)

        severity = min(1.0, len(cycle) / 3.0)

        return CircularReasoningResult(
            cycle=cycle,
            argument_ids=list(argument_ids),
            severity=severity,
        )

    def get_circular_chains(self) -> list[list[str]]:
        """Get raw cycle lists without metadata."""
        results = self.find_circular_reasoning()
        return [r.cycle for r in results]

    def find_weak_links(self, threshold: float = 0.4) -> list[WeakLinkResult]:
        """Find edges below the given strength threshold."""
        weak: list[WeakLinkResult] = []

        for edge in self.graph.edges:
            if edge.strength < threshold:
                attack_suggestions = self._generate_attack_suggestions(edge)
                weak.append(
                    WeakLinkResult(
                        source=edge.source_id,
                        target=edge.target_id,
                        strength=edge.strength,
                        weakness_score=1.0 - edge.strength,
                        attack_suggestions=attack_suggestions,
                        argument_ids=edge.argument_ids.copy(),
                    )
                )

        return sorted(weak, key=lambda x: x.weakness_score, reverse=True)

    def _generate_attack_suggestions(self, edge: CausalEdge) -> list[str]:
        """Generate attack suggestions for a weak link."""
        from artemis.core.causal import LinkType

        suggestions: list[str] = []
        source_label = self.graph._nodes.get(edge.source_id)
        target_label = self.graph._nodes.get(edge.target_id)

        if not source_label or not target_label:
            return suggestions

        source = source_label.label
        target = target_label.label

        if edge.strength < 0.3:
            suggestions.append(f"Challenge the causal connection between '{source}' and '{target}'")
            suggestions.append("Request empirical evidence for this claim")

        if edge.link_type == LinkType.CAUSES:
            suggestions.append(f"Propose alternative causes for '{target}'")
            suggestions.append("Question whether correlation implies causation")
        elif edge.link_type == LinkType.IMPLIES:
            suggestions.append("Challenge the logical validity of the implication")
        elif edge.link_type == LinkType.CORRELATES:
            suggestions.append("Point out that correlation does not imply causation")

        if edge.evidence_count < 2:
            suggestions.append("Note the lack of supporting evidence")

        return suggestions

    def get_weakest_path(self, start: str, end: str) -> tuple[list[str], float]:
        """Find the weakest link in the strongest path between two nodes."""
        path, _ = self.graph.get_strongest_path(start, end)
        if not path:
            return [], 0.0

        min_strength = 1.0
        for i in range(len(path) - 1):
            edge = self.graph._edges.get((path[i], path[i + 1]))
            if edge and edge.strength < min_strength:
                min_strength = edge.strength

        return path, min_strength

    def find_contradictions(self) -> list[ContradictionResult]:
        """Find edges that contradict each other (opposite types or bidirectional strong claims)."""
        from artemis.core.causal import LinkType

        contradictions: list[ContradictionResult] = []
        edges = list(self.graph._edges.values())

        for i, e1 in enumerate(edges):
            for e2 in edges[i + 1 :]:
                if e1.source_id == e2.source_id and e1.target_id == e2.target_id:
                    if self._are_contradictory_types(e1.link_type, e2.link_type):
                        contradictions.append(
                            ContradictionResult(
                                claim_a_source=e1.source_id,
                                claim_a_target=e1.target_id,
                                claim_a_type=e1.link_type.value,
                                claim_b_source=e2.source_id,
                                claim_b_target=e2.target_id,
                                claim_b_type=e2.link_type.value,
                                agents=list(set(e1.argument_ids) | set(e2.argument_ids)),
                                severity=max(e1.strength, e2.strength),
                                explanation=(
                                    f"Conflicting claims: '{e1.link_type.value}' vs "
                                    f"'{e2.link_type.value}' for same cause-effect pair"
                                ),
                            )
                        )

                if (
                    e1.source_id == e2.target_id
                    and e1.target_id == e2.source_id
                    and e1.link_type == LinkType.CAUSES
                    and e2.link_type == LinkType.CAUSES
                    and e1.strength > 0.6
                    and e2.strength > 0.6
                ):
                    contradictions.append(
                        ContradictionResult(
                            claim_a_source=e1.source_id,
                            claim_a_target=e1.target_id,
                            claim_a_type=e1.link_type.value,
                            claim_b_source=e2.source_id,
                            claim_b_target=e2.target_id,
                            claim_b_type=e2.link_type.value,
                            agents=list(set(e1.argument_ids) | set(e2.argument_ids)),
                            severity=(e1.strength + e2.strength) / 2,
                            explanation=(
                                "Bidirectional strong causation suggests "
                                "circular reasoning or oversimplification"
                            ),
                        )
                    )

        return contradictions

    def _are_contradictory_types(self, t1: LinkType, t2: LinkType) -> bool:
        # CAUSES vs PREVENTS, ENABLES vs PREVENTS
        from artemis.core.causal import LinkType

        contradictions = {
            (LinkType.CAUSES, LinkType.PREVENTS),
            (LinkType.ENABLES, LinkType.PREVENTS),
        }
        return (t1, t2) in contradictions or (t2, t1) in contradictions

    def compute_argument_strength(self, argument_id: str) -> ArgumentStrengthScore:
        """Compute strength score for an argument based on its causal support."""
        nodes_for_argument = [n for n in self.graph.nodes if argument_id in n.argument_ids]

        if not nodes_for_argument:
            return ArgumentStrengthScore(
                argument_id=argument_id,
                overall_score=0.0,
                causal_support=0.0,
                evidence_backing=0.0,
                vulnerability=1.0,
                critical_dependencies=[],
            )

        edges_for_argument = [e for e in self.graph.edges if argument_id in e.argument_ids]

        if not edges_for_argument:
            causal_support = 0.2
            evidence_backing = 0.0
        else:
            avg_strength = sum(e.strength for e in edges_for_argument) / len(edges_for_argument)
            causal_support = avg_strength
            total_evidence = sum(e.evidence_count for e in edges_for_argument)
            evidence_backing = min(1.0, total_evidence / (len(edges_for_argument) * 3))

        weak_edges = [e for e in edges_for_argument if e.strength < 0.4]
        vulnerability = len(weak_edges) / max(1, len(edges_for_argument))

        critical_deps = self._find_critical_dependencies(argument_id)

        overall = causal_support * 0.4 + evidence_backing * 0.3 + (1 - vulnerability) * 0.3

        return ArgumentStrengthScore(
            argument_id=argument_id,
            overall_score=overall,
            causal_support=causal_support,
            evidence_backing=evidence_backing,
            vulnerability=vulnerability,
            critical_dependencies=critical_deps,
        )

    def _find_critical_dependencies(self, argument_id: str) -> list[str]:
        # nodes with high-strength edges into this argument
        deps: list[str] = []
        nodes = [n for n in self.graph.nodes if argument_id in n.argument_ids]

        for node in nodes:
            incoming = self.graph._incoming.get(node.id, set())
            for source_id in incoming:
                edge = self.graph._edges.get((source_id, node.id))
                if edge and edge.strength > 0.7:
                    deps.append(source_id)

        return list(set(deps))

    def rank_arguments_by_strength(self) -> list[tuple[str, float]]:
        """Rank all arguments by their strength scores, descending."""
        argument_ids: set[str] = set()
        for node in self.graph.nodes:
            argument_ids.update(node.argument_ids)
        for edge in self.graph.edges:
            argument_ids.update(edge.argument_ids)

        scores = []
        for arg_id in argument_ids:
            strength = self.compute_argument_strength(arg_id)
            scores.append((arg_id, strength.overall_score))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    def find_critical_nodes(self) -> list[CriticalNodeResult]:
        """Find nodes with high betweenness centrality."""
        # TODO: this is O(n^3) - could use networkx for large graphs
        centrality: dict[str, int] = defaultdict(int)
        nodes = list(self.graph._nodes.keys())

        for source in nodes:
            for target in nodes:
                if source != target:
                    paths = self.graph.find_paths(source, target, max_length=5)
                    for path in paths:
                        for node in path[1:-1]:
                            centrality[node] += 1

        max_centrality = max(centrality.values()) if centrality else 1

        results: list[CriticalNodeResult] = []
        for node_id, score in centrality.items():
            node = self.graph._nodes.get(node_id)
            if not node:
                continue

            in_degree = len(self.graph._incoming.get(node_id, set()))
            out_degree = len(self.graph._outgoing.get(node_id, set()))
            impact = self._estimate_impact(node_id)

            results.append(
                CriticalNodeResult(
                    node_id=node_id,
                    label=node.label,
                    centrality_score=score / max_centrality,
                    dependent_arguments=node.argument_ids.copy(),
                    impact_if_challenged=impact,
                    in_degree=in_degree,
                    out_degree=out_degree,
                )
            )

        return sorted(results, key=lambda x: x.centrality_score, reverse=True)

    def _estimate_impact(self, node_id: str) -> float:
        # BFS to count downstream nodes
        downstream = set()
        to_visit = [node_id]

        while to_visit:
            current = to_visit.pop()
            for neighbor in self.graph._outgoing.get(current, set()):
                if neighbor not in downstream:
                    downstream.add(neighbor)
                    to_visit.append(neighbor)

        total_nodes = len(self.graph._nodes)
        if total_nodes == 0:
            return 0.0

        return len(downstream) / total_nodes

    def find_reasoning_gaps(self) -> list[ReasoningGap]:
        """Find gaps in causal reasoning chains (unsupported premises, weak conclusions)."""
        gaps: list[ReasoningGap] = []

        for node in self.graph.nodes:
            in_degree = len(self.graph._incoming.get(node.id, set()))
            out_degree = len(self.graph._outgoing.get(node.id, set()))

            if in_degree == 0 and out_degree > 0:
                gaps.append(
                    ReasoningGap(
                        start_node="[premise]",
                        end_node=node.id,
                        gap_type="unsupported_premise",
                        severity=0.3,
                        suggested_bridges=[],
                    )
                )

            if out_degree == 0 and in_degree > 0:
                weak_incoming = []
                for source in self.graph._incoming.get(node.id, set()):
                    edge = self.graph._edges.get((source, node.id))
                    if edge and edge.strength < 0.5:
                        weak_incoming.append(source)

                if weak_incoming:
                    gaps.append(
                        ReasoningGap(
                            start_node=weak_incoming[0],
                            end_node=node.id,
                            gap_type="weak_conclusion_support",
                            severity=0.5,
                            suggested_bridges=[],
                        )
                    )

        return gaps

    def compute_chain_completeness(self, path: list[str]) -> float:
        """Compute completeness score for a causal chain (0-1)."""
        if len(path) < 2:
            return 1.0

        total_strength = 0.0
        missing_links = 0

        for i in range(len(path) - 1):
            edge = self.graph._edges.get((path[i], path[i + 1]))
            if edge:
                total_strength += edge.strength
            else:
                missing_links += 1

        expected_links = len(path) - 1
        if expected_links == 0:
            return 1.0

        avg_strength = total_strength / expected_links
        missing_penalty = missing_links / expected_links

        return max(0.0, avg_strength - missing_penalty)

    def analyze(self) -> CausalAnalysisResult:
        """Perform complete analysis of the causal graph."""
        circular = self.find_circular_reasoning()
        weak = self.find_weak_links()
        contradictions = self.find_contradictions()
        critical = self.find_critical_nodes()
        gaps = self.find_reasoning_gaps()

        has_circular = len(circular) > 0

        issue_count = len(circular) + len(contradictions) + len(gaps)
        coherence_penalty = min(0.5, issue_count * 0.1)

        weak_link_penalty = sum(w.weakness_score for w in weak[:5]) * 0.05

        overall_coherence = max(0.0, 1.0 - coherence_penalty - weak_link_penalty)

        argument_strengths = {}
        arg_ids: set[str] = set()
        for node in self.graph.nodes:
            arg_ids.update(node.argument_ids)
        for edge in self.graph.edges:
            arg_ids.update(edge.argument_ids)

        for arg_id in arg_ids:
            strength = self.compute_argument_strength(arg_id)
            argument_strengths[arg_id] = strength

        return CausalAnalysisResult(
            has_circular_reasoning=has_circular,
            circular_chains=circular,
            weak_links=weak,
            contradictions=contradictions,
            critical_nodes=critical,
            reasoning_gaps=gaps,
            fallacies=[],
            overall_coherence=overall_coherence,
            argument_strengths=argument_strengths,
        )


class CausalFallacyDetector:
    """Detect causal reasoning fallacies in arguments and graphs."""

    POST_HOC_PATTERNS = [
        r"after\s+(.+?),?\s+(.+?)\s+(?:happened|occurred|began)",
        r"since\s+(.+?),?\s+(.+?)\s+has\s+(?:been|become)",
        r"following\s+(.+?),?\s+we\s+(?:saw|observed|noticed)",
    ]

    CORRELATION_PATTERNS = [
        r"(.+?)\s+(?:is|are)\s+(?:associated|correlated)\s+with\s+(.+?),"
        r"\s+(?:therefore|so|thus)",
        r"studies?\s+show\s+(.+?)\s+(?:is|are)\s+linked\s+to\s+(.+?),"
        r"\s+(?:proving|showing)\s+(?:that\s+)?(.+?)\s+causes",
    ]

    SLIPPERY_SLOPE_INDICATORS = [
        "will inevitably lead to",
        "will ultimately result in",
        "is just the first step towards",
        "opens the door to",
        "is a slippery slope to",
        "will snowball into",
    ]

    def __init__(self) -> None:
        # XXX: compiling regexes upfront - might be overkill for small inputs
        self._post_hoc_compiled = [re.compile(p, re.IGNORECASE) for p in self.POST_HOC_PATTERNS]
        self._correlation_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.CORRELATION_PATTERNS
        ]

    def detect_fallacies(
        self,
        argument: Argument,
        graph: CausalGraph,
    ) -> list[FallacyResult]:
        """Detect fallacies in an argument using both text and graph analysis."""
        fallacies: list[FallacyResult] = []

        fallacies.extend(self._check_post_hoc(argument))
        fallacies.extend(self._check_false_cause(argument))
        fallacies.extend(self._check_slippery_slope(argument, graph))
        fallacies.extend(self._check_circular_reasoning(argument, graph))

        return fallacies

    def _check_post_hoc(self, argument: Argument) -> list[FallacyResult]:
        results: list[FallacyResult] = []

        for pattern in self._post_hoc_compiled:
            matches = pattern.findall(argument.content)
            for match in matches:
                results.append(
                    FallacyResult(
                        fallacy_type=FallacyType.POST_HOC,
                        description=(
                            "Post hoc fallacy: Assumes that because one event "
                            "followed another, the first caused the second."
                        ),
                        evidence=[f"Pattern match: {match}"],
                        severity=0.6,
                        affected_arguments=[argument.id],
                    )
                )

        return results

    def _check_false_cause(self, argument: Argument) -> list[FallacyResult]:
        results: list[FallacyResult] = []

        for pattern in self._correlation_compiled:
            matches = pattern.findall(argument.content)
            for match in matches:
                results.append(
                    FallacyResult(
                        fallacy_type=FallacyType.FALSE_CAUSE,
                        description=(
                            "False cause fallacy: Treats correlation as "
                            "evidence of causation without sufficient support."
                        ),
                        evidence=[f"Pattern match: {match}"],
                        severity=0.7,
                        affected_arguments=[argument.id],
                    )
                )

        for link in argument.causal_links:
            if link.strength < 0.4 and not link.mechanism:
                results.append(
                    FallacyResult(
                        fallacy_type=FallacyType.FALSE_CAUSE,
                        description=(
                            "Weak causal link without mechanism explanation "
                            "may indicate false cause assumption."
                        ),
                        evidence=[
                            f"Link: {link.cause} -> {link.effect} (strength: {link.strength})"
                        ],
                        severity=0.4,
                        affected_links=[link.id],
                        affected_arguments=[argument.id],
                    )
                )

        return results

    def _check_slippery_slope(
        self,
        argument: Argument,
        graph: CausalGraph,
    ) -> list[FallacyResult]:
        results: list[FallacyResult] = []

        content_lower = argument.content.lower()
        for indicator in self.SLIPPERY_SLOPE_INDICATORS:
            if indicator in content_lower:
                results.append(
                    FallacyResult(
                        fallacy_type=FallacyType.SLIPPERY_SLOPE,
                        description=(
                            "Slippery slope fallacy: Assumes an inevitable "
                            "chain of negative consequences without evidence."
                        ),
                        evidence=[f"Indicator phrase: '{indicator}'"],
                        severity=0.5,
                        affected_arguments=[argument.id],
                    )
                )

        arg_nodes = [n for n in graph.nodes if argument.id in n.argument_ids]
        for node in arg_nodes:
            root_causes = graph.get_root_causes()
            for root in root_causes:
                paths = graph.find_paths(root.label, node.label, max_length=6)
                for path in paths:
                    if len(path) >= 4:
                        strength = graph.compute_path_strength(path)
                        if strength < 0.3:
                            results.append(
                                FallacyResult(
                                    fallacy_type=FallacyType.SLIPPERY_SLOPE,
                                    description=(
                                        "Long causal chain with weak overall "
                                        "strength may indicate slippery slope."
                                    ),
                                    evidence=[f"Path: {' -> '.join(path)}"],
                                    severity=0.6,
                                    affected_arguments=[argument.id],
                                )
                            )

        return results

    def _check_circular_reasoning(
        self,
        argument: Argument,
        graph: CausalGraph,
    ) -> list[FallacyResult]:
        results: list[FallacyResult] = []

        analyzer = CausalAnalyzer(graph)
        cycles = analyzer.find_circular_reasoning()

        for cycle_result in cycles:
            if argument.id in cycle_result.argument_ids:
                results.append(
                    FallacyResult(
                        fallacy_type=FallacyType.CIRCULAR_REASONING,
                        description=(
                            "Circular reasoning: The argument's conclusion "
                            "is used as a premise in a circular chain."
                        ),
                        evidence=[f"Cycle: {' -> '.join(cycle_result.cycle)}"],
                        severity=cycle_result.severity,
                        affected_arguments=[argument.id],
                    )
                )

        return results

    def check_post_hoc(self, link: CausalLink) -> FallacyResult | None:
        """Check a single causal link for post hoc fallacy indicators."""
        if link.strength < 0.3 and not link.mechanism:
            return FallacyResult(
                fallacy_type=FallacyType.POST_HOC,
                description=("Weak temporal link without mechanism may be post hoc."),
                evidence=[f"Link: {link.cause} -> {link.effect}"],
                severity=0.4,
                affected_links=[link.id],
            )
        return None

    def check_false_cause(self, link: CausalLink) -> FallacyResult | None:
        """Check a single causal link for false cause indicators."""
        correlation_words = ["correlate", "associate", "link", "relate"]
        combined = f"{link.cause} {link.effect}".lower()

        for word in correlation_words:
            if word in combined and link.strength > 0.6:
                return FallacyResult(
                    fallacy_type=FallacyType.FALSE_CAUSE,
                    description=(
                        "High-strength link with correlation language "
                        "may conflate correlation with causation."
                    ),
                    evidence=[f"Link: {link.cause} -> {link.effect}"],
                    severity=0.5,
                    affected_links=[link.id],
                )
        return None

    def check_slippery_slope(
        self,
        path: list[str],
        graph: CausalGraph,
    ) -> FallacyResult | None:
        """Check a causal path for slippery slope indicators."""
        if len(path) < 4:
            return None

        strength = graph.compute_path_strength(path)
        if strength < 0.2:
            return FallacyResult(
                fallacy_type=FallacyType.SLIPPERY_SLOPE,
                description=("Long causal chain with very weak cumulative strength."),
                evidence=[f"Path strength: {strength:.2f}"],
                severity=0.7,
            )
        return None
