"""Strategic analysis for debate agents based on causal graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from artemis.core.causal_analysis import CausalAnalyzer
from artemis.core.types import (
    Argument,
    AttackTarget,
    ReinforcementSuggestion,
    WeakLinkResult,
)

if TYPE_CHECKING:
    from artemis.core.causal import CausalGraph


@dataclass
class VulnerableClaim:
    """A claim that is vulnerable to attack."""

    node_id: str
    label: str
    vulnerability_score: float
    reasons: list[str] = field(default_factory=list)
    suggested_attacks: list[str] = field(default_factory=list)


@dataclass
class DefensePriority:
    """A defensive priority for protecting a claim."""

    node_id: str
    label: str
    priority_score: float
    threat_level: float
    suggested_defenses: list[str] = field(default_factory=list)


@dataclass
class PredictedTarget:
    """A predicted target that opponent might attack."""

    node_id: str
    label: str
    likelihood: float
    reasoning: str


@dataclass
class OpponentStrategyProfile:
    """Profile of opponent's debate strategy based on their graph."""

    primary_focus: str
    """Main theme of opponent's arguments."""
    argument_style: str
    """Style: 'evidence-heavy', 'logical', 'emotional', 'mixed'."""
    weak_points: list[str]
    """Identified weak points in opponent's reasoning."""
    strong_points: list[str]
    """Identified strong points to avoid."""
    predicted_moves: list[str]
    """Predicted next moves by opponent."""


@dataclass
class RebuttalSuggestion:
    """Suggestion for rebutting an opponent's argument."""

    target_claim: str
    """The claim to rebut."""
    strategy: str
    """Rebuttal strategy type."""
    suggested_content: str
    """Suggested rebuttal content."""
    expected_impact: float
    """Expected impact of the rebuttal."""
    evidence_needed: list[str]
    """Types of evidence that would strengthen the rebuttal."""


class CausalStrategy:
    """Strategic analysis for debate agents."""

    def __init__(
        self,
        own_graph: CausalGraph,
        opponent_graph: CausalGraph | None = None,
    ):
        """Initialize strategy analyzer."""
        self.own_graph = own_graph
        self.opponent_graph = opponent_graph
        self._own_analyzer = CausalAnalyzer(own_graph)
        self._opponent_analyzer = (
            CausalAnalyzer(opponent_graph) if opponent_graph else None
        )

    def map_attack_surface(self) -> list[AttackTarget]:
        """Find vulnerable points in opponent's causal graph."""
        if not self.opponent_graph or not self._opponent_analyzer:
            return []

        targets: list[AttackTarget] = []

        weak_links = self._opponent_analyzer.find_weak_links(threshold=0.5)
        for i, weak in enumerate(weak_links[:10]):
            strategies = self._generate_attack_strategies(weak)
            impact = self._estimate_attack_impact(weak)

            targets.append(
                AttackTarget(
                    source=weak.source,
                    target=weak.target,
                    vulnerability_score=weak.weakness_score,
                    attack_strategies=strategies,
                    expected_impact=impact,
                    priority=i,
                )
            )

        contradictions = self._opponent_analyzer.find_contradictions()
        for contradiction in contradictions:
            targets.append(
                AttackTarget(
                    source=contradiction.claim_a_source,
                    target=contradiction.claim_a_target,
                    vulnerability_score=contradiction.severity,
                    attack_strategies=[
                        "Point out the internal contradiction",
                        "Ask opponent to clarify which claim is correct",
                        "Use contradiction to undermine credibility",
                    ],
                    expected_impact=0.8,
                    priority=0,
                )
            )

        return sorted(
            targets, key=lambda x: (x.priority, -x.vulnerability_score)
        )

    def _generate_attack_strategies(
        self, weak: WeakLinkResult
    ) -> list[str]:
        strategies: list[str] = []

        source_label = ""
        target_label = ""
        if self.opponent_graph:
            source_node = self.opponent_graph._nodes.get(weak.source)
            target_node = self.opponent_graph._nodes.get(weak.target)
            if source_node:
                source_label = source_node.label
            if target_node:
                target_label = target_node.label

        if weak.weakness_score > 0.7:
            strategies.append(
                f"Directly challenge the claim that '{source_label}' "
                f"causes '{target_label}'"
            )
            strategies.append("Request evidence supporting this causal link")
        else:
            strategies.append(
                f"Question the strength of the connection between "
                f"'{source_label}' and '{target_label}'"
            )

        strategies.append(f"Propose alternative explanations for '{target_label}'")

        if len(weak.argument_ids) < 2:
            strategies.append(
                "Note that this claim lacks corroborating evidence"
            )

        return strategies

    def _estimate_attack_impact(self, weak: WeakLinkResult) -> float:
        # TODO: might want to cache this for large graphs
        if not self.opponent_graph:
            return 0.5

        target_node = self.opponent_graph._nodes.get(weak.target)
        if not target_node:
            return weak.weakness_score

        downstream = set()
        to_visit = [weak.target]
        while to_visit:
            current = to_visit.pop()
            for neighbor in self.opponent_graph._outgoing.get(current, set()):
                if neighbor not in downstream:
                    downstream.add(neighbor)
                    to_visit.append(neighbor)

        total_nodes = len(self.opponent_graph._nodes)
        if total_nodes == 0:
            return weak.weakness_score

        cascade_factor = len(downstream) / total_nodes
        return min(1.0, weak.weakness_score * 0.5 + cascade_factor * 0.5)

    def suggest_rebuttals(
        self, opponent_argument: Argument
    ) -> list[RebuttalSuggestion]:
        """Suggest rebuttals for an opponent's argument."""
        suggestions: list[RebuttalSuggestion] = []

        for link in opponent_argument.causal_links:
            if link.strength < 0.5:
                suggestions.append(
                    RebuttalSuggestion(
                        target_claim=f"{link.cause} -> {link.effect}",
                        strategy="challenge_causation",
                        suggested_content=(
                            f"The claim that {link.cause} leads to "
                            f"{link.effect} lacks sufficient support. "
                            "Correlation does not imply causation."
                        ),
                        expected_impact=0.7,
                        evidence_needed=[
                            "Studies showing no causal relationship",
                            "Alternative explanations",
                            "Counter-examples",
                        ],
                    )
                )

            if not link.mechanism:
                suggestions.append(
                    RebuttalSuggestion(
                        target_claim=f"{link.cause} -> {link.effect}",
                        strategy="request_mechanism",
                        suggested_content=(
                            f"How exactly does {link.cause} lead to "
                            f"{link.effect}? No mechanism has been provided."
                        ),
                        expected_impact=0.5,
                        evidence_needed=[],
                    )
                )

        if len(opponent_argument.evidence) < 2:
            suggestions.append(
                RebuttalSuggestion(
                    target_claim="overall argument",
                    strategy="insufficient_evidence",
                    suggested_content=(
                        "This argument lacks sufficient evidence to "
                        "support its claims."
                    ),
                    expected_impact=0.4,
                    evidence_needed=[],
                )
            )

        return sorted(
            suggestions, key=lambda x: x.expected_impact, reverse=True
        )

    def find_vulnerable_claims(self) -> list[VulnerableClaim]:
        """Find claims in own graph that are vulnerable to attack."""
        vulnerable: list[VulnerableClaim] = []

        weak_links = self._own_analyzer.find_weak_links(threshold=0.5)
        for weak in weak_links:
            node = self.own_graph._nodes.get(weak.target)
            if not node:
                continue

            reasons = []
            if weak.weakness_score > 0.6:
                reasons.append("Weak causal support")
            if len(weak.argument_ids) < 2:
                reasons.append("Limited evidence")

            vulnerable.append(
                VulnerableClaim(
                    node_id=node.id,
                    label=node.label,
                    vulnerability_score=weak.weakness_score,
                    reasons=reasons,
                    suggested_attacks=weak.attack_suggestions,
                )
            )

        return sorted(
            vulnerable, key=lambda x: x.vulnerability_score, reverse=True
        )

    def suggest_reinforcements(self) -> list[ReinforcementSuggestion]:
        """Suggest reinforcements for weak links in own arguments."""
        suggestions: list[ReinforcementSuggestion] = []

        weak_links = self._own_analyzer.find_weak_links(threshold=0.6)
        for weak in weak_links[:10]:
            source_node = self.own_graph._nodes.get(weak.source)
            target_node = self.own_graph._nodes.get(weak.target)

            if not source_node or not target_node:
                continue

            edge = self.own_graph._edges.get((weak.source, weak.target))
            evidence_types = self._suggest_evidence_types(edge)
            mechanisms = self._suggest_mechanisms(
                source_node.label, target_node.label
            )

            priority = weak.weakness_score
            if self.opponent_graph:
                opponent_strength = self._check_opponent_coverage(
                    weak.source, weak.target
                )
                if opponent_strength > 0.5:
                    priority = min(1.0, priority + 0.2)

            suggestions.append(
                ReinforcementSuggestion(
                    source=source_node.label,
                    target=target_node.label,
                    current_strength=weak.strength,
                    suggested_evidence=evidence_types,
                    suggested_mechanisms=mechanisms,
                    priority=priority,
                )
            )

        return sorted(suggestions, key=lambda x: x.priority, reverse=True)

    def _suggest_evidence_types(self, edge: object) -> list[str]:
        suggestions = []

        if edge and edge.evidence_count < 2:
            suggestions.append("Add empirical studies supporting this link")
            suggestions.append("Include statistical data")

        suggestions.append("Expert testimony or citations")
        suggestions.append("Real-world examples demonstrating the connection")

        return suggestions

    def _suggest_mechanisms(self, source: str, target: str) -> list[str]:
        return [
            f"Explain the process by which {source} leads to {target}",
            "Identify intermediate steps in the causal chain",
            "Reference theoretical frameworks supporting this connection",
        ]

    def _check_opponent_coverage(self, source: str, target: str) -> float:
        if not self.opponent_graph:
            return 0.0

        opponent_edge = self.opponent_graph._edges.get((source, target))
        if opponent_edge:
            return opponent_edge.strength

        return 0.0

    def identify_defensive_priorities(self) -> list[DefensePriority]:
        """Identify defensive priorities based on vulnerability analysis."""
        priorities: list[DefensePriority] = []

        critical_nodes = self._own_analyzer.find_critical_nodes()
        vulnerable_claims = self.find_vulnerable_claims()

        critical_ids = {n.node_id for n in critical_nodes[:5]}
        vulnerable_ids = {v.node_id for v in vulnerable_claims}

        highly_exposed = critical_ids & vulnerable_ids

        for node_id in highly_exposed:
            node = self.own_graph._nodes.get(node_id)
            if not node:
                continue

            critical = next(
                (n for n in critical_nodes if n.node_id == node_id), None
            )
            vulnerable = next(
                (v for v in vulnerable_claims if v.node_id == node_id), None
            )

            if critical and vulnerable:
                priority_score = (
                    critical.centrality_score * 0.5
                    + vulnerable.vulnerability_score * 0.5
                )
                threat_level = critical.impact_if_challenged

                defenses = [
                    f"Strengthen evidence for '{node.label}'",
                    "Add multiple supporting arguments",
                    "Provide detailed mechanism explanation",
                ]

                priorities.append(
                    DefensePriority(
                        node_id=node_id,
                        label=node.label,
                        priority_score=priority_score,
                        threat_level=threat_level,
                        suggested_defenses=defenses,
                    )
                )

        return sorted(priorities, key=lambda x: x.priority_score, reverse=True)

    def predict_opponent_targets(self) -> list[PredictedTarget]:
        """Predict which of our nodes opponent is likely to target."""
        predictions: list[PredictedTarget] = []

        vulnerable = self.find_vulnerable_claims()
        critical = self._own_analyzer.find_critical_nodes()

        vulnerable_ids = {v.node_id: v.vulnerability_score for v in vulnerable}
        critical_ids = {n.node_id: n.centrality_score for n in critical}

        all_ids = set(vulnerable_ids.keys()) | set(critical_ids.keys())

        for node_id in all_ids:
            node = self.own_graph._nodes.get(node_id)
            if not node:
                continue

            vuln_score = vulnerable_ids.get(node_id, 0.0)
            crit_score = critical_ids.get(node_id, 0.0)

            likelihood = vuln_score * 0.6 + crit_score * 0.4

            reasons = []
            if vuln_score > 0.5:
                reasons.append("weak causal support")
            if crit_score > 0.5:
                reasons.append("high impact if challenged")

            reasoning = (
                f"Target due to {', '.join(reasons)}"
                if reasons
                else "Moderate target"
            )

            predictions.append(
                PredictedTarget(
                    node_id=node_id,
                    label=node.label,
                    likelihood=likelihood,
                    reasoning=reasoning,
                )
            )

        return sorted(predictions, key=lambda x: x.likelihood, reverse=True)

    def analyze_opponent_strategy(self) -> OpponentStrategyProfile | None:
        """Analyze opponent's overall debate strategy."""
        if not self.opponent_graph or not self._opponent_analyzer:
            return None

        focus_areas: dict[str, int] = {}
        for node in self.opponent_graph.nodes:
            focus_areas[node.label] = (
                len(self.opponent_graph._incoming.get(node.id, set()))
                + len(self.opponent_graph._outgoing.get(node.id, set()))
            )

        primary_focus = (
            max(focus_areas, key=focus_areas.get)
            if focus_areas
            else "unknown"
        )

        total_evidence = sum(
            e.evidence_count for e in self.opponent_graph.edges
        )
        total_edges = len(self.opponent_graph.edges)
        avg_evidence = total_evidence / max(1, total_edges)

        if avg_evidence > 2:
            style = "evidence-heavy"
        elif total_edges > len(self.opponent_graph.nodes) * 1.5:
            style = "logical"
        else:
            style = "mixed"

        weak_links = self._opponent_analyzer.find_weak_links()
        weak_points = [
            f"{w.source} -> {w.target}"
            for w in weak_links[:3]
        ]

        strong_edges = [
            e for e in self.opponent_graph.edges if e.strength > 0.7
        ]
        strong_points = [
            f"{e.source_id} -> {e.target_id}"
            for e in strong_edges[:3]
        ]

        predicted_moves = []
        if weak_links:
            predicted_moves.append("May try to strengthen weak causal claims")
        if style == "evidence-heavy":
            predicted_moves.append("Likely to cite additional studies/data")
        else:
            predicted_moves.append("May shift to new arguments")

        return OpponentStrategyProfile(
            primary_focus=primary_focus,
            argument_style=style,
            weak_points=weak_points,
            strong_points=strong_points,
            predicted_moves=predicted_moves,
        )
