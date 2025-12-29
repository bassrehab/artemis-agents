"""Export causal graphs to DOT, Mermaid, JSON, and HTML formats."""

from __future__ import annotations

import html
from datetime import datetime
from typing import TYPE_CHECKING

from artemis.core.types import CausalAnalysisResult, GraphSnapshot

if TYPE_CHECKING:
    from artemis.core.causal import CausalGraph


class CausalVisualizer:
    """Export causal graphs to various visualization formats."""

    LINK_TYPE_STYLES = {
        "causes": {"color": "#2563eb", "style": "solid", "arrow": "->"},
        "enables": {"color": "#16a34a", "style": "dashed", "arrow": "-->"},
        "prevents": {"color": "#dc2626", "style": "solid", "arrow": "-X>"},
        "correlates": {"color": "#9333ea", "style": "dotted", "arrow": "-.->"},
        "implies": {"color": "#ea580c", "style": "dashed", "arrow": "==>"},
    }

    def __init__(self, graph: CausalGraph):
        """Initialize visualizer with a causal graph."""
        self.graph = graph

    def to_dot(
        self,
        highlight_weak: bool = False,
        weak_threshold: float = 0.4,
        title: str | None = None,
    ) -> str:
        """Export graph to DOT format for Graphviz."""
        lines = ["digraph CausalGraph {"]
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box, style=rounded];")

        if title:
            lines.append(f'    label="{title}";')
            lines.append("    labelloc=t;")

        for node in self.graph.nodes:
            node_id = self._escape_dot_id(node.id)
            label = self._escape_dot_label(node.label)
            in_deg = len(self.graph._incoming.get(node.id, set()))
            out_deg = len(self.graph._outgoing.get(node.id, set()))

            if in_deg == 0:
                color = "#16a34a"
                shape = "ellipse"
            elif out_deg == 0:
                color = "#dc2626"
                shape = "box"
            else:
                color = "#2563eb"
                shape = "box"

            lines.append(
                f'    "{node_id}" [label="{label}", '
                f'fillcolor="{color}", style="filled,rounded", '
                f"shape={shape}];"
            )

        for edge in self.graph.edges:
            source = self._escape_dot_id(edge.source_id)
            target = self._escape_dot_id(edge.target_id)

            style_info = self.LINK_TYPE_STYLES.get(
                edge.link_type.value, self.LINK_TYPE_STYLES["causes"]
            )

            if highlight_weak and edge.strength < weak_threshold:
                color = "#dc2626"
                penwidth = "1.0"
            else:
                color = style_info["color"]
                penwidth = str(1 + edge.strength * 2)

            style = style_info["style"]
            label = f"{edge.strength:.2f}"

            lines.append(
                f'    "{source}" -> "{target}" '
                f'[label="{label}", color="{color}", '
                f"penwidth={penwidth}, style={style}];"
            )

        lines.append("}")
        return "\n".join(lines)

    def _escape_dot_id(self, s: str) -> str:
        return s.replace('"', '\\"').replace("\n", " ")

    def _escape_dot_label(self, s: str) -> str:
        return s.replace('"', '\\"').replace("\n", "\\n")

    def to_mermaid(
        self,
        direction: str = "TB",
        highlight_weak: bool = False,
        weak_threshold: float = 0.4,
    ) -> str:
        """Export graph to Mermaid diagram format."""
        lines = [f"graph {direction}"]

        node_ids: dict[str, str] = {}
        for i, node in enumerate(self.graph.nodes):
            mermaid_id = f"N{i}"
            node_ids[node.id] = mermaid_id
            label = self._escape_mermaid(node.label)

            in_deg = len(self.graph._incoming.get(node.id, set()))
            out_deg = len(self.graph._outgoing.get(node.id, set()))

            if in_deg == 0:
                lines.append(f"    {mermaid_id}(({label}))")
            elif out_deg == 0:
                lines.append(f"    {mermaid_id}[/{label}/]")
            else:
                lines.append(f"    {mermaid_id}[{label}]")

        for edge in self.graph.edges:
            source = node_ids.get(edge.source_id, edge.source_id)
            target = node_ids.get(edge.target_id, edge.target_id)

            style_info = self.LINK_TYPE_STYLES.get(
                edge.link_type.value, self.LINK_TYPE_STYLES["causes"]
            )
            arrow = style_info["arrow"]

            label = f"|{edge.strength:.2f}|"

            if highlight_weak and edge.strength < weak_threshold:
                lines.append(f"    {source} {arrow} {label} {target}")
                lines.append(f"    linkStyle {len(lines) - 2} stroke:#dc2626,stroke-width:2px")
            else:
                lines.append(f"    {source} {arrow} {label} {target}")

        return "\n".join(lines)

    def _escape_mermaid(self, s: str) -> str:
        # mermaid chokes on brackets and quotes
        return s.replace('"', "'").replace("\n", " ").replace("[", "(").replace("]", ")")

    def to_json(self, include_analysis: bool = False) -> dict[str, object]:
        """Export graph to JSON format."""
        from artemis.core.causal_analysis import CausalAnalyzer

        nodes = []
        for node in self.graph.nodes:
            nodes.append(
                {
                    "id": node.id,
                    "label": node.label,
                    "argument_ids": node.argument_ids,
                    "in_degree": len(self.graph._incoming.get(node.id, set())),
                    "out_degree": len(self.graph._outgoing.get(node.id, set())),
                }
            )

        edges = []
        for edge in self.graph.edges:
            edges.append(
                {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "link_type": edge.link_type.value,
                    "strength": edge.strength,
                    "evidence_count": edge.evidence_count,
                    "argument_ids": edge.argument_ids,
                }
            )

        result = {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "exported_at": datetime.utcnow().isoformat(),
            },
        }

        if include_analysis:
            analyzer = CausalAnalyzer(self.graph)
            analysis = analyzer.analyze()
            result["analysis"] = {
                "has_circular_reasoning": analysis.has_circular_reasoning,
                "circular_chains": [c.model_dump() for c in analysis.circular_chains],
                "weak_links_count": len(analysis.weak_links),
                "contradictions_count": len(analysis.contradictions),
                "overall_coherence": analysis.overall_coherence,
            }

        return result

    def generate_timeline(self, snapshots: list[GraphSnapshot]) -> str:
        """Generate a timeline visualization showing graph evolution."""
        if not snapshots:
            return "gantt\n    title Graph Evolution\n    (no data)"

        lines = ["gantt"]
        lines.append("    title Causal Graph Evolution")
        lines.append("    dateFormat X")
        lines.append("    axisFormat Turn %s")
        lines.append("")
        lines.append("    section Nodes")

        for i, snap in enumerate(snapshots):
            lines.append(f"    Round {snap.round} Turn {snap.turn} :n{i}, {i}, {i + 1}")

        lines.append("")
        lines.append("    section Growth")

        prev_nodes = 0
        prev_edges = 0
        for i, snap in enumerate(snapshots):
            node_delta = snap.node_count - prev_nodes
            edge_delta = snap.edge_count - prev_edges

            if node_delta > 0 or edge_delta > 0:
                lines.append(f"    +{node_delta}N +{edge_delta}E :g{i}, {i}, {i + 1}")

            prev_nodes = snap.node_count
            prev_edges = snap.edge_count

        return "\n".join(lines)

    def generate_report(
        self,
        analysis: CausalAnalysisResult | None = None,
        title: str = "Causal Graph Analysis Report",
    ) -> str:
        """Generate an HTML report with graph visualization and analysis."""
        from artemis.core.causal_analysis import CausalAnalyzer

        if analysis is None:
            analyzer = CausalAnalyzer(self.graph)
            analysis = analyzer.analyze()

        mermaid_diagram = self.to_mermaid(highlight_weak=True)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #1f2937;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #374151;
            margin-top: 30px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 6px;
            background: #eff6ff;
            color: #1e40af;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 12px;
            opacity: 0.8;
        }}
        .warning {{
            background: #fef3c7;
            color: #92400e;
        }}
        .danger {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .success {{
            background: #d1fae5;
            color: #065f46;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}
        th {{
            background: #f9fafb;
            font-weight: 600;
        }}
        .mermaid {{
            background: white;
            padding: 20px;
            border-radius: 8px;
        }}
        .severity-high {{
            color: #dc2626;
            font-weight: bold;
        }}
        .severity-medium {{
            color: #d97706;
        }}
        .severity-low {{
            color: #16a34a;
        }}
    </style>
</head>
<body>
    <h1>{html.escape(title)}</h1>

    <div class="card">
        <h2>Overview</h2>
        <div class="metric">
            <div class="metric-value">{len(self.graph.nodes)}</div>
            <div class="metric-label">Nodes</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(self.graph.edges)}</div>
            <div class="metric-label">Edges</div>
        </div>
        <div class="metric {"success" if analysis.overall_coherence > 0.7 else "warning" if analysis.overall_coherence > 0.4 else "danger"}">
            <div class="metric-value">{analysis.overall_coherence:.0%}</div>
            <div class="metric-label">Coherence</div>
        </div>
        <div class="metric {"danger" if analysis.has_circular_reasoning else "success"}">
            <div class="metric-value">{"Yes" if analysis.has_circular_reasoning else "No"}</div>
            <div class="metric-label">Circular Reasoning</div>
        </div>
    </div>

    <div class="card">
        <h2>Graph Visualization</h2>
        <pre class="mermaid">
{mermaid_diagram}
        </pre>
    </div>

    {self._generate_weak_links_section(analysis)}
    {self._generate_contradictions_section(analysis)}
    {self._generate_critical_nodes_section(analysis)}
    {self._generate_fallacies_section(analysis)}

    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>
"""
        return html_content

    def _generate_weak_links_section(self, analysis: CausalAnalysisResult) -> str:
        if not analysis.weak_links:
            return ""

        rows = ""
        for weak in analysis.weak_links[:10]:
            severity_class = (
                "severity-high"
                if weak.weakness_score > 0.7
                else "severity-medium"
                if weak.weakness_score > 0.4
                else "severity-low"
            )
            rows += f"""
            <tr>
                <td>{html.escape(weak.source)}</td>
                <td>{html.escape(weak.target)}</td>
                <td class="{severity_class}">{weak.strength:.2f}</td>
                <td>{", ".join(weak.attack_suggestions[:2]) if weak.attack_suggestions else "-"}</td>
            </tr>
            """

        return f"""
    <div class="card">
        <h2>Weak Links ({len(analysis.weak_links)} found)</h2>
        <table>
            <tr>
                <th>Source</th>
                <th>Target</th>
                <th>Strength</th>
                <th>Attack Suggestions</th>
            </tr>
            {rows}
        </table>
    </div>
        """

    def _generate_contradictions_section(self, analysis: CausalAnalysisResult) -> str:
        if not analysis.contradictions:
            return ""

        rows = ""
        for c in analysis.contradictions:
            rows += f"""
            <tr>
                <td>{html.escape(c.claim_a_source)} → {html.escape(c.claim_a_target)}</td>
                <td>{html.escape(c.claim_a_type)}</td>
                <td>{html.escape(c.claim_b_type)}</td>
                <td class="severity-high">{c.severity:.2f}</td>
            </tr>
            """

        return f"""
    <div class="card">
        <h2>Contradictions ({len(analysis.contradictions)} found)</h2>
        <table>
            <tr>
                <th>Claim</th>
                <th>Type A</th>
                <th>Type B</th>
                <th>Severity</th>
            </tr>
            {rows}
        </table>
    </div>
        """

    def _generate_critical_nodes_section(self, analysis: CausalAnalysisResult) -> str:
        if not analysis.critical_nodes:
            return ""

        rows = ""
        for node in analysis.critical_nodes[:5]:
            rows += f"""
            <tr>
                <td>{html.escape(node.label)}</td>
                <td>{node.centrality_score:.2f}</td>
                <td>{node.impact_if_challenged:.0%}</td>
                <td>{node.in_degree} / {node.out_degree}</td>
            </tr>
            """

        return f"""
    <div class="card">
        <h2>Critical Nodes (Top 5)</h2>
        <table>
            <tr>
                <th>Node</th>
                <th>Centrality</th>
                <th>Impact if Challenged</th>
                <th>In/Out Degree</th>
            </tr>
            {rows}
        </table>
    </div>
        """

    def _generate_fallacies_section(self, analysis: CausalAnalysisResult) -> str:
        if not analysis.fallacies:
            return ""

        rows = ""
        for f in analysis.fallacies:
            severity_class = (
                "severity-high"
                if f.severity > 0.7
                else "severity-medium"
                if f.severity > 0.4
                else "severity-low"
            )
            rows += f"""
            <tr>
                <td>{html.escape(f.fallacy_type.value)}</td>
                <td>{html.escape(f.description)}</td>
                <td class="{severity_class}">{f.severity:.2f}</td>
            </tr>
            """

        return f"""
    <div class="card">
        <h2>Detected Fallacies ({len(analysis.fallacies)} found)</h2>
        <table>
            <tr>
                <th>Type</th>
                <th>Description</th>
                <th>Severity</th>
            </tr>
            {rows}
        </table>
    </div>
        """


def create_snapshot(
    graph: CausalGraph,
    round_num: int,
    turn_num: int,
) -> GraphSnapshot:
    """Create a snapshot of the current graph state."""
    return GraphSnapshot(
        round=round_num,
        turn=turn_num,
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        nodes=[n.id for n in graph.nodes],
        edges=[(e.source_id, e.target_id) for e in graph.edges],
    )
