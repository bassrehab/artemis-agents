"""SVG visualization generation for ARTEMIS debate analytics.

All charts are rendered as pure SVG strings that can be embedded
directly in HTML without external JavaScript dependencies.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from artemis.analytics.types import MomentumPoint
    from artemis.core.types import Turn


# Default color palette
DEFAULT_COLORS = {
    "pro": "#4CAF50",  # Green
    "con": "#F44336",  # Red
    "agent_0": "#2196F3",  # Blue
    "agent_1": "#FF9800",  # Orange
    "agent_2": "#9C27B0",  # Purple
    "agent_3": "#00BCD4",  # Cyan
    "positive": "#4CAF50",
    "negative": "#F44336",
    "neutral": "#9E9E9E",
    "grid": "#E0E0E0",
    "text": "#333333",
    "background": "#FFFFFF",
}


class SVGChart:
    """Base class for SVG chart generation."""

    def __init__(
        self,
        width: int = 600,
        height: int = 400,
        padding: int = 50,
        colors: dict[str, str] | None = None,
    ) -> None:
        """Initialize chart with dimensions and optional color palette."""
        self.width = width
        self.height = height
        self.padding = padding
        self.colors = {**DEFAULT_COLORS, **(colors or {})}

        # Calculate drawing area
        self.chart_width = width - 2 * padding
        self.chart_height = height - 2 * padding

    def _svg_header(self) -> str:
        # XXX: viewBox ensures scaling works properly
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" width="{self.width}" height="{self.height}">
  <rect width="100%" height="100%" fill="{self.colors["background"]}"/>'''

    def _svg_footer(self) -> str:
        return "</svg>"

    def _scale_x(self, value: float, min_val: float, max_val: float) -> float:
        if max_val == min_val:
            return self.padding + self.chart_width / 2
        ratio = (value - min_val) / (max_val - min_val)
        return self.padding + ratio * self.chart_width

    def _scale_y(self, value: float, min_val: float, max_val: float) -> float:
        # SVG Y-axis is inverted (0 at top)
        if max_val == min_val:
            return self.padding + self.chart_height / 2
        ratio = (value - min_val) / (max_val - min_val)
        return self.padding + self.chart_height - ratio * self.chart_height

    def _get_agent_color(self, agent: str, index: int = 0) -> str:
        if agent.lower() in ["pro", "proponent"]:
            return self.colors["pro"]
        elif agent.lower() in ["con", "opponent"]:
            return self.colors["con"]
        elif agent in self.colors:
            return self.colors[agent]
        else:
            return self.colors.get(f"agent_{index % 4}", "#2196F3")

    def _draw_grid(self, x_ticks: int = 5, y_ticks: int = 5) -> str:
        lines = []

        # Vertical grid lines
        for i in range(x_ticks + 1):
            x = self.padding + (i / x_ticks) * self.chart_width
            lines.append(
                f'<line x1="{x:.1f}" y1="{self.padding}" '
                f'x2="{x:.1f}" y2="{self.padding + self.chart_height}" '
                f'stroke="{self.colors["grid"]}" stroke-width="1"/>'
            )

        # Horizontal grid lines
        for i in range(y_ticks + 1):
            y = self.padding + (i / y_ticks) * self.chart_height
            lines.append(
                f'<line x1="{self.padding}" y1="{y:.1f}" '
                f'x2="{self.padding + self.chart_width}" y2="{y:.1f}" '
                f'stroke="{self.colors["grid"]}" stroke-width="1"/>'
            )

        return "\n  ".join(lines)

    def _draw_axis_labels(
        self,
        x_label: str = "",
        y_label: str = "",
        x_values: list[str] | None = None,
        y_values: list[str] | None = None,
    ) -> str:
        labels = []

        # X-axis label
        if x_label:
            labels.append(
                f'<text x="{self.width / 2}" y="{self.height - 10}" '
                f'text-anchor="middle" font-size="12" fill="{self.colors["text"]}">'
                f"{x_label}</text>"
            )

        # Y-axis label (rotated)
        if y_label:
            labels.append(
                f'<text x="15" y="{self.height / 2}" '
                f'text-anchor="middle" font-size="12" fill="{self.colors["text"]}" '
                f'transform="rotate(-90 15 {self.height / 2})">'
                f"{y_label}</text>"
            )

        # X-axis tick labels
        if x_values:
            for i, val in enumerate(x_values):
                x = (
                    self.padding + (i / (len(x_values) - 1)) * self.chart_width
                    if len(x_values) > 1
                    else self.padding + self.chart_width / 2
                )
                labels.append(
                    f'<text x="{x:.1f}" y="{self.padding + self.chart_height + 20}" '
                    f'text-anchor="middle" font-size="10" fill="{self.colors["text"]}">'
                    f"{val}</text>"
                )

        # Y-axis tick labels
        if y_values:
            for i, val in enumerate(y_values):
                y = (
                    self.padding + self.chart_height - (i / (len(y_values) - 1)) * self.chart_height
                    if len(y_values) > 1
                    else self.padding + self.chart_height / 2
                )
                labels.append(
                    f'<text x="{self.padding - 10}" y="{y:.1f}" '
                    f'text-anchor="end" font-size="10" fill="{self.colors["text"]}" '
                    f'dominant-baseline="middle">{val}</text>'
                )

        return "\n  ".join(labels)

    def _draw_legend(
        self, items: list[tuple[str, str]], x: int | None = None, y: int | None = None
    ) -> str:
        if x is None:
            x = self.width - self.padding - 80
        if y is None:
            y = self.padding + 10

        legend_items = []
        for i, (label, color) in enumerate(items):
            item_y = y + i * 20
            legend_items.append(
                f'<rect x="{x}" y="{item_y}" width="12" height="12" fill="{color}"/>'
                f'<text x="{x + 18}" y="{item_y + 10}" font-size="11" fill="{self.colors["text"]}">{label}</text>'
            )

        return "\n  ".join(legend_items)


class ScoreProgressionChart(SVGChart):
    """Line chart showing agent scores over rounds."""

    def render(
        self,
        round_scores: list[dict[str, float]],
        agents: list[str],
        highlight_turning_points: list[int] | None = None,
    ) -> str:
        """Render score progression as SVG."""
        if not round_scores or not agents:
            return self._render_empty("No score data available")

        parts = [self._svg_header()]

        # Draw grid
        parts.append(self._draw_grid())

        # Calculate scales
        num_rounds = len(round_scores)
        min_score = 0.0
        max_score = 1.0

        # Draw turning point highlights
        if highlight_turning_points:
            for tp_round in highlight_turning_points:
                if 0 <= tp_round < num_rounds:
                    x = (
                        self._scale_x(tp_round, 0, num_rounds - 1)
                        if num_rounds > 1
                        else self.padding + self.chart_width / 2
                    )
                    parts.append(
                        f'<rect x="{x - 5}" y="{self.padding}" width="10" '
                        f'height="{self.chart_height}" fill="#FFF3E0" opacity="0.7"/>'
                    )

        # Draw lines and points for each agent
        legend_items = []
        for agent_idx, agent in enumerate(agents):
            color = self._get_agent_color(agent, agent_idx)
            legend_items.append((agent, color))

            # Build path
            points = []
            for round_idx, scores in enumerate(round_scores):
                score = scores.get(agent, 0.0)
                x = (
                    self._scale_x(round_idx, 0, num_rounds - 1)
                    if num_rounds > 1
                    else self.padding + self.chart_width / 2
                )
                y = self._scale_y(score, min_score, max_score)
                points.append((x, y))

            if points:
                # Draw line
                path_d = f"M {points[0][0]:.1f} {points[0][1]:.1f}"
                for x, y in points[1:]:
                    path_d += f" L {x:.1f} {y:.1f}"

                parts.append(
                    f'<path d="{path_d}" fill="none" stroke="{color}" '
                    f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                )

                # Draw points
                for x, y in points:
                    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')

        # Draw axis labels
        x_values = [f"R{i}" for i in range(num_rounds)]
        y_values = ["0.0", "0.25", "0.5", "0.75", "1.0"]
        parts.append(self._draw_axis_labels("Round", "Score", x_values, y_values))

        # Draw legend
        parts.append(self._draw_legend(legend_items))

        parts.append(self._svg_footer())
        return "\n".join(parts)

    def _render_empty(self, message: str) -> str:
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}">
  <rect width="100%" height="100%" fill="#f5f5f5"/>
  <text x="{self.width / 2}" y="{self.height / 2}" text-anchor="middle" fill="#666">{message}</text>
</svg>'''


class MomentumChart(SVGChart):
    """Area/line chart showing momentum over time."""

    def render(
        self,
        momentum_history: list[MomentumPoint],
        agents: list[str],
        show_confidence_bands: bool = True,
    ) -> str:
        """Render momentum chart as SVG."""
        if not momentum_history or not agents:
            return self._render_empty("No momentum data available")

        parts = [self._svg_header()]

        # Draw grid
        parts.append(self._draw_grid())

        # Get rounds
        rounds = sorted({mp.round for mp in momentum_history})
        if not rounds:
            return self._render_empty("No rounds found")

        min_round, max_round = min(rounds), max(rounds)

        # Draw zero line
        zero_y = self._scale_y(0, -1, 1)
        parts.append(
            f'<line x1="{self.padding}" y1="{zero_y:.1f}" '
            f'x2="{self.padding + self.chart_width}" y2="{zero_y:.1f}" '
            f'stroke="{self.colors["neutral"]}" stroke-width="2" stroke-dasharray="5,5"/>'
        )

        # Draw area and line for each agent
        legend_items = []
        for agent_idx, agent in enumerate(agents):
            color = self._get_agent_color(agent, agent_idx)
            legend_items.append((agent, color))

            agent_points = [mp for mp in momentum_history if mp.agent == agent]
            agent_points.sort(key=lambda mp: mp.round)

            if not agent_points:
                continue

            # Build area path (filled from zero line)
            points = []
            for mp in agent_points:
                x = (
                    self._scale_x(mp.round, min_round, max_round)
                    if max_round > min_round
                    else self.padding + self.chart_width / 2
                )
                y = self._scale_y(mp.momentum, -1, 1)
                points.append((x, y, mp.momentum))

            if points:
                # Create area fill
                area_path = f"M {points[0][0]:.1f} {zero_y:.1f}"
                for x, y, _ in points:
                    area_path += f" L {x:.1f} {y:.1f}"
                area_path += f" L {points[-1][0]:.1f} {zero_y:.1f} Z"

                parts.append(f'<path d="{area_path}" fill="{color}" opacity="0.2"/>')

                # Draw line
                line_path = f"M {points[0][0]:.1f} {points[0][1]:.1f}"
                for x, y, _ in points[1:]:
                    line_path += f" L {x:.1f} {y:.1f}"

                parts.append(
                    f'<path d="{line_path}" fill="none" stroke="{color}" '
                    f'stroke-width="2" stroke-linecap="round"/>'
                )

                # Draw points with color based on positive/negative
                for x, y, momentum in points:
                    point_color = (
                        self.colors["positive"] if momentum >= 0 else self.colors["negative"]
                    )
                    parts.append(
                        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{point_color}" stroke="{color}" stroke-width="1"/>'
                    )

        # ugh, getting the axis labels right took forever
        y_values = ["-1.0", "-0.5", "0", "0.5", "1.0"]
        x_values = [f"R{r}" for r in rounds]
        parts.append(self._draw_axis_labels("Round", "Momentum", x_values, y_values))

        # Draw legend
        parts.append(self._draw_legend(legend_items))

        parts.append(self._svg_footer())
        return "\n".join(parts)

    def _render_empty(self, message: str) -> str:
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}">
  <rect width="100%" height="100%" fill="#f5f5f5"/>
  <text x="{self.width / 2}" y="{self.height / 2}" text-anchor="middle" fill="#666">{message}</text>
</svg>'''


class JuryVoteChart(SVGChart):
    """Bar/pie chart showing jury vote distribution."""

    def render_bar(
        self,
        agent_scores: dict[str, float],
        show_perspectives: bool = False,
        perspective_breakdown: dict[str, dict[str, float]] | None = None,
    ) -> str:
        """Render as horizontal bar chart."""
        if not agent_scores:
            return self._render_empty("No score data available")

        parts = [self._svg_header()]

        agents = list(agent_scores.keys())
        num_agents = len(agents)
        max_score = max(agent_scores.values()) if agent_scores else 1.0

        bar_height = min(40, (self.chart_height - 20) / num_agents)
        bar_gap = 10

        for i, agent in enumerate(agents):
            score = agent_scores[agent]
            color = self._get_agent_color(agent, i)

            y = self.padding + i * (bar_height + bar_gap)
            bar_width = (score / max_score) * (self.chart_width - 100) if max_score > 0 else 0

            # Draw bar
            parts.append(
                f'<rect x="{self.padding + 80}" y="{y:.1f}" '
                f'width="{bar_width:.1f}" height="{bar_height}" '
                f'fill="{color}" rx="4"/>'
            )

            # Draw label
            parts.append(
                f'<text x="{self.padding + 75}" y="{y + bar_height / 2:.1f}" '
                f'text-anchor="end" font-size="12" fill="{self.colors["text"]}" '
                f'dominant-baseline="middle">{agent}</text>'
            )

            # Draw value
            parts.append(
                f'<text x="{self.padding + 85 + bar_width:.1f}" y="{y + bar_height / 2:.1f}" '
                f'font-size="11" fill="{self.colors["text"]}" '
                f'dominant-baseline="middle">{score:.2f}</text>'
            )

        parts.append(self._svg_footer())
        return "\n".join(parts)

    def render_pie(
        self,
        agent_scores: dict[str, float],
    ) -> str:
        """Render as pie chart."""
        if not agent_scores:
            return self._render_empty("No score data available")

        parts = [self._svg_header()]

        total = sum(agent_scores.values())
        if total == 0:
            return self._render_empty("All scores are zero")

        cx = self.width / 2
        cy = self.height / 2 - 20
        radius = min(self.chart_width, self.chart_height) / 2 - 30

        start_angle = -90  # Start from top
        legend_items = []

        for i, (agent, score) in enumerate(agent_scores.items()):
            color = self._get_agent_color(agent, i)
            legend_items.append((f"{agent}: {score:.2f}", color))

            slice_angle = (score / total) * 360
            end_angle = start_angle + slice_angle

            # Calculate arc path
            start_rad = math.radians(start_angle)
            end_rad = math.radians(end_angle)

            x1 = cx + radius * math.cos(start_rad)
            y1 = cy + radius * math.sin(start_rad)
            x2 = cx + radius * math.cos(end_rad)
            y2 = cy + radius * math.sin(end_rad)

            # NOTE: large_arc flag is needed for arcs > 180 degrees
            large_arc = 1 if slice_angle > 180 else 0

            path = (
                f"M {cx:.1f} {cy:.1f} "
                f"L {x1:.1f} {y1:.1f} "
                f"A {radius:.1f} {radius:.1f} 0 {large_arc} 1 {x2:.1f} {y2:.1f} Z"
            )

            parts.append(f'<path d="{path}" fill="{color}" stroke="white" stroke-width="2"/>')

            start_angle = end_angle

        # Draw legend at bottom
        legend_y = self.height - 40
        legend_x = self.padding
        parts.append(self._draw_legend(legend_items, x=legend_x, y=legend_y))

        parts.append(self._svg_footer())
        return "\n".join(parts)

    def _render_empty(self, message: str) -> str:
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}">
  <rect width="100%" height="100%" fill="#f5f5f5"/>
  <text x="{self.width / 2}" y="{self.height / 2}" text-anchor="middle" fill="#666">{message}</text>
</svg>'''


class ArgumentFlowDiagram(SVGChart):
    """Diagram showing argument-rebuttal relationships."""

    def __init__(self, width: int = 800, height: int = 600, **kwargs: Any) -> None:
        super().__init__(width=width, height=height, **kwargs)

    def render(
        self,
        transcript: list[Turn],
        agents: list[str],
        max_depth: int = 10,
    ) -> str:
        """Render argument flow as SVG with swimlanes per agent."""
        if not transcript or not agents:
            return self._render_empty("No argument data available")

        parts = [self._svg_header()]

        num_agents = len(agents)
        lane_width = self.chart_width / num_agents
        box_width = lane_width - 40
        box_height = 60
        box_gap = 20

        # Group turns by agent
        agent_turns: dict[str, list[Turn]] = {a: [] for a in agents}
        for turn in transcript:
            if turn.agent in agents:
                agent_turns[turn.agent].append(turn)

        # Draw swimlane headers
        for i, agent in enumerate(agents):
            color = self._get_agent_color(agent, i)
            x = self.padding + i * lane_width + lane_width / 2

            parts.append(
                f'<text x="{x:.1f}" y="{self.padding - 10}" '
                f'text-anchor="middle" font-size="14" font-weight="bold" '
                f'fill="{color}">{agent}</text>'
            )

            # Draw lane separator
            if i > 0:
                sep_x = self.padding + i * lane_width
                parts.append(
                    f'<line x1="{sep_x:.1f}" y1="{self.padding}" '
                    f'x2="{sep_x:.1f}" y2="{self.height - self.padding}" '
                    f'stroke="{self.colors["grid"]}" stroke-dasharray="5,5"/>'
                )

        # Draw argument boxes
        turn_positions: dict[str, tuple[float, float]] = {}

        for agent_idx, agent in enumerate(agents):
            color = self._get_agent_color(agent, agent_idx)
            turns = agent_turns[agent][:max_depth]

            lane_x = self.padding + agent_idx * lane_width + 20

            for turn_idx, turn in enumerate(turns):
                x = lane_x
                y = self.padding + turn_idx * (box_height + box_gap)

                turn_positions[turn.id] = (x + box_width / 2, y + box_height / 2)

                # Draw box
                level = (
                    turn.argument.level.value
                    if turn.argument and hasattr(turn.argument.level, "value")
                    else "?"
                )
                score = turn.evaluation.total_score if turn.evaluation else 0

                # Color intensity based on score
                opacity = 0.3 + score * 0.7

                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{box_width:.1f}" height="{box_height}" '
                    f'fill="{color}" opacity="{opacity:.2f}" rx="4" stroke="{color}" stroke-width="1"/>'
                )

                # Draw level badge
                parts.append(
                    f'<text x="{x + 5}" y="{y + 15}" font-size="10" fill="white" '
                    f'font-weight="bold">{level[:1].upper()}</text>'
                )

                # Draw round number
                parts.append(
                    f'<text x="{x + box_width - 5}" y="{y + 15}" text-anchor="end" '
                    f'font-size="10" fill="white">R{turn.round}</text>'
                )

                # Draw score
                parts.append(
                    f'<text x="{x + box_width / 2}" y="{y + box_height - 10}" '
                    f'text-anchor="middle" font-size="11" fill="white">{score:.2f}</text>'
                )

        # Draw rebuttal arrows
        for turn in transcript:
            if not turn.argument:
                continue

            if hasattr(turn.argument, "rebuts") and turn.argument.rebuts:
                for rebutted_id in turn.argument.rebuts:
                    if turn.id in turn_positions and rebutted_id in turn_positions:
                        from_pos = turn_positions[turn.id]
                        to_pos = turn_positions[rebutted_id]

                        # Draw curved arrow
                        mid_x = (from_pos[0] + to_pos[0]) / 2
                        mid_y = (from_pos[1] + to_pos[1]) / 2 - 30

                        parts.append(
                            f'<path d="M {from_pos[0]:.1f} {from_pos[1]:.1f} '
                            f'Q {mid_x:.1f} {mid_y:.1f} {to_pos[0]:.1f} {to_pos[1]:.1f}" '
                            f'fill="none" stroke="{self.colors["negative"]}" stroke-width="2" '
                            f'marker-end="url(#arrowhead)"/>'
                        )

        # Add arrowhead marker definition
        parts.insert(
            1,
            """
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#F44336"/>
    </marker>
  </defs>""",
        )

        parts.append(self._svg_footer())
        return "\n".join(parts)

    def _render_empty(self, message: str) -> str:
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}">
  <rect width="100%" height="100%" fill="#f5f5f5"/>
  <text x="{self.width / 2}" y="{self.height / 2}" text-anchor="middle" fill="#666">{message}</text>
</svg>'''


class TopicCoverageHeatmap(SVGChart):
    """Heatmap showing topic coverage by agent."""

    def render(
        self,
        coverage: dict[str, dict[str, int]],
        agents: list[str],
    ) -> str:
        """Render topic coverage heatmap."""
        if not coverage or not agents:
            return self._render_empty("No coverage data available")

        parts = [self._svg_header()]

        # Collect all topics
        all_topics: set[str] = set()
        for agent_topics in coverage.values():
            all_topics.update(agent_topics.keys())

        topics = sorted(all_topics)[:15]  # Limit to 15 topics

        if not topics:
            return self._render_empty("No topics found")

        # Calculate cell dimensions
        cell_width = (self.chart_width - 100) / len(agents)
        cell_height = min(25, (self.chart_height - 40) / len(topics))

        # Find max count for color scaling
        max_count = (
            max(count for agent_topics in coverage.values() for count in agent_topics.values())
            if coverage
            else 1
        )

        # Draw column headers (agents)
        for i, agent in enumerate(agents):
            x = self.padding + 100 + i * cell_width + cell_width / 2
            color = self._get_agent_color(agent, i)
            parts.append(
                f'<text x="{x:.1f}" y="{self.padding - 5}" '
                f'text-anchor="middle" font-size="11" font-weight="bold" '
                f'fill="{color}">{agent}</text>'
            )

        # Draw heatmap cells
        for row_idx, topic in enumerate(topics):
            y = self.padding + row_idx * cell_height

            # Draw topic label
            display_topic = topic[:15] + "..." if len(topic) > 15 else topic
            parts.append(
                f'<text x="{self.padding + 95}" y="{y + cell_height / 2:.1f}" '
                f'text-anchor="end" font-size="10" fill="{self.colors["text"]}" '
                f'dominant-baseline="middle">{display_topic}</text>'
            )

            for col_idx, agent in enumerate(agents):
                x = self.padding + 100 + col_idx * cell_width
                count = coverage.get(agent, {}).get(topic, 0)

                # Calculate color intensity
                intensity = count / max_count if max_count > 0 else 0
                color = self._get_heat_color(intensity)

                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" '
                    f'width="{cell_width - 2}" height="{cell_height - 2}" '
                    f'fill="{color}" rx="2"/>'
                )

                # Show count if > 0
                if count > 0:
                    parts.append(
                        f'<text x="{x + cell_width / 2:.1f}" y="{y + cell_height / 2:.1f}" '
                        f'text-anchor="middle" font-size="9" fill="white" '
                        f'dominant-baseline="middle">{count}</text>'
                    )

        parts.append(self._svg_footer())
        return "\n".join(parts)

    def _get_heat_color(self, intensity: float) -> str:
        if intensity == 0:
            return "#f5f5f5"

        # Gradient from light pink to dark pink
        r = int(233 - intensity * 100)
        g = int(30 + (1 - intensity) * 150)
        b = int(99 + (1 - intensity) * 100)

        return f"rgb({r},{g},{b})"

    def _render_empty(self, message: str) -> str:
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}">
  <rect width="100%" height="100%" fill="#f5f5f5"/>
  <text x="{self.width / 2}" y="{self.height / 2}" text-anchor="middle" fill="#666">{message}</text>
</svg>'''
