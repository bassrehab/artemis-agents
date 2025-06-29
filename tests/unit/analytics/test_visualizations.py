"""Tests for SVG visualizations in ARTEMIS analytics."""

import pytest

from artemis.analytics.types import MomentumPoint
from artemis.analytics.visualizations import (
    ArgumentFlowDiagram,
    JuryVoteChart,
    MomentumChart,
    ScoreProgressionChart,
    SVGChart,
    TopicCoverageHeatmap,
)


class TestSVGChart:
    """Tests for SVGChart base class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        chart = SVGChart()
        assert chart.width == 600
        assert chart.height == 400
        assert chart.padding == 50

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        chart = SVGChart(width=800, height=600, padding=30)
        assert chart.width == 800
        assert chart.height == 600
        assert chart.padding == 30

    def test_scale_x(self):
        """Test X coordinate scaling."""
        chart = SVGChart(width=600, height=400, padding=50)
        # Middle value should be in middle
        x = chart._scale_x(0.5, 0, 1)
        assert x == 50 + (600 - 100) / 2

    def test_scale_y(self):
        """Test Y coordinate scaling (inverted)."""
        chart = SVGChart(width=600, height=400, padding=50)
        # Max value should be at top (low Y)
        y = chart._scale_y(1.0, 0, 1)
        assert y == 50  # Top of chart area

    def test_get_agent_color_pro(self):
        """Test getting color for 'pro' agent."""
        chart = SVGChart()
        color = chart._get_agent_color("pro")
        assert color == "#4CAF50"  # Green

    def test_get_agent_color_con(self):
        """Test getting color for 'con' agent."""
        chart = SVGChart()
        color = chart._get_agent_color("con")
        assert color == "#F44336"  # Red

    def test_get_agent_color_unknown(self):
        """Test getting color for unknown agent."""
        chart = SVGChart()
        color = chart._get_agent_color("agent_x", index=0)
        assert color == "#2196F3"  # Blue (agent_0)


class TestScoreProgressionChart:
    """Tests for ScoreProgressionChart class."""

    def test_render_empty_data(self):
        """Test rendering with empty data."""
        chart = ScoreProgressionChart()
        svg = chart.render([], [])
        assert "<svg" in svg
        assert "No score data" in svg

    def test_render_single_round(self):
        """Test rendering single round of data."""
        chart = ScoreProgressionChart()
        round_scores = [{"pro": 0.7, "con": 0.5}]
        svg = chart.render(round_scores, ["pro", "con"])

        assert "<svg" in svg
        assert "</svg>" in svg
        assert "circle" in svg  # Data points

    def test_render_multiple_rounds(self):
        """Test rendering multiple rounds."""
        chart = ScoreProgressionChart()
        round_scores = [
            {"pro": 0.6, "con": 0.5},
            {"pro": 0.7, "con": 0.4},
            {"pro": 0.8, "con": 0.3},
        ]
        svg = chart.render(round_scores, ["pro", "con"])

        assert "<svg" in svg
        assert "path" in svg  # Lines
        assert "#4CAF50" in svg  # Pro color (green)
        assert "#F44336" in svg  # Con color (red)

    def test_render_with_turning_points(self):
        """Test rendering with turning point highlights."""
        chart = ScoreProgressionChart()
        round_scores = [
            {"pro": 0.6, "con": 0.5},
            {"pro": 0.5, "con": 0.6},  # Turning point
            {"pro": 0.4, "con": 0.7},
        ]
        svg = chart.render(
            round_scores,
            ["pro", "con"],
            highlight_turning_points=[1],
        )

        assert "<svg" in svg
        assert "FFF3E0" in svg  # Highlight color


class TestMomentumChart:
    """Tests for MomentumChart class."""

    def test_render_empty_data(self):
        """Test rendering with empty data."""
        chart = MomentumChart()
        svg = chart.render([], [])
        assert "<svg" in svg
        assert "No momentum data" in svg

    def test_render_momentum_history(self):
        """Test rendering momentum history."""
        chart = MomentumChart()
        momentum_history = [
            MomentumPoint(round=1, agent="pro", score=0.6, momentum=0.1, cumulative_advantage=0.1),
            MomentumPoint(round=1, agent="con", score=0.5, momentum=-0.1, cumulative_advantage=-0.1),
            MomentumPoint(round=2, agent="pro", score=0.7, momentum=0.2, cumulative_advantage=0.2),
            MomentumPoint(round=2, agent="con", score=0.4, momentum=-0.2, cumulative_advantage=-0.2),
        ]
        svg = chart.render(momentum_history, ["pro", "con"])

        assert "<svg" in svg
        assert "path" in svg
        # Zero line should be dashed
        assert "stroke-dasharray" in svg

    def test_render_includes_legend(self):
        """Test that legend is included."""
        chart = MomentumChart()
        momentum_history = [
            MomentumPoint(round=1, agent="pro", score=0.6, momentum=0.1, cumulative_advantage=0.1),
        ]
        svg = chart.render(momentum_history, ["pro"])

        assert "pro" in svg  # Legend text


class TestJuryVoteChart:
    """Tests for JuryVoteChart class."""

    def test_render_bar_empty(self):
        """Test rendering empty bar chart."""
        chart = JuryVoteChart()
        svg = chart.render_bar({})
        assert "<svg" in svg
        assert "No score data" in svg

    def test_render_bar_with_scores(self):
        """Test rendering bar chart with scores."""
        chart = JuryVoteChart()
        svg = chart.render_bar({"pro": 0.75, "con": 0.55})

        assert "<svg" in svg
        assert "rect" in svg  # Bars
        assert "0.75" in svg  # Score label
        assert "0.55" in svg

    def test_render_pie_empty(self):
        """Test rendering empty pie chart."""
        chart = JuryVoteChart()
        svg = chart.render_pie({})
        assert "<svg" in svg
        assert "No score data" in svg

    def test_render_pie_with_scores(self):
        """Test rendering pie chart with scores."""
        chart = JuryVoteChart()
        svg = chart.render_pie({"pro": 0.6, "con": 0.4})

        assert "<svg" in svg
        assert "path" in svg  # Pie slices


class TestArgumentFlowDiagram:
    """Tests for ArgumentFlowDiagram class."""

    def test_render_empty(self):
        """Test rendering empty diagram."""
        chart = ArgumentFlowDiagram()
        svg = chart.render([], [])
        assert "<svg" in svg
        assert "No argument data" in svg

    def test_render_with_turns(self):
        """Test rendering with turn data."""
        from unittest.mock import MagicMock

        turn1 = MagicMock()
        turn1.id = "turn_1"
        turn1.agent = "pro"
        turn1.round = 1
        turn1.argument = MagicMock()
        turn1.argument.level = MagicMock()
        turn1.argument.level.value = "strategic"
        turn1.argument.rebuts = []
        turn1.evaluation = MagicMock()
        turn1.evaluation.total_score = 0.7

        chart = ArgumentFlowDiagram()
        svg = chart.render([turn1], ["pro"])

        assert "<svg" in svg
        assert "rect" in svg  # Argument boxes


class TestTopicCoverageHeatmap:
    """Tests for TopicCoverageHeatmap class."""

    def test_render_empty(self):
        """Test rendering empty heatmap."""
        chart = TopicCoverageHeatmap()
        svg = chart.render({}, [])
        assert "<svg" in svg
        assert "No coverage data" in svg

    def test_render_with_coverage(self):
        """Test rendering heatmap with coverage data."""
        chart = TopicCoverageHeatmap()
        coverage = {
            "pro": {"economy": 5, "environment": 3},
            "con": {"economy": 2, "environment": 6},
        }
        svg = chart.render(coverage, ["pro", "con"])

        assert "<svg" in svg
        assert "rect" in svg  # Heatmap cells
        assert "5" in svg  # Count labels
