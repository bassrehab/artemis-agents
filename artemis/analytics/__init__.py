"""ARTEMIS Debate Analytics Module.

Provides momentum tracking, metrics calculation, and visualizations for debates.

Example:
    from artemis.analytics import analyze_debate, export_analytics_report

    result = await debate.run()
    analytics = analyze_debate(result)
    export_analytics_report(result, "report.html")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from artemis.analytics.types import (
    DebateAnalytics,
    JurySentiment,
    MomentumPoint,
    RoundMetrics,
    SwayEvent,
    TurningPoint,
)

if TYPE_CHECKING:
    from artemis.core.agent import Agent
    from artemis.core.types import DebateResult, Turn


__all__ = [
    # Types
    "DebateAnalytics",
    "MomentumPoint",
    "SwayEvent",
    "TurningPoint",
    "JurySentiment",
    "RoundMetrics",
    # Main classes
    "DebateAnalyzer",
    # Convenience functions
    "analyze_debate",
    "export_analytics_report",
]


class DebateAnalyzer:
    """Main entry point for debate analytics.

    Computes momentum tracking, metrics, and generates visualizations
    from a debate transcript.

    Example:
        analyzer = DebateAnalyzer(result.transcript, result.metadata.agents)
        analytics = analyzer.analyze()
        html = analyzer.generate_report_html(analytics)
    """

    def __init__(
        self,
        transcript: list[Turn],
        agents: list[Agent] | list[str],
        debate_id: str = "",
        topic: str = "",
    ) -> None:
        """Initialize the analyzer.

        Args:
            transcript: List of Turn objects from the debate
            agents: List of agents or agent names
            debate_id: Optional debate identifier
            topic: Optional debate topic
        """
        self._transcript = transcript
        self._agents = [
            a.name if hasattr(a, "name") else str(a) for a in agents
        ]
        self._debate_id = debate_id
        self._topic = topic
        self._cache: dict[str, Any] = {}

    def analyze(
        self,
        jury_pulse: list[JurySentiment] | None = None,
    ) -> DebateAnalytics:
        """Compute complete analytics for the debate.

        Args:
            jury_pulse: Optional pre-computed jury sentiment history

        Returns:
            DebateAnalytics object with all computed metrics
        """
        from artemis.analytics.metrics import DebateMetricsCalculator
        from artemis.analytics.momentum import MomentumTracker

        # Compute momentum
        momentum_tracker = MomentumTracker()
        momentum_history, turning_points = momentum_tracker.compute_from_transcript(
            self._transcript, self._agents
        )
        sway_events = momentum_tracker.detect_sway_events(self._transcript)

        # Compute metrics
        metrics_calc = DebateMetricsCalculator(self._transcript, self._agents)
        round_metrics = metrics_calc.get_all_round_metrics()

        # Determine number of rounds
        if self._transcript:
            max_round = max(t.round for t in self._transcript)
        else:
            max_round = 0

        # Build final momentum from last round's data
        final_momentum = {}
        if momentum_history:
            for agent in self._agents:
                agent_points = [mp for mp in momentum_history if mp.agent == agent]
                if agent_points:
                    final_momentum[agent] = agent_points[-1].momentum

        return DebateAnalytics(
            debate_id=self._debate_id,
            topic=self._topic,
            agents=self._agents,
            rounds=max_round,
            momentum_history=momentum_history,
            sway_events=sway_events,
            turning_points=turning_points,
            round_metrics=round_metrics,
            jury_sentiment_history=jury_pulse,
            final_momentum=final_momentum,
            rebuttal_effectiveness_overall=metrics_calc.rebuttal_effectiveness,
            evidence_utilization_overall=metrics_calc.evidence_utilization,
            argument_diversity_index=metrics_calc.argument_diversity_index,
        )

    def generate_report_html(
        self,
        analytics: DebateAnalytics | None = None,
        include_charts: list[str] | None = None,
    ) -> str:
        """Generate standalone HTML report with visualizations.

        Args:
            analytics: Pre-computed analytics (computed if not provided)
            include_charts: List of chart types to include, or None for all

        Returns:
            Complete HTML document as string
        """
        if analytics is None:
            analytics = self.analyze()

        from artemis.analytics.visualizations import (
            JuryVoteChart,
            MomentumChart,
            ScoreProgressionChart,
        )

        charts_html = []

        # Score progression chart
        if include_charts is None or "score_progression" in include_charts:
            chart = ScoreProgressionChart()
            round_scores = [rm.agent_scores for rm in analytics.round_metrics]
            if round_scores:
                svg = chart.render(
                    round_scores=round_scores,
                    agents=analytics.agents,
                    highlight_turning_points=[tp.round for tp in analytics.turning_points],
                )
                charts_html.append(f'<div class="chart"><h3>Score Progression</h3>{svg}</div>')

        # Momentum chart
        if include_charts is None or "momentum" in include_charts:
            chart = MomentumChart()
            if analytics.momentum_history:
                svg = chart.render(
                    momentum_history=analytics.momentum_history,
                    agents=analytics.agents,
                )
                charts_html.append(f'<div class="chart"><h3>Momentum Over Time</h3>{svg}</div>')

        # Final jury vote chart
        if include_charts is None or "jury_votes" in include_charts:
            chart = JuryVoteChart()
            if analytics.round_metrics:
                final_scores = analytics.round_metrics[-1].agent_scores
                if final_scores:
                    svg = chart.render_bar(agent_scores=final_scores)
                    charts_html.append(f'<div class="chart"><h3>Final Scores</h3>{svg}</div>')

        # Build HTML document
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Debate Analytics: {analytics.topic or analytics.debate_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #e91e63; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #666; margin-bottom: 10px; }}
        .chart {{ margin: 20px 0; padding: 20px; background: #fafafa; border-radius: 4px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric {{ background: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 4px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #e91e63; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .turning-point {{ background: #fff3e0; border-left: 4px solid #ff9800; padding: 10px; margin: 10px 0; }}
        svg {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Debate Analytics</h1>
        <p><strong>Topic:</strong> {analytics.topic or "N/A"}</p>
        <p><strong>Agents:</strong> {", ".join(analytics.agents)}</p>
        <p><strong>Rounds:</strong> {analytics.rounds}</p>

        <h2>Key Metrics</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{len(analytics.turning_points)}</div>
                <div class="metric-label">Turning Points</div>
            </div>
            <div class="metric">
                <div class="metric-value">{analytics.count_lead_changes()}</div>
                <div class="metric-label">Lead Changes</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(analytics.sway_events)}</div>
                <div class="metric-label">Sway Events</div>
            </div>
        </div>

        <h2>Visualizations</h2>
        {"".join(charts_html)}

        {"<h2>Turning Points</h2>" if analytics.turning_points else ""}
        {"".join(f'<div class="turning-point"><strong>Round {tp.round}</strong> ({tp.agent}): {tp.analysis}</div>' for tp in analytics.turning_points)}
    </div>
</body>
</html>"""
        return html


def analyze_debate(result: DebateResult) -> DebateAnalytics:
    """Convenience function to analyze a DebateResult.

    Args:
        result: DebateResult from a completed debate

    Returns:
        DebateAnalytics with all computed metrics
    """
    analyzer = DebateAnalyzer(
        transcript=result.transcript,
        agents=result.metadata.agents,
        debate_id=result.debate_id,
        topic=result.topic,
    )
    return analyzer.analyze()


def export_analytics_report(
    result: DebateResult,
    output_path: Path | str,
    include_charts: bool = True,
) -> None:
    """Export debate with analytics to HTML file.

    Args:
        result: DebateResult from a completed debate
        output_path: Path to write the HTML report
        include_charts: Whether to include visualizations
    """
    analyzer = DebateAnalyzer(
        transcript=result.transcript,
        agents=result.metadata.agents,
        debate_id=result.debate_id,
        topic=result.topic,
    )
    analytics = analyzer.analyze()

    if include_charts:
        html = analyzer.generate_report_html(analytics)
    else:
        html = analyzer.generate_report_html(analytics, include_charts=[])

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
