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

    def generate_comprehensive_report(
        self,
        analytics: DebateAnalytics | None = None,
        result: "DebateResult | None" = None,
    ) -> str:
        """Generate comprehensive HTML report with analytics AND full transcript.

        Args:
            analytics: Pre-computed analytics (computed if not provided)
            result: DebateResult for full transcript details

        Returns:
            Complete HTML document as string
        """
        import html as html_escape

        if analytics is None:
            analytics = self.analyze()

        from artemis.analytics.visualizations import (
            JuryVoteChart,
            MomentumChart,
            ScoreProgressionChart,
        )

        # Generate charts
        charts_html = []

        # Score progression chart
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
        chart = MomentumChart()
        if analytics.momentum_history:
            svg = chart.render(
                momentum_history=analytics.momentum_history,
                agents=analytics.agents,
            )
            charts_html.append(f'<div class="chart"><h3>Momentum Over Time</h3>{svg}</div>')

        # Final jury vote chart
        chart = JuryVoteChart()
        if analytics.round_metrics:
            final_scores = analytics.round_metrics[-1].agent_scores
            if final_scores:
                svg = chart.render_bar(agent_scores=final_scores)
                charts_html.append(f'<div class="chart"><h3>Final Scores</h3>{svg}</div>')

        # Build transcript HTML
        transcript_html = []
        if self._transcript:
            current_round = None
            agent_colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50", "#E91E63"]

            for i, turn in enumerate(self._transcript):
                if turn.round != current_round:
                    current_round = turn.round
                    round_label = "Opening Statements" if current_round == 0 else f"Round {current_round}"
                    transcript_html.append(f"<h3>{round_label}</h3>")

                agent_idx = self._agents.index(turn.agent) if turn.agent in self._agents else 0
                color = agent_colors[agent_idx % len(agent_colors)]

                transcript_html.append(f"<div class='turn' style='border-left-color: {color}'>")
                transcript_html.append(f"<div class='turn-header'>")
                transcript_html.append(f"<h4>{html_escape.escape(turn.agent)}</h4>")
                transcript_html.append(f"<span class='turn-meta'>Level: {turn.argument.level.value} | ID: <code>{turn.id}</code></span>")
                transcript_html.append("</div>")

                # Full argument content
                content = turn.argument.content
                transcript_html.append(f"<div class='turn-content'>{html_escape.escape(content)}</div>")

                # Evidence
                if turn.argument.evidence:
                    transcript_html.append(f"<div class='evidence'><h5>Evidence ({len(turn.argument.evidence)})</h5>")
                    for ev in turn.argument.evidence:
                        verified = " (Verified)" if getattr(ev, 'verified', False) else ""
                        transcript_html.append(f"<div class='evidence-item'>")
                        transcript_html.append(f"<div class='type'>{html_escape.escape(ev.type)}</div>")
                        transcript_html.append(f"<div class='content'>{html_escape.escape(ev.content)}</div>")
                        transcript_html.append(f"<div class='meta'>Source: {html_escape.escape(ev.source or 'N/A')} | Confidence: {ev.confidence:.2f}{verified}</div>")
                        transcript_html.append("</div>")
                    transcript_html.append("</div>")

                # Causal links
                if turn.argument.causal_links:
                    transcript_html.append(f"<div class='causal-links'><h5>Causal Links ({len(turn.argument.causal_links)})</h5>")
                    for link in turn.argument.causal_links:
                        transcript_html.append(f"<div class='causal-link'>")
                        transcript_html.append(f"<span class='cause'>{html_escape.escape(link.cause)}</span>")
                        transcript_html.append("<span class='arrow'>→</span>")
                        transcript_html.append(f"<span class='effect'>{html_escape.escape(link.effect)}</span>")
                        transcript_html.append("</div>")
                    transcript_html.append("</div>")

                # Rebuts/Supports
                if turn.argument.rebuts or turn.argument.supports:
                    transcript_html.append("<div class='rebuts-supports'>")
                    if turn.argument.rebuts:
                        transcript_html.append("<span class='rebuts'><strong>Rebuts:</strong> ")
                        for r in turn.argument.rebuts:
                            transcript_html.append(f"<span>{html_escape.escape(r)}</span> ")
                        transcript_html.append("</span>")
                    if turn.argument.supports:
                        transcript_html.append("<span class='supports'><strong>Supports:</strong> ")
                        for s in turn.argument.supports:
                            transcript_html.append(f"<span>{html_escape.escape(s)}</span> ")
                        transcript_html.append("</span>")
                    transcript_html.append("</div>")

                # Evaluation
                if turn.evaluation:
                    transcript_html.append("<div class='evaluation'>")
                    transcript_html.append(f"<details><summary>Evaluation Score: {turn.evaluation.total_score:.2f}</summary>")
                    if turn.evaluation.scores:
                        transcript_html.append("<div class='evaluation-grid'>")
                        for criterion, score in turn.evaluation.scores.items():
                            weight = turn.evaluation.weights.get(criterion, 0)
                            transcript_html.append(f"<div class='eval-metric'><div class='value'>{score:.2f}</div><div class='label'>{criterion} (w: {weight:.2f})</div></div>")
                        transcript_html.append("</div>")
                    transcript_html.append("</details></div>")

                # Safety results
                for safety in turn.safety_results:
                    if safety.severity > 0.3:
                        css_class = "failed" if safety.severity > 0.7 else "warning"
                        icon = "⛔" if safety.severity > 0.7 else "⚠️"
                        transcript_html.append(f"<div class='safety-check {css_class}'>")
                        transcript_html.append(f"<span class='safety-icon'>{icon}</span>")
                        transcript_html.append(f"<div><strong>{html_escape.escape(safety.monitor)}</strong> - Severity: {safety.severity:.2f}")
                        if safety.analysis_notes:
                            transcript_html.append(f"<br><small>{html_escape.escape(safety.analysis_notes)}</small>")
                        transcript_html.append("</div></div>")

                transcript_html.append("</div>")

        # Verdict section
        verdict_html = ""
        if result and result.verdict:
            verdict = result.verdict
            verdict_html = f"""
            <div class='verdict'>
                <h2>Verdict</h2>
                <div class='verdict-decision'>{html_escape.escape(verdict.decision)}</div>
                <div class='verdict-confidence'>Confidence: {verdict.confidence:.0%} | {'Unanimous' if verdict.unanimous else 'Not Unanimous'}</div>
                <div class='verdict-reasoning'><h4>Reasoning</h4><p>{html_escape.escape(verdict.reasoning)}</p></div>
                <div class='scores'>
                    {"".join(f"<div class='score-card'><div class='score-value'>{score:.2f}</div><div class='score-label'>{html_escape.escape(agent)}</div></div>" for agent, score in verdict.score_breakdown.items())}
                </div>
            </div>
            """

        # Build complete HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Debate Report: {html_escape.escape(analytics.topic or analytics.debate_id)}</title>
    <style>
        :root {{ --primary: #e91e63; --success: #4CAF50; --warning: #FF9800; --danger: #f44336; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid var(--primary); padding-bottom: 15px; }}
        h2 {{ color: #555; margin-top: 40px; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        h3 {{ color: #666; margin-top: 25px; }}
        h4 {{ color: #333; margin: 10px 0; }}
        .toc {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .toc h3 {{ margin-top: 0; }}
        .toc ul {{ list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 15px; }}
        .toc a {{ color: var(--primary); text-decoration: none; padding: 8px 15px; background: white; border-radius: 4px; display: inline-block; }}
        .toc a:hover {{ background: #fce4ec; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin: 25px 0; }}
        .metric {{ background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%); border: 1px solid #ddd; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2.5em; font-weight: bold; color: var(--primary); }}
        .metric-label {{ font-size: 0.85em; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
        .chart {{ margin: 25px 0; padding: 25px; background: #fafafa; border-radius: 8px; }}
        .chart h3 {{ margin-top: 0; color: #555; }}
        svg {{ max-width: 100%; height: auto; }}
        .turning-point {{ background: #fff3e0; border-left: 4px solid var(--warning); padding: 15px; margin: 15px 0; border-radius: 0 8px 8px 0; }}
        .turn {{ background: #fff; border-left: 4px solid #ccc; padding: 20px; margin: 20px 0; border-radius: 0 8px 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
        .turn-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #eee; }}
        .turn-header h4 {{ margin: 0; }}
        .turn-meta {{ font-size: 0.85em; color: #666; }}
        .turn-meta code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
        .turn-content {{ color: #333; white-space: pre-wrap; line-height: 1.8; }}
        .evidence {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 15px; }}
        .evidence h5 {{ margin: 0 0 10px 0; color: #555; }}
        .evidence-item {{ padding: 12px; margin: 8px 0; background: white; border-radius: 4px; border-left: 3px solid var(--success); }}
        .evidence-item .type {{ font-weight: bold; color: var(--success); text-transform: uppercase; font-size: 0.8em; }}
        .evidence-item .content {{ margin: 8px 0; }}
        .evidence-item .meta {{ font-size: 0.85em; color: #666; }}
        .causal-links {{ background: #fff3e0; padding: 15px; border-radius: 8px; margin-top: 15px; }}
        .causal-link {{ padding: 8px; margin: 5px 0; background: white; border-radius: 4px; display: flex; align-items: center; gap: 10px; }}
        .causal-link .cause {{ color: #1976D2; font-weight: 500; }}
        .causal-link .arrow {{ color: #666; }}
        .causal-link .effect {{ color: #388E3C; font-weight: 500; }}
        .rebuts-supports {{ margin-top: 12px; font-size: 0.9em; }}
        .rebuts-supports span {{ display: inline-block; padding: 4px 10px; border-radius: 4px; margin: 3px; }}
        .rebuts span {{ background: #ffcdd2; color: #c62828; }}
        .supports span {{ background: #c8e6c9; color: #2e7d32; }}
        .evaluation {{ background: #e3f2fd; padding: 15px; border-radius: 8px; margin-top: 15px; }}
        .evaluation summary {{ cursor: pointer; font-weight: bold; padding: 5px; }}
        .evaluation-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin: 15px 0; }}
        .eval-metric {{ background: white; padding: 12px; border-radius: 4px; text-align: center; }}
        .eval-metric .value {{ font-size: 1.4em; font-weight: bold; color: var(--primary); }}
        .eval-metric .label {{ font-size: 0.75em; color: #666; }}
        .safety-check {{ padding: 12px 15px; border-radius: 4px; margin: 10px 0; display: flex; align-items: center; gap: 12px; }}
        .safety-check.warning {{ background: #fff3e0; border-left: 4px solid var(--warning); }}
        .safety-check.failed {{ background: #ffebee; border-left: 4px solid var(--danger); }}
        .safety-icon {{ font-size: 1.3em; }}
        .verdict {{ background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); padding: 30px; border-radius: 12px; margin: 30px 0; }}
        .verdict h2 {{ margin-top: 0; color: #1b5e20; }}
        .verdict-decision {{ font-size: 1.8em; font-weight: bold; color: #1b5e20; }}
        .verdict-confidence {{ color: #388E3C; font-size: 1.1em; margin: 10px 0; }}
        .verdict-reasoning {{ background: white; padding: 20px; border-radius: 8px; margin-top: 20px; }}
        .scores {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
        .score-card {{ background: white; padding: 25px; border-radius: 8px; text-align: center; flex: 1; min-width: 140px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .score-value {{ font-size: 2.5em; font-weight: bold; color: var(--primary); }}
        .score-label {{ color: #666; font-size: 1em; margin-top: 8px; }}
        details[open] summary {{ margin-bottom: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Debate Analytics Report</h1>
        <p><strong>Topic:</strong> {html_escape.escape(analytics.topic or 'N/A')}</p>
        <p><strong>Debate ID:</strong> <code>{analytics.debate_id}</code></p>
        <p><strong>Agents:</strong> {", ".join(analytics.agents)}</p>
        <p><strong>Rounds:</strong> {analytics.rounds}</p>

        <div class="toc">
            <h3>Contents</h3>
            <ul>
                <li><a href="#metrics">Key Metrics</a></li>
                <li><a href="#charts">Visualizations</a></li>
                <li><a href="#turning-points">Turning Points</a></li>
                <li><a href="#transcript">Full Transcript</a></li>
                <li><a href="#verdict">Verdict</a></li>
            </ul>
        </div>

        <h2 id="metrics">Key Metrics</h2>
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
            <div class="metric">
                <div class="metric-value">{analytics.rounds}</div>
                <div class="metric-label">Rounds</div>
            </div>
        </div>

        <h2 id="charts">Visualizations</h2>
        {"".join(charts_html)}

        {"<h2 id='turning-points'>Turning Points</h2>" if analytics.turning_points else ""}
        {"".join(f'<div class="turning-point"><strong>Round {tp.round}</strong> ({tp.agent}): {tp.analysis}</div>' for tp in analytics.turning_points)}

        <h2 id="transcript">Full Transcript</h2>
        {"".join(transcript_html)}

        <div id="verdict">
        {verdict_html}
        </div>
    </div>
</body>
</html>"""
        return html

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
    include_transcript: bool = True,
) -> None:
    """Export debate with analytics to comprehensive HTML file.

    Args:
        result: DebateResult from a completed debate
        output_path: Path to write the HTML report
        include_charts: Whether to include visualizations
        include_transcript: Whether to include full transcript (default True)
    """
    analyzer = DebateAnalyzer(
        transcript=result.transcript,
        agents=result.metadata.agents,
        debate_id=result.debate_id,
        topic=result.topic,
    )
    analytics = analyzer.analyze()

    if include_transcript:
        # Generate comprehensive report with full transcript
        html = analyzer.generate_comprehensive_report(analytics, result)
    elif include_charts:
        html = analyzer.generate_report_html(analytics)
    else:
        html = analyzer.generate_report_html(analytics, include_charts=[])

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
