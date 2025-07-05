# Debate Analytics

ARTEMIS provides comprehensive analytics for analyzing debate dynamics, tracking momentum shifts, and visualizing results.

## Quick Start

```python
from artemis import Debate, Agent
from artemis.analytics import analyze_debate, export_analytics_report

# Run a debate
debate = Debate(
    topic="Should AI development be regulated?",
    agents=[
        Agent(name="Advocate", role="Argues for regulation"),
        Agent(name="Skeptic", role="Argues against regulation"),
    ],
    rounds=3,
)
result = await debate.run()

# Compute analytics
analytics = analyze_debate(result)

# Export HTML report with visualizations
export_analytics_report(result, "debate_report.html")
```

## Analytics Features

### Momentum Tracking

Track how each agent's performance evolves over the debate:

```python
from artemis.analytics import DebateAnalyzer

analyzer = DebateAnalyzer(
    transcript=result.transcript,
    agents=["Advocate", "Skeptic"],
    topic="AI Regulation",
)
analytics = analyzer.analyze()

# Momentum history shows score changes per round
for point in analytics.momentum_history:
    print(f"Round {point.round}: {point.agent} - "
          f"Score: {point.score:.2f}, Momentum: {point.momentum:+.2f}")
```

### Turning Points

Detect key moments where the debate dynamics shifted:

```python
for tp in analytics.turning_points:
    print(f"Round {tp.round}: {tp.agent}")
    print(f"  Significance: {tp.significance:.2f}")
    print(f"  {tp.analysis}")
```

### Metrics

Compute detailed metrics for each agent:

```python
from artemis.analytics.metrics import DebateMetricsCalculator

calc = DebateMetricsCalculator(result.transcript, agents)

# Rebuttal effectiveness - how well agents counter opponents
print(calc.rebuttal_effectiveness)
# {'Advocate': 0.72, 'Skeptic': 0.65}

# Evidence utilization - use of supporting evidence
print(calc.evidence_utilization)
# {'Advocate': 0.85, 'Skeptic': 0.78}

# Argument diversity - variety of argument levels used
print(calc.argument_diversity_index)
# {'Advocate': 0.92, 'Skeptic': 0.71}
```

### Round-by-Round Analysis

Get detailed metrics for each round:

```python
for rm in calc.get_all_round_metrics():
    print(f"Round {rm.round}:")
    print(f"  Scores: {rm.agent_scores}")
    print(f"  Score Changes: {rm.score_delta}")
```

## Visualizations

ARTEMIS generates pure SVG visualizations with no JavaScript dependencies.

### Score Progression Chart

Shows how scores evolve over rounds with turning point highlights:

```python
from artemis.analytics.visualizations import ScoreProgressionChart

chart = ScoreProgressionChart(width=800, height=400)
svg = chart.render(
    round_scores=[
        {"Advocate": 0.65, "Skeptic": 0.66},
        {"Advocate": 0.69, "Skeptic": 0.64},
        {"Advocate": 0.65, "Skeptic": 0.67},
    ],
    agents=["Advocate", "Skeptic"],
    highlight_turning_points=[2],  # Highlight round 2
)
```

### Momentum Chart

Visualizes momentum with positive/negative indicators:

```python
from artemis.analytics.visualizations import MomentumChart

chart = MomentumChart()
svg = chart.render(analytics.momentum_history, agents=["Advocate", "Skeptic"])
```

### Jury Vote Chart

Display final scores as bars or pie chart:

```python
from artemis.analytics.visualizations import JuryVoteChart

chart = JuryVoteChart()

# Bar chart
bar_svg = chart.render_bar({"Advocate": 0.61, "Skeptic": 0.59})

# Pie chart
pie_svg = chart.render_pie({"Advocate": 0.61, "Skeptic": 0.59})
```

### Argument Flow Diagram

Shows the flow of arguments and rebuttals:

```python
from artemis.analytics.visualizations import ArgumentFlowDiagram

chart = ArgumentFlowDiagram(width=800, height=600)
svg = chart.render(result.transcript, agents=["Advocate", "Skeptic"])
```

### Topic Coverage Heatmap

Visualizes which topics each agent covered:

```python
from artemis.analytics.visualizations import TopicCoverageHeatmap

coverage = {
    "Advocate": {"safety": 5, "innovation": 3, "economics": 2},
    "Skeptic": {"freedom": 4, "innovation": 6, "economics": 3},
}
chart = TopicCoverageHeatmap()
svg = chart.render(coverage, agents=["Advocate", "Skeptic"])
```

## HTML Report Generation

Generate a complete HTML report with all visualizations:

```python
from artemis.analytics import export_analytics_report

# Export with default settings
export_analytics_report(result, "report.html")

# Customize
export_analytics_report(
    result,
    "report.html",
    include_charts=True,  # Include SVG visualizations
)
```

The generated report includes:
- Debate metadata (topic, agents, rounds)
- Key metrics summary (turning points, lead changes, sway events)
- Score progression chart
- Momentum over time chart
- Final scores bar chart
- Turning point analysis

## Using with Debate.get_analytics()

You can also compute analytics directly from a Debate instance:

```python
debate = Debate(topic="...", agents=[...], rounds=3)
result = await debate.run()

# Get analytics directly
analytics = debate.get_analytics()

print(f"Turning points: {len(analytics.turning_points)}")
print(f"Lead changes: {analytics.count_lead_changes()}")
print(f"Final momentum: {analytics.final_momentum}")
```

## Analytics Data Structures

### DebateAnalytics

The main analytics container:

```python
class DebateAnalytics(BaseModel):
    debate_id: str
    topic: str
    agents: list[str]
    rounds: int
    momentum_history: list[MomentumPoint]
    turning_points: list[TurningPoint]
    round_metrics: list[RoundMetrics]
    final_momentum: dict[str, float]
    sway_events: list[SwayEvent]

    # Computed properties
    def get_leader_per_round(self) -> list[str | None]: ...
    def count_lead_changes(self) -> int: ...
```

### MomentumPoint

Tracks momentum for each agent per round:

```python
class MomentumPoint(BaseModel):
    round: int
    agent: str
    score: float           # Raw score (0-1)
    momentum: float        # Rate of change (-1 to +1)
    cumulative_advantage: float  # Running advantage
```

### TurningPoint

Represents a significant shift in debate dynamics:

```python
class TurningPoint(BaseModel):
    round: int
    turn_id: str
    agent: str
    before_momentum: dict[str, float]
    after_momentum: dict[str, float]
    significance: float    # 0-1, how significant the shift was
    analysis: str          # Human-readable explanation
```

## Configuration

### MomentumTracker Options

```python
from artemis.analytics.momentum import MomentumTracker

tracker = MomentumTracker(
    smoothing_window=2,        # Rounds to smooth over
    turning_point_threshold=0.3,  # Momentum change threshold
)
```

### Chart Customization

All charts support customization:

```python
from artemis.analytics.visualizations import ScoreProgressionChart

chart = ScoreProgressionChart(
    width=800,      # SVG width
    height=400,     # SVG height
    padding=50,     # Padding around chart area
    colors={        # Custom color scheme
        "agent_0": "#2196F3",
        "agent_1": "#FF9800",
        "pro": "#4CAF50",
        "con": "#F44336",
    }
)
```
