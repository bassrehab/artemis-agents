# Causal Graph Analysis

ARTEMIS provides advanced causal graph analysis capabilities for detecting reasoning issues, analyzing argument strength, and providing strategic debate support.

## Overview

The CausalGraph v2 features extend the basic causal graph with:

- **Circular Reasoning Detection** - Identify logical loops
- **Argument Strength Analysis** - Score arguments by causal support
- **Weak Link Identification** - Find vulnerable edges
- **Contradiction Detection** - Find conflicting claims
- **Fallacy Detection** - Identify common reasoning fallacies
- **Strategic Analysis** - Attack and defense planning
- **Visualization** - Export to DOT, Mermaid, and HTML

## CausalAnalyzer

The `CausalAnalyzer` class provides comprehensive analysis on causal graphs.

```python
from artemis.core import CausalGraph, CausalAnalyzer
from artemis.core.types import CausalLink

# Build a graph
graph = CausalGraph()
graph.add_link(CausalLink(cause="AI", effect="Automation", strength=0.8))
graph.add_link(CausalLink(cause="Automation", effect="Job Loss", strength=0.5))
graph.add_link(CausalLink(cause="AI", effect="Productivity", strength=0.9))

# Analyze
analyzer = CausalAnalyzer(graph)
result = analyzer.analyze()

print(f"Has circular reasoning: {result.has_circular_reasoning}")
print(f"Overall coherence: {result.overall_coherence:.2f}")
print(f"Weak links found: {len(result.weak_links)}")
```

### Circular Reasoning Detection

Detects logical loops where conclusions support their own premises:

```python
cycles = analyzer.find_circular_reasoning()

for cycle in cycles:
    print(f"Cycle: {' -> '.join(cycle.cycle)}")
    print(f"Severity: {cycle.severity:.2f}")
    print(f"Arguments involved: {cycle.argument_ids}")
```

### Weak Link Analysis

Finds causal links below a strength threshold:

```python
weak_links = analyzer.find_weak_links(threshold=0.4)

for weak in weak_links:
    print(f"{weak.source} -> {weak.target}")
    print(f"  Strength: {weak.strength:.2f}")
    print(f"  Attack suggestions: {weak.attack_suggestions}")
```

### Contradiction Detection

Finds conflicting causal claims:

```python
contradictions = analyzer.find_contradictions()

for contradiction in contradictions:
    print(f"Claim A: {contradiction.claim_a_source} {contradiction.claim_a_type} {contradiction.claim_a_target}")
    print(f"Claim B: {contradiction.claim_b_source} {contradiction.claim_b_type} {contradiction.claim_b_target}")
    print(f"Severity: {contradiction.severity:.2f}")
```

### Argument Strength

Compute how well an argument is supported by the causal graph:

```python
strength = analyzer.compute_argument_strength("arg-123")

print(f"Overall score: {strength.overall_score:.2f}")
print(f"Causal support: {strength.causal_support:.2f}")
print(f"Evidence backing: {strength.evidence_backing:.2f}")
print(f"Vulnerability: {strength.vulnerability:.2f}")
```

### Critical Node Analysis

Find nodes that many paths depend on:

```python
critical = analyzer.find_critical_nodes()

for node in critical[:5]:
    print(f"{node.label}")
    print(f"  Centrality: {node.centrality_score:.2f}")
    print(f"  Impact if challenged: {node.impact_if_challenged:.2f}")
```

## Fallacy Detection

The `CausalFallacyDetector` identifies common reasoning fallacies.

```python
from artemis.core import CausalFallacyDetector
from artemis.core.types import Argument, ArgumentLevel

detector = CausalFallacyDetector()

argument = Argument(
    agent="test",
    level=ArgumentLevel.TACTICAL,
    content="After the policy change, crime dropped. This proves the policy works.",
    causal_links=[
        CausalLink(cause="policy", effect="crime_reduction", strength=0.3)
    ],
)

fallacies = detector.detect_fallacies(argument, graph)

for fallacy in fallacies:
    print(f"Type: {fallacy.fallacy_type.value}")
    print(f"Description: {fallacy.description}")
    print(f"Severity: {fallacy.severity:.2f}")
```

### Detected Fallacy Types

| Fallacy | Description |
|---------|-------------|
| `POST_HOC` | A occurred before B, therefore A caused B |
| `FALSE_CAUSE` | Treating correlation as causation |
| `SLIPPERY_SLOPE` | Unwarranted chain of consequences |
| `CIRCULAR_REASONING` | A because B because A |
| `APPEAL_TO_CONSEQUENCE` | True because of its consequences |

## Strategic Analysis

The `CausalStrategy` class provides strategic debate support.

```python
from artemis.core import CausalStrategy

# With opponent's graph
strategy = CausalStrategy(own_graph, opponent_graph)

# Map attack surface
targets = strategy.map_attack_surface()
for target in targets[:3]:
    print(f"Target: {target.source} -> {target.target}")
    print(f"  Vulnerability: {target.vulnerability_score:.2f}")
    print(f"  Strategies: {target.attack_strategies}")

# Get reinforcement suggestions
reinforcements = strategy.suggest_reinforcements()
for suggestion in reinforcements[:3]:
    print(f"Strengthen: {suggestion.source} -> {suggestion.target}")
    print(f"  Current: {suggestion.current_strength:.2f}")
    print(f"  Suggested evidence: {suggestion.suggested_evidence}")
```

### Attack Planning

```python
# Map opponent's weak points
targets = strategy.map_attack_surface()

# Suggest rebuttals for specific argument
rebuttals = strategy.suggest_rebuttals(opponent_argument)
```

### Defense Planning

```python
# Find own vulnerable claims
vulnerable = strategy.find_vulnerable_claims()

# Identify defensive priorities
priorities = strategy.identify_defensive_priorities()

# Predict what opponent might target
predictions = strategy.predict_opponent_targets()
```

### Opponent Modeling

```python
profile = strategy.analyze_opponent_strategy()

if profile:
    print(f"Primary focus: {profile.primary_focus}")
    print(f"Argument style: {profile.argument_style}")
    print(f"Weak points: {profile.weak_points}")
    print(f"Predicted moves: {profile.predicted_moves}")
```

## Visualization

The `CausalVisualizer` class exports graphs to various formats.

### DOT Format (Graphviz)

```python
from artemis.core import CausalVisualizer

visualizer = CausalVisualizer(graph)

dot = visualizer.to_dot(
    highlight_weak=True,
    weak_threshold=0.4,
    title="Debate Causal Graph"
)

# Save to file
with open("graph.dot", "w") as f:
    f.write(dot)

# Or render with graphviz
# dot -Tpng graph.dot -o graph.png
```

### Mermaid Diagrams

```python
mermaid = visualizer.to_mermaid(direction="TB")

# Embed in markdown
print(f"```mermaid\n{mermaid}\n```")
```

### JSON Export

```python
data = visualizer.to_json(include_analysis=True)

import json
with open("graph.json", "w") as f:
    json.dump(data, f, indent=2)
```

### HTML Report

```python
html = visualizer.generate_report(title="Debate Analysis")

with open("report.html", "w") as f:
    f.write(html)
```

The HTML report includes:
- Overview metrics
- Interactive Mermaid diagram
- Weak links table
- Contradictions table
- Critical nodes analysis
- Detected fallacies

## Graph Snapshots

Track graph evolution over the debate:

```python
from artemis.core import create_snapshot

snapshots = []

# After each turn
snapshot = create_snapshot(graph, round_num=1, turn_num=1)
snapshots.append(snapshot)

# Generate timeline
timeline = visualizer.generate_timeline(snapshots)
```

## New Graph Methods

CausalGraph v2 adds these methods:

```python
# Get all cycles (not just boolean)
cycles = graph.get_all_cycles()

# Find contradicting edges
contradictions = graph.get_contradicting_edges()

# Compute betweenness centrality
centrality = graph.compute_betweenness_centrality()

# Extract subgraph
subgraph = graph.get_subgraph(["node_a", "node_b", "node_c"])

# Get neighborhood
neighborhood = graph.get_neighborhood("central_node", depth=2)
```

## Integration with Evaluation

The causal analysis integrates with the L-AE-CR evaluation system:

```python
from artemis.core import AdaptiveEvaluator

evaluator = AdaptiveEvaluator()

# Evaluation automatically considers:
# - Causal chain strength
# - Circular reasoning detection
# - Fallacy patterns
# - Argument coherence
```

## Best Practices

1. **Build graphs incrementally** - Add links as arguments are made
2. **Track argument IDs** - Associate nodes and edges with arguments
3. **Monitor for cycles** - Check `has_circular_reasoning` after each turn
4. **Visualize regularly** - Generate reports for debugging
5. **Use strategic analysis** - Help agents improve their arguments

## Next Steps

- See [H-L-DAG Arguments](h-l-dag.md) for argument structure
- See [L-AE-CR Evaluation](l-ae-cr.md) for how causal analysis affects scoring
- See [API Reference](../api/core.md) for complete API documentation
