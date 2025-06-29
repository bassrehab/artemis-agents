# Audit Log Export

ARTEMIS provides a comprehensive audit log system for exporting debate records in multiple formats.

## Overview

The `AuditLog` class captures every aspect of a debate:

- Debate metadata (topic, agents, timestamps)
- Complete argument transcript
- Evaluation scores for each turn
- Safety alerts and interventions
- Final verdict with reasoning

## Basic Usage

```python
from artemis.core.debate import Debate
from artemis.core.agent import Agent
from artemis.utils.audit import AuditLog

# Run a debate
agents = [
    Agent(name="pro", role="Advocate", model="gpt-4o"),
    Agent(name="con", role="Opponent", model="gpt-4o"),
]

debate = Debate(topic="Should we adopt renewable energy?", agents=agents)
debate.assign_positions({
    "pro": "supports renewable energy adoption",
    "con": "advocates for traditional energy sources",
})

result = await debate.run(rounds=3)

# Generate audit log
audit = AuditLog.from_debate_result(result)
```

## Export Formats

### JSON Export

Full structured data suitable for programmatic processing:

```python
# Export to file
audit.to_json("audit_output.json")

# Get as string
json_string = audit.to_json()
print(json_string)
```

**JSON Structure:**

```json
{
  "debate_id": "debate_abc123",
  "topic": "Should we adopt renewable energy?",
  "started_at": "2025-06-28T14:30:00Z",
  "ended_at": "2025-06-28T14:35:00Z",
  "agents": ["pro", "con"],
  "entries": [
    {
      "timestamp": "2025-06-28T14:30:05Z",
      "event_type": "argument_generated",
      "agent": "pro",
      "round": 1,
      "details": {
        "level": "strategic",
        "content": "Renewable energy is essential...",
        "evidence_count": 2
      }
    }
  ],
  "verdict": {
    "decision": "pro",
    "confidence": 0.85,
    "reasoning": "..."
  }
}
```

### Markdown Export

Human-readable report format:

```python
# Export to file
audit.to_markdown("audit_output.md")

# Get as string
md_string = audit.to_markdown()
```

**Markdown Structure:**

```markdown
# Debate Audit Log

## Metadata

| Field | Value |
|-------|-------|
| Topic | Should we adopt renewable energy? |
| Started | 2025-06-28 14:30:00 |
| Duration | 5m 32s |

## Agents

- **pro**: Advocate (gpt-4o)
- **con**: Opponent (gpt-4o)

## Transcript

### Round 1

**pro** (Strategic):
> Renewable energy is essential for our future...

*Evidence*: 2 sources cited
*Evaluation*: 0.82

---

**con** (Strategic):
> While renewable energy has merits...

*Evidence*: 1 source cited
*Evaluation*: 0.78

## Verdict

**Winner**: pro
**Confidence**: 85%

The jury determined that the pro side presented...

## Safety Alerts

No safety alerts recorded.
```

### HTML Export

Styled report with interactive elements:

```python
# Export to file
audit.to_html("audit_output.html")

# Get as string
html_string = audit.to_html()
```

The HTML export includes:

- Styled headers and sections
- Color-coded severity levels for safety alerts
- Collapsible transcript sections
- Score visualizations
- Responsive layout

## Working with Entries

### Audit Entry Types

| Event Type | Description |
|------------|-------------|
| `argument_generated` | An agent produced an argument |
| `argument_evaluated` | An argument was scored |
| `safety_alert` | A safety monitor raised an alert |
| `round_completed` | A debate round finished |
| `verdict_reached` | Final verdict was determined |

### Filtering Entries

```python
# Get all safety alerts
alerts = [e for e in audit.entries if e.event_type == "safety_alert"]

# Get entries for a specific agent
pro_entries = [e for e in audit.entries if e.agent == "pro"]

# Get entries by round
round_2 = [e for e in audit.entries if e.round == 2]
```

## Integration with Safety Monitors

When safety monitors are active, their alerts are captured in the audit log:

```python
from artemis.safety import SandbagDetector, DeceptionMonitor

debate = Debate(
    topic="Your topic",
    agents=agents,
    safety_monitors=[
        SandbagDetector(sensitivity=0.7).process,
        DeceptionMonitor(sensitivity=0.6).process,
    ],
)

result = await debate.run()
audit = AuditLog.from_debate_result(result)

# Check for safety alerts in the audit
for entry in audit.entries:
    if entry.event_type == "safety_alert":
        print(f"Alert: {entry.details['alert_type']}")
        print(f"Severity: {entry.details['severity']}")
        print(f"Message: {entry.details['message']}")
```

## Complete Example

```python
import asyncio
from pathlib import Path

from artemis.core.debate import Debate
from artemis.core.agent import Agent
from artemis.core.jury import JuryPanel
from artemis.utils.audit import AuditLog

async def run_and_export():
    # Create agents with different models
    agents = [
        Agent(name="pro", role="Advocate", model="gpt-4o"),
        Agent(name="con", role="Opponent", model="claude-sonnet-4-20250514"),
    ]

    # Create jury with multiple perspectives
    jury = JuryPanel(
        evaluators=3,
        models=["gpt-4o", "gemini-2.0-flash", "claude-sonnet-4-20250514"],
    )

    # Run debate
    debate = Debate(
        topic="Should AI systems have legal personhood?",
        agents=agents,
        jury=jury,
    )
    debate.assign_positions({
        "pro": "supports AI legal personhood",
        "con": "opposes AI legal personhood",
    })

    result = await debate.run(rounds=2)

    # Generate audit log
    audit = AuditLog.from_debate_result(result)

    # Export all formats
    output_dir = Path("audit_output")
    output_dir.mkdir(exist_ok=True)

    audit.to_json(output_dir / "debate_audit.json")
    audit.to_markdown(output_dir / "debate_audit.md")
    audit.to_html(output_dir / "debate_audit.html")

    print(f"Audit logs exported to {output_dir}/")
    print(f"Verdict: {result.verdict.decision} ({result.verdict.confidence:.0%})")

if __name__ == "__main__":
    asyncio.run(run_and_export())
```

## Next Steps

- Learn about [Safety Monitors](safety-monitors.md)
- Explore [Multi-Agent Debates](multi-agent.md)
- See [Jury Configuration](custom-jury.md)
