# Quick Start

This guide will help you run your first ARTEMIS debate in minutes.

## Basic Debate

The simplest way to use ARTEMIS is to create agents and run a debate:

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate

async def run_debate():
    # Create two agents with opposing positions
    pro_agent = Agent(
        name="proponent",
        role="Advocate arguing in favor of the proposition",
        model="gpt-4o",
    )

    con_agent = Agent(
        name="opponent",
        role="Advocate arguing against the proposition",
        model="gpt-4o",
    )

    # Create and run the debate
    debate = Debate(
        topic="Should AI development be regulated by governments?",
        agents=[pro_agent, con_agent],
        rounds=3,
    )

    # Assign positions
    debate.assign_positions({
        "proponent": "supports government regulation of AI",
        "opponent": "opposes government regulation of AI",
    })

    # Run the debate
    result = await debate.run()

    # Print results
    print(f"Topic: {result.topic}")
    print(f"Verdict: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")
    print(f"Reasoning: {result.verdict.reasoning}")

# Run it
asyncio.run(run_debate())
```

## With Safety Monitoring

Add safety monitors to detect problematic behavior:

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.safety import SandbagDetector, DeceptionMonitor, MonitorMode

async def run_safe_debate():
    # Create agents
    agents = [
        Agent(
            name="pro",
            role="Advocate arguing for the proposition",
            model="gpt-4o",
        ),
        Agent(
            name="con",
            role="Advocate arguing against the proposition",
            model="gpt-4o",
        ),
    ]

    # Create safety monitors
    sandbag_detector = SandbagDetector(
        mode=MonitorMode.PASSIVE,
        sensitivity=0.7,
    )
    deception_monitor = DeceptionMonitor(
        mode=MonitorMode.PASSIVE,
        sensitivity=0.6,
    )

    # Create debate with safety monitors
    debate = Debate(
        topic="Should cryptocurrency replace traditional banking?",
        agents=agents,
        rounds=3,
        safety_monitors=[sandbag_detector.process, deception_monitor.process],
    )

    debate.assign_positions({
        "pro": "supports cryptocurrency adoption",
        "con": "supports traditional banking",
    })

    # Run the debate
    result = await debate.run()

    # Check for safety alerts
    if result.safety_alerts:
        print("Safety alerts detected:")
        for alert in result.safety_alerts:
            print(f"  {alert.type}: {alert.severity:.0%} severity")

    # Check results in transcript
    for turn in result.transcript:
        print(f"Round {turn.round} - {turn.agent}")
        if turn.evaluation:
            print(f"  Score: {turn.evaluation.total_score:.1f}")

asyncio.run(run_safe_debate())
```

## With Reasoning Models

Use reasoning models (o1, DeepSeek R1) for deeper analysis:

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.types import ReasoningConfig

async def run_reasoning_debate():
    # Create reasoning config for extended thinking
    reasoning_config = ReasoningConfig(
        enabled=True,
        thinking_budget=8000,
        include_trace_in_output=False,
    )

    # Create agents with reasoning models
    pro_agent = Agent(
        name="deep_thinker_pro",
        role="Philosopher arguing consciousness is computable",
        model="deepseek-reasoner",  # DeepSeek R1
        reasoning=reasoning_config,
    )

    con_agent = Agent(
        name="deep_thinker_con",
        role="Philosopher arguing consciousness is not computable",
        model="deepseek-reasoner",
        reasoning=reasoning_config,
    )

    debate = Debate(
        topic="Is consciousness computable?",
        agents=[pro_agent, con_agent],
        rounds=2,  # Fewer rounds with deeper thinking
    )

    debate.assign_positions({
        "deep_thinker_pro": "consciousness can be computed",
        "deep_thinker_con": "consciousness cannot be computed",
    })

    result = await debate.run()

    print(f"Verdict: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")

asyncio.run(run_reasoning_debate())
```

## Using the MCP Server

Start ARTEMIS as an MCP server for integration with other tools:

```bash
# Start the server
artemis-mcp --port 8080
```

Then connect from any MCP client:

```python
# Example: Using the MCP server programmatically
from artemis.mcp import ArtemisMCPServer

async def use_mcp():
    server = ArtemisMCPServer(default_model="gpt-4o")

    # Start a debate via tool call
    result = await server.handle_tool_call(
        "artemis_debate_start",
        {
            "topic": "Should remote work be the default?",
            "rounds": 3,
        }
    )

    debate_id = result["debate_id"]
    print(f"Started debate: {debate_id}")

    # Get verdict
    verdict = await server.handle_tool_call(
        "artemis_get_verdict",
        {"debate_id": debate_id}
    )

    print(f"Verdict: {verdict['verdict']}")
```

## Framework Integration Examples

### LangChain

```python
from artemis.integrations.langchain import ArtemisDebateTool

# Create the tool
debate_tool = ArtemisDebateTool(model="gpt-4o")

# Use as a LangChain tool - returns formatted string result
result = debate_tool.run(
    topic="Should we adopt microservices?",
    rounds=2,
)

print(result)  # Formatted debate analysis string
```

### LangGraph

```python
from artemis.integrations.langgraph import create_debate_workflow

# Create a complete debate workflow
workflow = create_debate_workflow(model="gpt-4o")

# Run it
result = await workflow.ainvoke({
    "topic": "Is functional programming better than OOP?",
    "rounds": 2,
})

print(f"Verdict: {result['verdict']['decision']}")
```

### CrewAI

```python
from artemis.integrations.crewai import ArtemisCrewTool

# Create the tool
crew_tool = ArtemisCrewTool(model="gpt-4o")

# Run a debate - returns formatted string
result = crew_tool.run(
    topic="Should we use NoSQL or SQL for this project?",
    rounds=2,
)

print(result)  # Formatted debate analysis string
```

## Next Steps

- Learn about [Core Concepts](../concepts/overview.md)
- Explore [Safety Monitoring](../safety/overview.md)
- See more [Examples](../examples/basic-debate.md)
- Read the [API Reference](../api/core.md)
