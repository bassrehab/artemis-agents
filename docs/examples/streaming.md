# Working with Debate Results

This example demonstrates how to work with ARTEMIS debate results and implement progress tracking patterns.

> **Note**: Real-time streaming of debate turns is a planned feature. Currently, debates run asynchronously and return complete results. The patterns below show how to work effectively with the async API.

## Basic Async Execution

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate

async def run_basic_debate():
    agents = [
        Agent(name="pro", role="Proposition advocate", model="gpt-4o"),
        Agent(name="con", role="Opposition advocate", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Should companies require return-to-office?",
        agents=agents,
        rounds=3,
    )

    debate.assign_positions({
        "pro": "supports return-to-office policies",
        "con": "supports remote work flexibility",
    })

    print("DEBATE: Return to Office")
    print("=" * 60)
    print("Running debate... (this may take a moment)")

    # Run the debate - returns complete result
    result = await debate.run()

    # Process results after completion
    print(f"\nCompleted {len(result.transcript)} turns")

    for turn in result.transcript:
        print(f"\n[Round {turn.round}] {turn.agent.upper()}")
        print("-" * 40)
        print(turn.argument.content[:300] + "..." if len(turn.argument.content) > 300 else turn.argument.content)

        if turn.evaluation:
            print(f"\n  Score: {turn.evaluation.total_score:.1f}/10")

    print("\n" + "=" * 60)
    print(f"VERDICT: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")
    print(f"\n{result.verdict.reasoning}")

asyncio.run(run_basic_debate())
```

## Running Multiple Debates Concurrently

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate

async def run_single_debate(topic: str, debate_id: int) -> dict:
    """Run a single debate and return results."""
    agents = [
        Agent(name="pro", role="Advocate", model="gpt-4o"),
        Agent(name="con", role="Critic", model="gpt-4o"),
    ]

    debate = Debate(topic=topic, agents=agents, rounds=2)
    debate.assign_positions({
        "pro": "supports the proposition",
        "con": "opposes the proposition",
    })

    print(f"[Debate {debate_id}] Starting: {topic[:40]}...")
    result = await debate.run()
    print(f"[Debate {debate_id}] Completed: {result.verdict.decision}")

    return {
        "id": debate_id,
        "topic": topic,
        "verdict": result.verdict.decision,
        "confidence": result.verdict.confidence,
    }

async def run_parallel_debates():
    """Run multiple debates concurrently."""
    topics = [
        "Should AI development be open-sourced?",
        "Is blockchain technology overhyped?",
        "Should programming be taught in elementary schools?",
    ]

    print("Starting parallel debates...")
    print("=" * 60)

    # Run all debates concurrently
    tasks = [
        run_single_debate(topic, i)
        for i, topic in enumerate(topics, 1)
    ]

    results = await asyncio.gather(*tasks)

    print("\n" + "=" * 60)
    print("ALL RESULTS:")
    for r in results:
        print(f"\nDebate {r['id']}: {r['topic'][:40]}...")
        print(f"  Verdict: {r['verdict']} ({r['confidence']:.0%} confidence)")

asyncio.run(run_parallel_debates())
```

## Progress Tracking with Task Wrapper

```python
import asyncio
import time
from artemis.core.agent import Agent
from artemis.core.debate import Debate

class DebateRunner:
    """Wrapper for running debates with progress tracking."""

    def __init__(self, topic: str, agents: list[Agent], rounds: int = 3):
        self.topic = topic
        self.debate = Debate(topic=topic, agents=agents, rounds=rounds)
        self.start_time = None
        self.end_time = None

    async def run_with_progress(self, positions: dict[str, str]) -> dict:
        """Run debate and track timing."""
        self.debate.assign_positions(positions)

        self.start_time = time.time()
        print(f"Starting debate on: {self.topic[:50]}...")

        result = await self.debate.run()

        self.end_time = time.time()
        duration = self.end_time - self.start_time

        return {
            "result": result,
            "duration_seconds": duration,
            "turns": len(result.transcript),
            "turns_per_second": len(result.transcript) / duration if duration > 0 else 0,
        }

async def main():
    agents = [
        Agent(name="optimist", role="Technology optimist", model="gpt-4o"),
        Agent(name="realist", role="Pragmatic realist", model="gpt-4o"),
    ]

    runner = DebateRunner(
        topic="Will AGI be achieved within 10 years?",
        agents=agents,
        rounds=2,
    )

    stats = await runner.run_with_progress({
        "optimist": "believes AGI will be achieved within 10 years",
        "realist": "believes AGI is further away than optimists think",
    })

    print(f"\nDebate completed in {stats['duration_seconds']:.1f}s")
    print(f"Total turns: {stats['turns']}")
    print(f"Verdict: {stats['result'].verdict.decision}")

asyncio.run(main())
```

## Processing Results with Rich Console

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate

# Install: pip install rich
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

async def run_rich_debate():
    agents = [
        Agent(name="pro", role="Cryptocurrency advocate", model="gpt-4o"),
        Agent(name="con", role="Traditional finance advocate", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Is cryptocurrency the future of finance?",
        agents=agents,
        rounds=2,
    )

    debate.assign_positions({
        "pro": "argues cryptocurrency will transform finance",
        "con": "argues traditional finance will remain dominant",
    })

    console.print(Panel.fit(
        "[bold blue]ARTEMIS Debate[/bold blue]\n"
        f"Topic: {debate.topic}",
        title="Starting Debate"
    ))

    with console.status("[cyan]Running debate...[/cyan]"):
        result = await debate.run()

    # Display results with rich formatting
    for turn in result.transcript:
        agent_color = "green" if turn.agent == "pro" else "red"
        panel = Panel(
            turn.argument.content,
            title=f"[bold {agent_color}]Round {turn.round} - {turn.agent.upper()}[/bold {agent_color}]",
            subtitle=f"Level: {turn.argument.level.value}",
            border_style=agent_color,
        )
        console.print(panel)

        if turn.argument.evidence:
            table = Table(title="Evidence", show_header=True)
            table.add_column("Type")
            table.add_column("Source")
            for e in turn.argument.evidence[:3]:
                table.add_row(e.type, e.source)
            console.print(table)

    # Verdict
    verdict = result.verdict
    decision_color = "green" if "pro" in verdict.decision else "red" if "con" in verdict.decision else "yellow"

    console.print(Panel(
        f"[bold {decision_color}]{verdict.decision.upper()}[/bold {decision_color}]\n\n"
        f"Confidence: {verdict.confidence:.0%}\n\n"
        f"{verdict.reasoning}",
        title="Final Verdict",
        border_style=decision_color,
    ))

asyncio.run(run_rich_debate())
```

## LangGraph Integration for Step-by-Step Execution

For step-by-step execution with intermediate state access, use the LangGraph integration:

```python
import asyncio
from artemis.integrations.langgraph import create_debate_workflow

async def run_stepwise_debate():
    # Create step-by-step workflow
    workflow = create_debate_workflow(
        model="gpt-4o",
        step_by_step=True,
    )

    initial_state = {
        "topic": "Should we adopt microservices architecture?",
        "agents": [
            {"name": "architect", "role": "System Architect", "position": "pro microservices"},
            {"name": "pragmatist", "role": "Senior Dev", "position": "pro monolith"},
        ],
        "rounds": 2,
    }

    print("Starting stepwise debate...")

    # In LangGraph, you can checkpoint and observe state between steps
    result = await workflow.ainvoke(initial_state)

    print(f"\nPhase: {result['phase']}")
    print(f"Verdict: {result['verdict']['decision']}")
    print(f"Confidence: {result['verdict']['confidence']:.0%}")

asyncio.run(run_stepwise_debate())
```

## Sending Results to WebSocket

```python
import asyncio
import json
from artemis.core.agent import Agent
from artemis.core.debate import Debate

async def run_debate_for_websocket(topic: str) -> dict:
    """Run debate and format for WebSocket transmission."""
    agents = [
        Agent(name="pro", role="Advocate", model="gpt-4o"),
        Agent(name="con", role="Critic", model="gpt-4o"),
    ]

    debate = Debate(topic=topic, agents=agents, rounds=2)
    debate.assign_positions({
        "pro": "supports the proposition",
        "con": "opposes the proposition",
    })

    result = await debate.run()

    # Format for JSON transmission
    return {
        "type": "debate_complete",
        "topic": topic,
        "transcript": [
            {
                "round": turn.round,
                "agent": turn.agent,
                "level": turn.argument.level.value,
                "content": turn.argument.content,
                "score": turn.evaluation.total_score if turn.evaluation else None,
            }
            for turn in result.transcript
        ],
        "verdict": {
            "decision": result.verdict.decision,
            "confidence": result.verdict.confidence,
            "reasoning": result.verdict.reasoning,
        },
    }

# Example FastAPI endpoint
"""
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/debate")
async def debate_endpoint(websocket: WebSocket):
    await websocket.accept()

    data = await websocket.receive_json()
    topic = data.get("topic", "Default topic")

    # Send start message
    await websocket.send_json({"type": "debate_started", "topic": topic})

    # Run debate
    result = await run_debate_for_websocket(topic)

    # Send complete result
    await websocket.send_json(result)
"""
```

## Buffering Results for Late Consumers

```python
import asyncio
from collections import deque
from artemis.core.agent import Agent
from artemis.core.debate import Debate

class DebateResultBuffer:
    """Buffer debate results for consumers that join late."""

    def __init__(self, max_size: int = 100):
        self.buffer = deque(maxlen=max_size)
        self.consumers = []

    def add_result(self, result: dict):
        """Add a result to the buffer."""
        self.buffer.append(result)

        # Notify all consumers
        for consumer in self.consumers:
            try:
                consumer(result)
            except Exception as e:
                print(f"Consumer error: {e}")

    def add_consumer(self, callback):
        """Add a consumer callback."""
        self.consumers.append(callback)

    def get_all(self) -> list[dict]:
        """Get all buffered results."""
        return list(self.buffer)

async def main():
    buffer = DebateResultBuffer()

    # Add a simple consumer
    def print_consumer(result):
        print(f"New result: {result['verdict']}")

    buffer.add_consumer(print_consumer)

    # Run a debate
    agents = [
        Agent(name="pro", role="Advocate", model="gpt-4o"),
        Agent(name="con", role="Critic", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Is remote work more productive?",
        agents=agents,
        rounds=2,
    )
    debate.assign_positions({
        "pro": "argues remote work is more productive",
        "con": "argues office work is more productive",
    })

    result = await debate.run()

    # Add to buffer
    buffer.add_result({
        "topic": debate.topic,
        "verdict": result.verdict.decision,
        "confidence": result.verdict.confidence,
    })

    # Late consumer can get buffered results
    print("\nBuffered results:")
    for r in buffer.get_all():
        print(f"  {r['topic'][:30]}... -> {r['verdict']}")

asyncio.run(main())
```

## Next Steps

- See [Basic Debate](basic-debate.md) for fundamentals
- Explore [LangGraph Workflow](langgraph-workflow.md) for step-by-step execution
- Add [Safety Monitors](safety-monitors.md) to track debate safety
