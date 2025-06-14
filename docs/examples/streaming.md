# Streaming Output

This example demonstrates how to stream debate progress in real-time for better user experience.

## Basic Streaming

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate

async def stream_basic_debate():
    agents = [
        Agent(name="pro", model="gpt-4o"),
        Agent(name="con", model="gpt-4o"),
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

    # Stream each turn as it happens
    async for turn in debate.stream():
        print(f"\n[Round {turn.round}] {turn.agent.upper()}")
        print("-" * 40)
        print(turn.argument.content)

        if turn.evaluation:
            print(f"\n  Score: {turn.evaluation.total_score:.1f}/10")

    # Get final verdict
    result = await debate.get_result()
    print("\n" + "=" * 60)
    print(f"VERDICT: {result.verdict.decision}")
    print(f"Confidence: {result.verdict.confidence:.0%}")
    print(f"\n{result.verdict.reasoning}")

asyncio.run(stream_basic_debate())
```

## Streaming with Progress Indicators

```python
import asyncio
import sys
from artemis.core.agent import Agent
from artemis.core.debate import Debate

async def stream_with_progress():
    agents = [
        Agent(name="advocate", model="gpt-4o"),
        Agent(name="critic", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Is blockchain technology overhyped?",
        agents=agents,
        rounds=3,
    )

    debate.assign_positions({
        "advocate": "argues blockchain is transformative",
        "critic": "argues blockchain is overhyped",
    })

    total_turns = len(agents) * 3  # agents * rounds
    current_turn = 0

    def print_progress(current: int, total: int, agent: str, status: str):
        bar_length = 30
        filled = int(bar_length * current / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        percent = current / total * 100
        sys.stdout.write(f"\r[{bar}] {percent:.0f}% - {agent}: {status}")
        sys.stdout.flush()

    print("Starting debate...\n")

    async for event in debate.stream_events():
        if event.type == "turn_start":
            current_turn += 1
            print_progress(current_turn, total_turns, event.agent, "thinking...")

        elif event.type == "turn_progress":
            print_progress(current_turn, total_turns, event.agent, "generating...")

        elif event.type == "turn_complete":
            print_progress(current_turn, total_turns, event.agent, "done ✓")
            print()  # New line after progress bar

            # Print the argument
            print(f"\n{event.agent.upper()}:")
            print(event.turn.argument.content[:300] + "...")

        elif event.type == "evaluation_complete":
            print(f"  → Score: {event.evaluation.total_score:.1f}/10")

        elif event.type == "round_complete":
            print(f"\n--- Round {event.round} Complete ---\n")

        elif event.type == "deliberation_start":
            print("\n🎓 Jury deliberating...")

        elif event.type == "verdict_complete":
            print(f"\n✅ Verdict: {event.verdict.decision}")

    result = await debate.get_result()
    print(f"\nFinal reasoning: {result.verdict.reasoning}")

asyncio.run(stream_with_progress())
```

## Streaming with Callbacks

```python
import asyncio
from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.types import Turn, Evaluation, Verdict

class DebateStreamHandler:
    """Handler for debate streaming events."""

    def __init__(self):
        self.turns = []
        self.start_time = None

    async def on_debate_start(self, topic: str, agents: list[str]):
        import time
        self.start_time = time.time()
        print(f"🎬 Debate starting: {topic}")
        print(f"   Participants: {', '.join(agents)}")

    async def on_round_start(self, round_num: int):
        print(f"\n📍 Round {round_num}")

    async def on_turn_start(self, agent: str, round_num: int):
        print(f"   {agent} is formulating argument...", end="", flush=True)

    async def on_turn_complete(self, turn: Turn):
        print(" done!")
        self.turns.append(turn)

        # Show preview of argument
        preview = turn.argument.content[:100].replace("\n", " ")
        print(f"   └─ \"{preview}...\"")

    async def on_evaluation_complete(self, agent: str, evaluation: Evaluation):
        scores = evaluation.scores
        print(f"   └─ Scores: Logic={scores.get('logical_coherence', 0):.1f}, "
              f"Evidence={scores.get('evidence_quality', 0):.1f}")

    async def on_deliberation_start(self):
        print("\n⚖️  Jury deliberating...")

    async def on_verdict(self, verdict: Verdict):
        import time
        duration = time.time() - self.start_time
        print(f"\n🏆 VERDICT: {verdict.decision.upper()}")
        print(f"   Confidence: {verdict.confidence:.0%}")
        print(f"   Duration: {duration:.1f}s")

    async def on_safety_alert(self, alert):
        print(f"\n⚠️  Safety Alert: {alert.type} (severity: {alert.severity:.2f})")


async def stream_with_callbacks():
    handler = DebateStreamHandler()

    agents = [
        Agent(name="optimist", model="gpt-4o"),
        Agent(name="realist", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Will AGI be achieved within 10 years?",
        agents=agents,
        rounds=2,
    )

    debate.assign_positions({
        "optimist": "believes AGI will be achieved within 10 years",
        "realist": "believes AGI is further away than optimists think",
    })

    # Register callbacks
    debate.on("debate_start", handler.on_debate_start)
    debate.on("round_start", handler.on_round_start)
    debate.on("turn_start", handler.on_turn_start)
    debate.on("turn_complete", handler.on_turn_complete)
    debate.on("evaluation_complete", handler.on_evaluation_complete)
    debate.on("deliberation_start", handler.on_deliberation_start)
    debate.on("verdict", handler.on_verdict)
    debate.on("safety_alert", handler.on_safety_alert)

    result = await debate.run()

    print(f"\n📊 Summary: {len(handler.turns)} turns completed")

asyncio.run(stream_with_callbacks())
```

## Streaming to Web Interface

```python
import asyncio
import json
from artemis.core.agent import Agent
from artemis.core.debate import Debate

async def stream_to_websocket(websocket):
    """Stream debate events to a WebSocket connection."""

    agents = [
        Agent(name="pro", model="gpt-4o"),
        Agent(name="con", model="gpt-4o"),
    ]

    debate = Debate(
        topic="Should AI development be open-sourced?",
        agents=agents,
        rounds=2,
    )

    debate.assign_positions({
        "pro": "supports open-source AI development",
        "con": "supports controlled AI development",
    })

    async for event in debate.stream_events():
        # Send event to WebSocket client
        message = {
            "type": event.type,
            "data": event.to_dict(),
            "timestamp": event.timestamp.isoformat(),
        }

        await websocket.send(json.dumps(message))

        # Handle specific event types
        if event.type == "turn_complete":
            # Also send structured argument data
            arg_message = {
                "type": "argument",
                "agent": event.agent,
                "round": event.turn.round,
                "level": event.turn.argument.level,
                "content": event.turn.argument.content,
                "evidence": [
                    {"type": e.type, "source": e.source}
                    for e in event.turn.argument.evidence
                ],
            }
            await websocket.send(json.dumps(arg_message))

        elif event.type == "verdict_complete":
            # Send final verdict
            verdict_message = {
                "type": "final_verdict",
                "decision": event.verdict.decision,
                "confidence": event.verdict.confidence,
                "reasoning": event.verdict.reasoning,
            }
            await websocket.send(json.dumps(verdict_message))


# Example FastAPI integration
"""
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect

app = FastAPI()

@app.websocket("/debate/stream")
async def debate_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        await stream_to_websocket(websocket)
    except WebSocketDisconnect:
        pass
"""
```

## Streaming with Rich Console

```python
import asyncio
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from artemis.core.agent import Agent
from artemis.core.debate import Debate

console = Console()

async def stream_with_rich():
    agents = [
        Agent(name="pro", model="gpt-4o"),
        Agent(name="con", model="gpt-4o"),
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
        title="🎯 Starting Debate"
    ))

    current_round = 0

    async for event in debate.stream_events():
        if event.type == "round_start":
            current_round = event.round
            console.print(f"\n[bold yellow]━━━ Round {current_round} ━━━[/bold yellow]")

        elif event.type == "turn_start":
            with console.status(f"[cyan]{event.agent}[/cyan] is thinking..."):
                # Wait for turn to complete
                pass

        elif event.type == "turn_complete":
            # Create argument panel
            agent_color = "green" if event.agent == "pro" else "red"
            panel = Panel(
                event.turn.argument.content,
                title=f"[bold {agent_color}]{event.agent.upper()}[/bold {agent_color}]",
                subtitle=f"Level: {event.turn.argument.level}",
                border_style=agent_color,
            )
            console.print(panel)

            # Show evidence if any
            if event.turn.argument.evidence:
                table = Table(title="Evidence", show_header=True)
                table.add_column("Type")
                table.add_column("Source")
                for e in event.turn.argument.evidence[:3]:
                    table.add_row(e.type, e.source)
                console.print(table)

        elif event.type == "evaluation_complete":
            score = event.evaluation.total_score
            score_color = "green" if score >= 7 else "yellow" if score >= 5 else "red"
            console.print(f"  [dim]Score:[/dim] [{score_color}]{score:.1f}/10[/{score_color}]")

        elif event.type == "deliberation_start":
            console.print("\n[bold magenta]🎓 Jury Deliberation[/bold magenta]")

        elif event.type == "verdict_complete":
            verdict = event.verdict
            decision_color = "green" if verdict.decision == "pro" else "red" if verdict.decision == "con" else "yellow"

            console.print(Panel(
                f"[bold {decision_color}]{verdict.decision.upper()}[/bold {decision_color}]\n\n"
                f"Confidence: {verdict.confidence:.0%}\n\n"
                f"{verdict.reasoning}",
                title="🏆 Final Verdict",
                border_style=decision_color,
            ))

asyncio.run(stream_with_rich())
```

## Streaming in LangGraph

```python
import asyncio
from artemis.integrations import create_debate_workflow

async def stream_langgraph_debate():
    workflow = create_debate_workflow(
        model="gpt-4o",
        rounds=2,
        enable_safety=True,
    )

    app = workflow.compile()

    print("Starting LangGraph debate stream...")
    print("=" * 60)

    async for event in app.astream(
        {"topic": "Should we adopt microservices architecture?"},
        stream_mode="values",  # Stream state updates
    ):
        if "current_round" in event:
            print(f"\n📍 Round {event['current_round']}")

        if "transcript" in event:
            transcript = event["transcript"]
            if transcript:
                latest = transcript[-1]
                print(f"\n{latest['agent'].upper()}:")
                print(f"{latest['argument']['content'][:200]}...")

        if "verdict" in event and event["verdict"]:
            verdict = event["verdict"]
            print(f"\n🏆 Verdict: {verdict['decision']}")
            print(f"Confidence: {verdict['confidence']:.0%}")

        if "safety_alerts" in event and event["safety_alerts"]:
            for alert in event["safety_alerts"]:
                print(f"\n⚠️ Safety: {alert['type']}")

asyncio.run(stream_langgraph_debate())
```

## Buffered Streaming for Slow Connections

```python
import asyncio
from collections import deque
from artemis.core.agent import Agent
from artemis.core.debate import Debate

class BufferedStreamer:
    """Buffer events for slow consumers."""

    def __init__(self, max_buffer: int = 100):
        self.buffer = deque(maxlen=max_buffer)
        self.consumers = []

    async def add_event(self, event):
        self.buffer.append(event)

        # Notify all consumers
        for consumer in self.consumers:
            try:
                await consumer(event)
            except Exception as e:
                print(f"Consumer error: {e}")

    def add_consumer(self, callback):
        self.consumers.append(callback)

    def get_buffered_events(self):
        """Get all buffered events (for late joiners)."""
        return list(self.buffer)


async def stream_with_buffer():
    streamer = BufferedStreamer()

    # Add a consumer
    async def print_consumer(event):
        if event["type"] == "turn_complete":
            print(f"[{event['agent']}] {event['content'][:50]}...")

    streamer.add_consumer(print_consumer)

    agents = [
        Agent(name="pro", model="gpt-4o"),
        Agent(name="con", model="gpt-4o"),
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

    async for event in debate.stream_events():
        await streamer.add_event({
            "type": event.type,
            "agent": getattr(event, "agent", None),
            "content": getattr(event.turn.argument, "content", None)
                       if hasattr(event, "turn") else None,
            "timestamp": event.timestamp.isoformat(),
        })

    # Late joiner can get buffered events
    print("\n--- Buffered Events ---")
    for event in streamer.get_buffered_events():
        if event["type"] == "turn_complete":
            print(f"Buffered: [{event['agent']}]")

asyncio.run(stream_with_buffer())
```

## Next Steps

- See [Basic Debate](basic-debate.md) for fundamentals
- Explore [LangGraph Workflow](langgraph-workflow.md) for graph-based streaming
- Add [Safety Monitors](safety-monitors.md) with streamed alerts
