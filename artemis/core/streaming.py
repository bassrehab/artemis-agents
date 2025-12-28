"""Streaming support for ARTEMIS debates.

Provides real-time streaming of debate events including argument chunks,
evaluations, and safety checks.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from typing import Any
from uuid import uuid4

from artemis.core.agent import Agent
from artemis.core.disagreement import DisagreementAnalyzer
from artemis.core.evaluation import AdaptiveEvaluator
from artemis.core.feedback import FeedbackSynthesizer
from artemis.core.jury import JuryPanel
from artemis.core.llm_evaluation import EvaluatorFactory
from artemis.core.types import (
    ArgumentLevel,
    DebateConfig,
    DebateContext,
    DebateMetadata,
    DebateResult,
    DebateState,
    SafetyAlert,
    SafetyResult,
    StreamEvent,
    StreamEventType,
    Turn,
    Verdict,
)
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


# Type alias for safety monitor callback
SafetyMonitorCallback = Callable[[Turn, DebateContext], SafetyResult]


class StreamCallback(ABC):
    """Abstract base class for streaming callbacks.

    Implement this class to receive streaming events during debate execution.

    Example:
        class PrintCallback(StreamCallback):
            async def on_event(self, event: StreamEvent) -> None:
                if event.event_type == StreamEventType.CHUNK:
                    print(event.content, end="", flush=True)
    """

    @abstractmethod
    async def on_event(self, event: StreamEvent) -> None:
        """Handle a streaming event.

        Args:
            event: The streaming event to handle.
        """
        pass


class ConsoleStreamCallback(StreamCallback):
    """A simple callback that prints events to console."""

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the callback.

        Args:
            verbose: If True, print all events. If False, only print chunks.
        """
        self.verbose = verbose

    async def on_event(self, event: StreamEvent) -> None:
        """Print the event to console."""
        if event.event_type == StreamEventType.CHUNK:
            print(event.content, end="", flush=True)
        elif event.event_type == StreamEventType.ARGUMENT_COMPLETE:
            print()  # Newline after argument
        elif self.verbose:
            print(f"\n[{event.event_type.value}] {event.agent or ''}")


class StreamingDebate:
    """A debate that streams events in real-time.

    This class provides an async iterator interface for consuming debate events
    as they happen, enabling real-time UI updates and progress tracking.

    Example:
        ```python
        debate = StreamingDebate(
            topic="Should AI be regulated?",
            agents=[agent1, agent2],
            rounds=3,
        )

        async for event in debate.run_streaming():
            if event.event_type == StreamEventType.CHUNK:
                print(event.content, end="", flush=True)
            elif event.event_type == StreamEventType.VERDICT:
                print(f"Winner: {event.verdict.decision}")
        ```
    """

    def __init__(
        self,
        topic: str,
        agents: list[Agent],
        rounds: int = 5,
        config: DebateConfig | None = None,
        jury: JuryPanel | None = None,
        evaluator: AdaptiveEvaluator | None = None,
        safety_monitors: list[SafetyMonitorCallback] | None = None,
        callbacks: list[StreamCallback] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a streaming debate.

        Args:
            topic: The debate topic.
            agents: List of agents participating in the debate.
            rounds: Number of debate rounds.
            config: Debate configuration.
            jury: Jury panel for verdict.
            evaluator: Argument evaluator.
            safety_monitors: Safety monitor callbacks.
            callbacks: Stream callbacks to notify on events.
            **kwargs: Additional arguments (jury_size, jury_model, etc.)
        """
        if len(agents) < 2:
            raise ValueError("need at least 2 agents for a debate")

        self.debate_id = str(uuid4())
        self.topic = topic
        self.agents = agents
        self.total_rounds = rounds
        self.config = config or DebateConfig()
        self._callbacks = callbacks or []

        # Core components
        self._jury = jury or JuryPanel(
            evaluators=kwargs.get("jury_size", 3),
            model=kwargs.get("jury_model", "gpt-4o"),
        )

        # Create evaluator based on evaluation mode
        if evaluator is not None:
            self._evaluator = evaluator
        else:
            self._evaluator = EvaluatorFactory.create(
                mode=self.config.evaluation_mode,
                model=kwargs.get("evaluator_model", "gpt-4o-mini"),
                api_key=kwargs.get("api_key"),
            )

        # Safety monitoring
        self._safety_monitors = safety_monitors or []
        self._safety_alerts: list[SafetyAlert] = []

        # Feedback synthesis
        self._feedback_synthesizer = FeedbackSynthesizer()
        self._agent_feedback: dict[str, str] = {}

        # Disagreement analysis
        self._disagreement_analyzer = DisagreementAnalyzer()

        # State tracking
        self._state = DebateState.SETUP
        self._current_round = 0
        self._transcript: list[Turn] = []
        self._agent_positions: dict[str, str] = {}
        self._started_at: datetime | None = None
        self._ended_at: datetime | None = None

        # Event queue for streaming
        self._event_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        logger.info(
            "StreamingDebate initialized",
            debate_id=self.debate_id,
            topic=topic[:50],
            agents=[a.name for a in agents],
            rounds=rounds,
        )

    def add_callback(self, callback: StreamCallback) -> None:
        """Add a streaming callback.

        Args:
            callback: The callback to add.
        """
        self._callbacks.append(callback)

    def assign_positions(self, positions: dict[str, str]) -> None:
        """Assign positions to agents."""
        for agent_name, position in positions.items():
            self._agent_positions[agent_name] = position
            for agent in self.agents:
                if agent.name == agent_name:
                    agent.set_position(position)

    async def _emit_event(self, event: StreamEvent) -> None:
        """Emit an event to all callbacks and the queue."""
        # Put in queue for async iterator consumption
        await self._event_queue.put(event)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                await callback.on_event(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def run_streaming(self) -> AsyncIterator[StreamEvent]:
        """Run the debate and yield streaming events.

        Yields:
            StreamEvent objects as the debate progresses.
        """
        # Create a list to collect events
        events: list[StreamEvent] = []
        original_emit = self._emit_event

        async def collect_event(event: StreamEvent) -> None:
            """Collect event and call original emit."""
            events.append(event)
            # Also notify callbacks
            for callback in self._callbacks:
                try:
                    await callback.on_event(event)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        # Override emit to collect events
        self._emit_event = collect_event

        try:
            # Run the debate
            await self._run_debate()

            # Yield all collected events
            for event in events:
                yield event

        except Exception as e:
            # Emit error event
            error_event = StreamEvent(
                event_type=StreamEventType.ERROR,
                error=str(e),
            )
            events.append(error_event)
            for event in events:
                yield event
            raise
        finally:
            # Restore original emit
            self._emit_event = original_emit

    async def _run_debate(self) -> DebateResult:
        """Run the complete debate, emitting events along the way."""
        self._started_at = datetime.utcnow()

        await self._emit_event(
            StreamEvent(
                event_type=StreamEventType.DEBATE_START,
                metadata={
                    "topic": self.topic,
                    "agents": ",".join(a.name for a in self.agents),
                    "rounds": self.total_rounds,
                },
            )
        )

        try:
            # Opening statements
            await self._run_opening()

            # Main debate rounds
            await self._run_main_rounds()

            # Closing arguments
            await self._run_closing()

            # Jury deliberation
            verdict = await self._run_deliberation()

            self._state = DebateState.COMPLETE

        except Exception as e:
            self._state = DebateState.HALTED
            await self._emit_event(
                StreamEvent(
                    event_type=StreamEventType.ERROR,
                    error=str(e),
                )
            )
            verdict = Verdict(
                decision="halted",
                confidence=0.0,
                reasoning=f"Debate halted due to error: {e}",
                unanimous=True,
            )

        self._ended_at = datetime.utcnow()

        # Emit final event
        await self._emit_event(
            StreamEvent(
                event_type=StreamEventType.DEBATE_END,
                verdict=verdict,
                metadata={
                    "duration_seconds": (
                        self._ended_at - self._started_at
                    ).total_seconds(),
                    "total_turns": len(self._transcript),
                },
            )
        )

        return self._build_result(verdict)

    async def _run_opening(self) -> None:
        """Opening statements phase."""
        self._state = DebateState.OPENING

        await self._emit_event(
            StreamEvent(
                event_type=StreamEventType.ROUND_START,
                round_num=0,
                metadata={"phase": "opening"},
            )
        )

        context = self._build_context()

        for i, agent in enumerate(self.agents):
            await self._emit_event(
                StreamEvent(
                    event_type=StreamEventType.TURN_START,
                    agent=agent.name,
                    round_num=0,
                    turn_num=i,
                )
            )

            # Generate argument with streaming
            argument = await self._stream_argument_generation(
                agent, context, ArgumentLevel.STRATEGIC, 0, i
            )

            turn = Turn(
                round=0,
                sequence=i,
                agent=agent.name,
                argument=argument,
            )

            # Evaluate and check safety
            await self._process_turn(turn, context)

            context = self._build_context()

        await self._emit_event(
            StreamEvent(
                event_type=StreamEventType.ROUND_COMPLETE,
                round_num=0,
            )
        )

        self._state = DebateState.DEBATE

    async def _run_main_rounds(self) -> None:
        """Main debate rounds."""
        for round_num in range(1, self.total_rounds + 1):
            self._current_round = round_num
            await self._execute_round(round_num)

    async def _execute_round(self, round_num: int) -> list[Turn]:
        """Execute a single round."""
        await self._emit_event(
            StreamEvent(
                event_type=StreamEventType.ROUND_START,
                round_num=round_num,
            )
        )

        round_turns: list[Turn] = []
        context = self._build_context()
        level = self._get_round_level(round_num)

        for i, agent in enumerate(self.agents):
            await self._emit_event(
                StreamEvent(
                    event_type=StreamEventType.TURN_START,
                    agent=agent.name,
                    round_num=round_num,
                    turn_num=i,
                )
            )

            # Update opponent models
            self._update_opponent_models(agent)

            # Generate argument with streaming
            argument = await self._stream_argument_generation(
                agent, context, level, round_num, i
            )

            turn = Turn(
                round=round_num,
                sequence=i,
                agent=agent.name,
                argument=argument,
            )

            # Evaluate and check safety
            await self._process_turn(turn, context)
            round_turns.append(turn)

            context = self._build_context()

        await self._emit_event(
            StreamEvent(
                event_type=StreamEventType.ROUND_COMPLETE,
                round_num=round_num,
            )
        )

        # Synthesize feedback
        self._synthesize_feedback()

        return round_turns

    async def _run_closing(self) -> None:
        """Closing arguments phase."""
        self._state = DebateState.CLOSING
        closing_round = self.total_rounds + 1

        await self._emit_event(
            StreamEvent(
                event_type=StreamEventType.ROUND_START,
                round_num=closing_round,
                metadata={"phase": "closing"},
            )
        )

        context = self._build_context()

        for i, agent in enumerate(self.agents):
            await self._emit_event(
                StreamEvent(
                    event_type=StreamEventType.TURN_START,
                    agent=agent.name,
                    round_num=closing_round,
                    turn_num=i,
                )
            )

            argument = await self._stream_argument_generation(
                agent, context, ArgumentLevel.STRATEGIC, closing_round, i
            )

            turn = Turn(
                round=closing_round,
                sequence=i,
                agent=agent.name,
                argument=argument,
            )

            await self._process_turn(turn, context)
            context = self._build_context()

        await self._emit_event(
            StreamEvent(
                event_type=StreamEventType.ROUND_COMPLETE,
                round_num=closing_round,
            )
        )

    async def _run_deliberation(self) -> Verdict:
        """Jury deliberation phase."""
        self._state = DebateState.DELIBERATION

        context = self._build_context()
        verdict = await self._jury.deliberate(self._transcript, context)

        await self._emit_event(
            StreamEvent(
                event_type=StreamEventType.VERDICT,
                verdict=verdict,
            )
        )

        return verdict

    async def _stream_argument_generation(
        self,
        agent: Agent,
        context: DebateContext,
        level: ArgumentLevel,
        round_num: int,
        turn_num: int,
    ):
        """Generate an argument, emitting chunk events.

        Currently emits the complete argument as a single chunk since
        streaming is not yet implemented in the agent. This can be
        extended to support true token-by-token streaming when the
        underlying models support it.
        """
        # Generate the argument
        argument = await agent.generate_argument(context, level)

        # Emit the content as a chunk (simulated streaming)
        # In future, this could be replaced with actual token streaming
        await self._emit_event(
            StreamEvent(
                event_type=StreamEventType.CHUNK,
                agent=agent.name,
                content=argument.content,
                round_num=round_num,
                turn_num=turn_num,
            )
        )

        # Emit argument complete
        await self._emit_event(
            StreamEvent(
                event_type=StreamEventType.ARGUMENT_COMPLETE,
                agent=agent.name,
                argument=argument,
                round_num=round_num,
                turn_num=turn_num,
            )
        )

        return argument

    async def _process_turn(
        self, turn: Turn, context: DebateContext
    ) -> None:
        """Process a turn: evaluate and check safety."""
        # Emit evaluation start
        await self._emit_event(
            StreamEvent(
                event_type=StreamEventType.EVALUATION_START,
                agent=turn.agent,
                round_num=turn.round,
            )
        )

        # Evaluate argument
        evaluation = await self._evaluator.evaluate_argument(
            turn.argument, context
        )

        turn = Turn(
            id=turn.id,
            round=turn.round,
            sequence=turn.sequence,
            agent=turn.agent,
            argument=turn.argument,
            evaluation=evaluation,
            timestamp=turn.timestamp,
        )

        # Emit evaluation complete
        await self._emit_event(
            StreamEvent(
                event_type=StreamEventType.EVALUATION_COMPLETE,
                agent=turn.agent,
                evaluation=evaluation,
                round_num=turn.round,
            )
        )

        # Run safety monitors
        safety_results = await self._run_safety_monitors(turn, context)
        if safety_results:
            turn = Turn(
                id=turn.id,
                round=turn.round,
                sequence=turn.sequence,
                agent=turn.agent,
                argument=turn.argument,
                evaluation=turn.evaluation,
                safety_results=safety_results,
                timestamp=turn.timestamp,
            )

        # Check for safety halts
        self._check_safety_halt(turn)

        # Record turn
        self._transcript.append(turn)

        # Emit turn complete
        await self._emit_event(
            StreamEvent(
                event_type=StreamEventType.TURN_COMPLETE,
                agent=turn.agent,
                turn=turn,
                round_num=turn.round,
            )
        )

    async def _run_safety_monitors(
        self, turn: Turn, context: DebateContext
    ) -> list[SafetyResult]:
        """Run all safety monitors."""
        if not self._safety_monitors:
            return []

        results: list[SafetyResult] = []

        for monitor in self._safety_monitors:
            try:
                # Emit safety check event
                await self._emit_event(
                    StreamEvent(
                        event_type=StreamEventType.SAFETY_CHECK,
                        agent=turn.agent,
                        metadata={"monitor": str(monitor)},
                    )
                )

                # Run monitor
                if asyncio.iscoroutinefunction(monitor):
                    result = await monitor(turn, context)
                else:
                    result = monitor(turn, context)

                if result and result.severity > 0:
                    results.append(result)

                    if result.should_alert:
                        alert = SafetyAlert(
                            monitor=result.monitor,
                            agent=turn.agent,
                            type=result.monitor,
                            severity=result.severity,
                            indicators=result.indicators,
                            turn_id=turn.id,
                        )
                        self._safety_alerts.append(alert)

                        # Emit safety alert event
                        await self._emit_event(
                            StreamEvent(
                                event_type=StreamEventType.SAFETY_ALERT,
                                agent=turn.agent,
                                safety_result=result,
                                metadata={
                                    "monitor": result.monitor,
                                    "severity": result.severity,
                                },
                            )
                        )

            except Exception as e:
                logger.error(f"Safety monitor error: {e}")

        return results

    def _check_safety_halt(self, turn: Turn) -> None:
        """Check if debate should be halted due to safety violation."""
        if not self.config.halt_on_safety_violation:
            return

        for result in turn.safety_results:
            if result.should_halt:
                alert = SafetyAlert(
                    monitor=result.monitor,
                    agent=turn.agent,
                    type="halt",
                    severity=result.severity,
                    indicators=result.indicators,
                    turn_id=turn.id,
                )
                self._safety_alerts.append(alert)
                raise RuntimeError(
                    f"Safety halt triggered by {result.monitor}"
                )

    def _update_opponent_models(self, agent: Agent) -> None:
        """Update agent's model of opponents."""
        for turn in self._transcript:
            if turn.agent != agent.name:
                agent.observe_opponent(turn.argument)

    def _get_round_level(self, round_num: int) -> ArgumentLevel:
        """Determine argument level for the round."""
        from artemis.core.types import EvaluationMode

        if self.config.evaluation_mode == EvaluationMode.FAST:
            return self._mechanical_level(round_num)

        disagreement = self._disagreement_analyzer.analyze(
            transcript=self._transcript,
            current_round=round_num,
            total_rounds=self.total_rounds,
        )

        return self._disagreement_analyzer.recommend_level(
            disagreement=disagreement,
            current_round=round_num,
            total_rounds=self.total_rounds,
        )

    def _mechanical_level(self, round_num: int) -> ArgumentLevel:
        """Mechanical level selection based on round progress."""
        progress = round_num / self.total_rounds

        if progress <= 0.3:
            return ArgumentLevel.STRATEGIC
        elif progress <= 0.7:
            return ArgumentLevel.TACTICAL
        else:
            return ArgumentLevel.OPERATIONAL

    def _build_context(self) -> DebateContext:
        """Build current debate context."""
        return DebateContext(
            topic=self.topic,
            current_round=self._current_round,
            total_rounds=self.total_rounds,
            turn_in_round=len(
                [t for t in self._transcript if t.round == self._current_round]
            ),
            transcript=self._transcript.copy(),
            agent_positions=self._agent_positions.copy(),
            agent_feedback=self._agent_feedback.copy(),
        )

    def _synthesize_feedback(self) -> None:
        """Synthesize feedback for agents."""
        from artemis.core.types import EvaluationMode

        if self.config.evaluation_mode == EvaluationMode.FAST:
            return

        if not self._transcript:
            return

        for agent in self.agents:
            own_turns = [t for t in self._transcript if t.agent == agent.name]
            opponent_turns = [
                t for t in self._transcript if t.agent != agent.name
            ]

            feedback = self._feedback_synthesizer.synthesize(
                agent_name=agent.name,
                own_turns=own_turns,
                opponent_turns=opponent_turns,
            )

            self._agent_feedback[
                agent.name
            ] = self._feedback_synthesizer.format_for_prompt(feedback)

    def _build_result(self, verdict: Verdict) -> DebateResult:
        """Build the final debate result."""
        metadata = DebateMetadata(
            started_at=self._started_at or datetime.utcnow(),
            ended_at=self._ended_at,
            total_rounds=self.total_rounds,
            total_turns=len(self._transcript),
            agents=[a.name for a in self.agents],
            jury_size=len(self._jury),
            safety_monitors=[str(m) for m in self._safety_monitors],
        )

        return DebateResult(
            debate_id=self.debate_id,
            topic=self.topic,
            verdict=verdict,
            transcript=self._transcript,
            safety_alerts=self._safety_alerts,
            metadata=metadata,
            final_state=self._state,
        )

    @property
    def transcript(self) -> list[Turn]:
        """Get the debate transcript."""
        return self._transcript.copy()

    @property
    def state(self) -> DebateState:
        """Get the current debate state."""
        return self._state

    def __repr__(self) -> str:
        return (
            f"StreamingDebate(topic={self.topic[:30]!r}..., "
            f"state={self._state.value}, "
            f"agents={[a.name for a in self.agents]})"
        )
