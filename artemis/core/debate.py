"""Debate orchestrator for multi-agent debates."""

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Any
from uuid import uuid4

from artemis.core.agent import Agent
from artemis.core.evaluation import AdaptiveEvaluator
from artemis.core.jury import JuryPanel
from artemis.core.types import (
    ArgumentLevel,
    DebateConfig,
    DebateContext,
    DebateMetadata,
    DebateResult,
    DebateState,
    SafetyAlert,
    SafetyResult,
    Turn,
    Verdict,
)
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


# Type alias for safety monitor callback
SafetyMonitorCallback = Callable[[Turn, DebateContext], SafetyResult]


class DebateError(Exception):
    """Base exception for debate errors."""

    pass


class DebateHaltedError(DebateError):
    """Raised when debate is halted due to safety violation."""

    def __init__(self, message: str, alert: SafetyAlert):
        super().__init__(message)
        self.alert = alert


class Debate:
    """Orchestrates a structured multi-agent debate."""

    def __init__(
        self,
        topic: str,
        agents: list[Agent],
        rounds: int = 5,
        config: DebateConfig | None = None,
        jury: JuryPanel | None = None,
        evaluator: AdaptiveEvaluator | None = None,
        safety_monitors: list[SafetyMonitorCallback] | None = None,
        **kwargs: Any,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("need at least 2 agents for a debate")

        self.debate_id = str(uuid4())
        self.topic = topic
        self.agents = agents
        self.total_rounds = rounds
        self.config = config or DebateConfig()

        # Core components
        self._jury = jury or JuryPanel(
            evaluators=kwargs.get("jury_size", 3),
            model=kwargs.get("jury_model", "gpt-4o"),
        )
        self._evaluator = evaluator or AdaptiveEvaluator()

        # Safety monitoring
        self._safety_monitors = safety_monitors or []
        self._safety_alerts: list[SafetyAlert] = []

        # State tracking
        self._state = DebateState.SETUP
        self._current_round = 0
        self._transcript: list[Turn] = []
        self._agent_positions: dict[str, str] = {}
        self._started_at: datetime | None = None
        self._ended_at: datetime | None = None

        # Token usage tracking
        self._model_usage: dict[str, dict[str, int]] = {}

        logger.info(
            "Debate initialized",
            debate_id=self.debate_id,
            topic=topic[:50],
            agents=[a.name for a in agents],
            rounds=rounds,
        )

    @property
    def state(self) -> DebateState:
        """Current debate state."""
        return self._state

    @property
    def transcript(self) -> list[Turn]:
        """Complete debate transcript."""
        return self._transcript.copy()

    @property
    def current_round(self) -> int:
        """Current round number."""
        return self._current_round

    @property
    def safety_alerts(self) -> list[SafetyAlert]:
        """All safety alerts raised during debate."""
        return self._safety_alerts.copy()

    def assign_positions(self, positions: dict[str, str]) -> None:
        """Assign positions to agents."""
        for agent_name, position in positions.items():
            self._agent_positions[agent_name] = position
            # Also set on the agent
            for agent in self.agents:
                if agent.name == agent_name:
                    agent.set_position(position)

        logger.debug("Positions assigned", positions=positions)

    async def run(self) -> DebateResult:
        """Run the complete debate and return results."""
        logger.info("Starting debate", debate_id=self.debate_id)
        self._started_at = datetime.utcnow()

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

        except DebateHaltedError:
            self._state = DebateState.HALTED
            # Create a halted verdict
            verdict = Verdict(
                decision="halted",
                confidence=0.0,
                reasoning="Debate halted due to safety violation.",
                unanimous=True,
            )

        self._ended_at = datetime.utcnow()

        result = self._build_result(verdict)
        logger.info(
            "Debate complete",
            debate_id=self.debate_id,
            state=self._state.value,
            verdict=verdict.decision,
            turns=len(self._transcript),
        )

        return result

    async def run_single_round(self) -> list[Turn]:
        """Run a single debate round (useful for step-by-step execution)."""
        if self._state == DebateState.SETUP:
            await self._run_opening()
            return self._get_round_turns(0)

        if self._state == DebateState.COMPLETE:
            raise DebateError("Debate already complete")

        if self._current_round >= self.total_rounds:
            await self._run_closing()
            return self._get_round_turns(-1)

        round_turns = await self._execute_round(self._current_round)
        self._current_round += 1

        return round_turns

    async def _run_opening(self):
        """Opening statements phase."""
        logger.info("Opening statements phase", debate_id=self.debate_id)
        self._state = DebateState.OPENING

        context = self._build_context()

        for i, agent in enumerate(self.agents):
            argument = await agent.generate_argument(
                context, ArgumentLevel.STRATEGIC
            )

            turn = Turn(
                round=0,
                sequence=i,
                agent=agent.name,
                argument=argument,
            )

            # Evaluate and check safety
            await self._process_turn(turn, context)

            # Update context for next agent
            context = self._build_context()

        self._state = DebateState.DEBATE

    async def _run_main_rounds(self):
        # FIXME: consider adding timeout per round
        logger.info(
            "Main debate phase",
            debate_id=self.debate_id,
            rounds=self.total_rounds,
        )

        for round_num in range(1, self.total_rounds + 1):
            self._current_round = round_num
            await self._execute_round(round_num)

    async def _execute_round(self, round_num: int):
        logger.debug("Executing round", round=round_num)
        round_turns: list[Turn] = []

        context = self._build_context()

        # Determine argument level based on round progression
        level = self._get_round_level(round_num)

        for i, agent in enumerate(self.agents):
            # Let agent observe opponent's latest argument
            self._update_opponent_models(agent)

            # Generate argument
            argument = await agent.generate_argument(context, level)

            turn = Turn(
                round=round_num,
                sequence=i,
                agent=agent.name,
                argument=argument,
            )

            # Evaluate and check safety
            await self._process_turn(turn, context)
            round_turns.append(turn)

            # Update context for next agent
            context = self._build_context()

        return round_turns

    async def _run_closing(self):
        logger.info("Closing arguments phase", debate_id=self.debate_id)
        self._state = DebateState.CLOSING

        context = self._build_context()

        # Use total_rounds + 1 as closing round marker
        closing_round = self.total_rounds + 1

        for i, agent in enumerate(self.agents):
            argument = await agent.generate_argument(
                context, ArgumentLevel.STRATEGIC
            )

            turn = Turn(
                round=closing_round,
                sequence=i,
                agent=agent.name,
                argument=argument,
            )

            await self._process_turn(turn, context)
            context = self._build_context()

    async def _run_deliberation(self):
        logger.info("Jury deliberation phase", debate_id=self.debate_id)
        self._state = DebateState.DELIBERATION

        context = self._build_context()
        verdict = await self._jury.deliberate(self._transcript, context)

        return verdict

    async def _process_turn(self, turn: Turn, context: DebateContext):
        # evaluate, check safety, record
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

        logger.debug(
            "Turn processed",
            agent=turn.agent,
            round=turn.round,
            score=evaluation.total_score if evaluation else None,
        )

    async def _run_safety_monitors(self, turn, context):
        if not self._safety_monitors:
            return []

        results: list[SafetyResult] = []

        for monitor in self._safety_monitors:
            try:
                # Support both sync and async monitors
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
                        logger.warning(
                            "Safety alert raised",
                            monitor=result.monitor,
                            agent=turn.agent,
                            severity=result.severity,
                        )

            except Exception as e:
                logger.error(
                    "Safety monitor error",
                    monitor=str(monitor),
                    error=str(e),
                )

        return results

    def _check_safety_halt(self, turn):
        # TODO: add configurable severity thresholds
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

                logger.error(
                    "Debate halted due to safety violation",
                    agent=turn.agent,
                    monitor=result.monitor,
                )
                raise DebateHaltedError(
                    f"Safety halt triggered by {result.monitor}", alert
                )

    def _update_opponent_models(self, agent):
        for turn in self._transcript:
            if turn.agent != agent.name:
                agent.observe_opponent(turn.argument)

    def _get_round_level(self, round_num):
        # XXX: these thresholds are somewhat arbitrary
        progress = round_num / self.total_rounds

        if progress <= 0.3:
            return ArgumentLevel.STRATEGIC
        elif progress <= 0.7:
            return ArgumentLevel.TACTICAL
        else:
            return ArgumentLevel.OPERATIONAL

    def _get_round_turns(self, round_num):
        return [t for t in self._transcript if t.round == round_num]

    def _build_context(self):
        return DebateContext(
            topic=self.topic,
            current_round=self._current_round,
            total_rounds=self.total_rounds,
            turn_in_round=len(self._get_round_turns(self._current_round)),
            transcript=self._transcript.copy(),
            agent_positions=self._agent_positions.copy(),
        )

    def _build_result(self, verdict):
        metadata = DebateMetadata(
            started_at=self._started_at or datetime.utcnow(),
            ended_at=self._ended_at,
            total_rounds=self.total_rounds,
            total_turns=len(self._transcript),
            agents=[a.name for a in self.agents],
            jury_size=len(self._jury),
            safety_monitors=[str(m) for m in self._safety_monitors],
            model_usage=self._model_usage,
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

    def add_safety_monitor(self, monitor: SafetyMonitorCallback) -> None:
        self._safety_monitors.append(monitor)
        logger.debug("Safety monitor added", monitor=str(monitor))

    def get_agent(self, name: str) -> Agent | None:
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def get_scores(self) -> dict[str, float]:
        scores: dict[str, list[float]] = {a.name: [] for a in self.agents}

        for turn in self._transcript:
            if turn.evaluation:
                scores[turn.agent].append(turn.evaluation.total_score)

        return {
            agent: (sum(s) / len(s) if s else 0.0)
            for agent, s in scores.items()
        }

    def get_analytics(self, include_jury_pulse: bool = False):
        """Compute analytics for this debate.

        Args:
            include_jury_pulse: If True, includes per-round jury sentiment
                               (requires additional LLM calls if not cached)

        Returns:
            DebateAnalytics object with momentum, metrics, and turning points
        """
        from artemis.analytics import DebateAnalyzer

        analyzer = DebateAnalyzer(
            transcript=self._transcript,
            agents=[a.name for a in self.agents],
            debate_id=self.debate_id,
            topic=self.topic,
        )
        return analyzer.analyze()

    def __repr__(self) -> str:
        return (
            f"Debate(topic={self.topic[:30]!r}..., "
            f"state={self._state.value}, "
            f"agents={[a.name for a in self.agents]})"
        )
