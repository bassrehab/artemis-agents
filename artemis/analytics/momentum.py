"""Momentum tracking and turning point detection for ARTEMIS debates."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from artemis.analytics.types import (
    JurySentiment,
    MomentumPoint,
    SwayEvent,
    TurningPoint,
)

if TYPE_CHECKING:
    from artemis.core.jury import JuryPanel
    from artemis.core.types import DebateContext, Turn


class MomentumTracker:
    """Tracks debate momentum and detects turning points.

    Momentum is calculated based on evaluation scores and their rate of change.
    Turning points are detected when momentum significantly shifts.

    Example:
        tracker = MomentumTracker(smoothing_window=2)
        momentum_history, turning_points = tracker.compute_from_transcript(
            transcript, agents
        )
    """

    def __init__(
        self,
        smoothing_window: int = 2,
        turning_point_threshold: float = 0.3,
    ) -> None:
        """Initialize momentum tracker.

        Args:
            smoothing_window: Number of rounds to average for smoothing
            turning_point_threshold: Minimum momentum shift to flag as turning point
        """
        self.smoothing_window = smoothing_window
        self.turning_point_threshold = turning_point_threshold

    def compute_from_transcript(
        self,
        transcript: list[Turn],
        agents: list[str],
    ) -> tuple[list[MomentumPoint], list[TurningPoint]]:
        """Compute momentum history and detect turning points.

        Args:
            transcript: List of Turn objects from the debate
            agents: List of agent names

        Returns:
            Tuple of (momentum_history, turning_points)
        """
        if not transcript:
            return [], []

        # Group turns by round
        rounds_data: dict[int, list[Turn]] = defaultdict(list)
        for turn in transcript:
            rounds_data[turn.round].append(turn)

        sorted_rounds = sorted(rounds_data.keys())

        # Track scores and momentum per agent
        agent_scores: dict[str, list[float]] = {agent: [] for agent in agents}
        momentum_history: list[MomentumPoint] = []
        previous_momentum: dict[str, float] = {agent: 0.0 for agent in agents}
        cumulative_advantage: dict[str, float] = {agent: 0.0 for agent in agents}

        for round_num in sorted_rounds:
            round_turns = rounds_data[round_num]
            round_agent_scores = self._compute_round_scores(round_turns, agents)

            # Update cumulative advantage (lead over average opponent score)
            if round_agent_scores:
                avg_score = sum(round_agent_scores.values()) / len(round_agent_scores)
                for agent in agents:
                    score = round_agent_scores.get(agent, 0.0)
                    cumulative_advantage[agent] += score - avg_score

            for agent in agents:
                score = round_agent_scores.get(agent, 0.0)
                agent_scores[agent].append(score)

                # Calculate momentum (rate of change)
                momentum = self._compute_momentum(
                    agent_scores[agent],
                    self.smoothing_window,
                )

                momentum_history.append(
                    MomentumPoint(
                        round=round_num,
                        agent=agent,
                        score=score,
                        momentum=momentum,
                        cumulative_advantage=cumulative_advantage[agent],
                    )
                )

                previous_momentum[agent] = momentum

        # Detect turning points
        turning_points = self._detect_turning_points(momentum_history, agents)

        return momentum_history, turning_points

    def _compute_round_scores(
        self,
        round_turns: list[Turn],
        agents: list[str],
    ) -> dict[str, float]:
        """Compute average score per agent for a round."""
        agent_scores: dict[str, list[float]] = {agent: [] for agent in agents}

        for turn in round_turns:
            if turn.evaluation and turn.agent in agents:
                agent_scores[turn.agent].append(turn.evaluation.total_score)

        return {
            agent: sum(scores) / len(scores) if scores else 0.0
            for agent, scores in agent_scores.items()
        }

    def _compute_momentum(
        self,
        scores: list[float],
        window: int,
    ) -> float:
        """Compute momentum using exponential smoothing.

        Momentum is the smoothed rate of change of scores.
        """
        if len(scores) < 2:
            return 0.0

        # Calculate deltas
        deltas = []
        for i in range(1, len(scores)):
            if scores[i - 1] > 0:
                delta = (scores[i] - scores[i - 1]) / scores[i - 1]
            else:
                delta = scores[i] - scores[i - 1]
            deltas.append(delta)

        # Apply exponential smoothing
        if not deltas:
            return 0.0

        # Use last `window` deltas
        recent_deltas = deltas[-window:] if len(deltas) >= window else deltas

        # Exponential weights (more recent = higher weight)
        weights = [2 ** i for i in range(len(recent_deltas))]
        total_weight = sum(weights)

        if total_weight == 0:
            return 0.0

        smoothed = sum(d * w for d, w in zip(recent_deltas, weights)) / total_weight

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, smoothed))

    def _detect_turning_points(
        self,
        momentum_history: list[MomentumPoint],
        agents: list[str],
    ) -> list[TurningPoint]:
        """Detect turning points where momentum significantly shifts."""
        turning_points: list[TurningPoint] = []

        # Group by round
        rounds_data: dict[int, dict[str, MomentumPoint]] = defaultdict(dict)
        for mp in momentum_history:
            rounds_data[mp.round][mp.agent] = mp

        sorted_rounds = sorted(rounds_data.keys())

        for i in range(1, len(sorted_rounds)):
            prev_round = sorted_rounds[i - 1]
            curr_round = sorted_rounds[i]

            prev_momentum = {
                agent: rounds_data[prev_round].get(agent, MomentumPoint(
                    round=prev_round, agent=agent, score=0, momentum=0, cumulative_advantage=0
                )).momentum
                for agent in agents
            }
            curr_momentum = {
                agent: rounds_data[curr_round].get(agent, MomentumPoint(
                    round=curr_round, agent=agent, score=0, momentum=0, cumulative_advantage=0
                )).momentum
                for agent in agents
            }

            # Check for significant shifts
            for agent in agents:
                prev_m = prev_momentum.get(agent, 0.0)
                curr_m = curr_momentum.get(agent, 0.0)
                shift = abs(curr_m - prev_m)

                # Detect turning point conditions:
                # 1. Sign flip (momentum changed direction)
                # 2. Large magnitude change
                sign_flip = (prev_m > 0 and curr_m < 0) or (prev_m < 0 and curr_m > 0)
                large_shift = shift >= self.turning_point_threshold

                if sign_flip or large_shift:
                    # Find the turn that caused this
                    curr_data = rounds_data[curr_round].get(agent)
                    turn_id = f"round_{curr_round}_{agent}"

                    analysis = self._generate_turning_point_analysis(
                        agent, prev_m, curr_m, sign_flip, large_shift
                    )

                    turning_points.append(
                        TurningPoint(
                            round=curr_round,
                            turn_id=turn_id,
                            agent=agent,
                            before_momentum=prev_momentum,
                            after_momentum=curr_momentum,
                            significance=min(1.0, shift / self.turning_point_threshold),
                            analysis=analysis,
                        )
                    )

        return turning_points

    def _generate_turning_point_analysis(
        self,
        agent: str,
        prev_momentum: float,
        curr_momentum: float,
        sign_flip: bool,
        large_shift: bool,
    ) -> str:
        """Generate human-readable analysis of a turning point."""
        if sign_flip:
            if curr_momentum > 0:
                return f"{agent} reversed their declining trend and gained momentum"
            else:
                return f"{agent} lost their positive momentum and began declining"
        elif large_shift:
            if curr_momentum > prev_momentum:
                return f"{agent} significantly accelerated their momentum"
            else:
                return f"{agent} experienced a significant momentum drop"
        return f"Momentum shift detected for {agent}"

    def detect_sway_events(
        self,
        transcript: list[Turn],
    ) -> list[SwayEvent]:
        """Detect significant sway events in the debate.

        A sway event is when a single argument significantly affects
        the balance of the debate.
        """
        sway_events: list[SwayEvent] = []

        # Look for turns with high scores that follow low opponent scores
        prev_turn: Turn | None = None

        for turn in transcript:
            if turn.evaluation is None:
                prev_turn = turn
                continue

            current_score = turn.evaluation.total_score

            # Check if this is a strong argument (score > 0.75)
            if current_score > 0.75:
                # Determine sway type based on argument characteristics
                trigger_type = self._determine_trigger_type(turn)

                # Calculate sway magnitude based on score and previous context
                sway_magnitude = (current_score - 0.5) * 2  # Scale 0.5-1.0 to 0-1

                sway_events.append(
                    SwayEvent(
                        round=turn.round,
                        turn_id=turn.id,
                        agent=turn.agent,
                        sway_magnitude=min(1.0, sway_magnitude),
                        sway_direction="toward",
                        trigger_type=trigger_type,
                        description=f"Strong {trigger_type} argument by {turn.agent}",
                    )
                )

            # Check if opponent was hurt by a rebuttal
            if prev_turn and prev_turn.agent != turn.agent:
                if hasattr(turn.argument, "rebuts") and turn.argument.rebuts:
                    # This argument rebuts something
                    if current_score > 0.6:
                        sway_events.append(
                            SwayEvent(
                                round=turn.round,
                                turn_id=turn.id,
                                agent=prev_turn.agent,
                                sway_magnitude=current_score - 0.5,
                                sway_direction="away_from",
                                trigger_type="rebuttal",
                                description=f"{turn.agent} effectively rebutted {prev_turn.agent}",
                            )
                        )

            prev_turn = turn

        return sway_events

    def _determine_trigger_type(self, turn: Turn) -> str:
        """Determine what type of argument caused the sway."""
        if not turn.argument:
            return "rhetorical"

        # Check for evidence
        evidence_count = len(turn.argument.evidence) if turn.argument.evidence else 0
        if evidence_count > 2:
            return "evidence"

        # Check for rebuttals
        if hasattr(turn.argument, "rebuts") and turn.argument.rebuts:
            return "rebuttal"

        return "rhetorical"

    def compute_round_momentum(
        self,
        round_turns: list[Turn],
        previous_scores: dict[str, list[float]],
        agents: list[str],
    ) -> dict[str, float]:
        """Compute momentum for a single round.

        Useful for live/streaming updates during a debate.

        Args:
            round_turns: Turns from the current round
            previous_scores: Historical scores per agent
            agents: List of agent names

        Returns:
            Dictionary of agent -> momentum value
        """
        round_scores = self._compute_round_scores(round_turns, agents)
        momentum = {}

        for agent in agents:
            scores = previous_scores.get(agent, []).copy()
            scores.append(round_scores.get(agent, 0.0))
            momentum[agent] = self._compute_momentum(scores, self.smoothing_window)

        return momentum


class JuryPulseTracker:
    """Track jury sentiment during debate execution.

    This is optional and expensive as it requires additional LLM calls
    to poll the jury after each round.

    Example:
        pulse_tracker = JuryPulseTracker(jury, sample_frequency=1)
        sentiment = await pulse_tracker.sample_sentiment(transcript, context)
    """

    def __init__(
        self,
        jury: JuryPanel,
        sample_frequency: int = 1,
    ) -> None:
        """Initialize pulse tracker.

        Args:
            jury: The jury panel to poll
            sample_frequency: How often to sample (1 = every round)
        """
        self._jury = jury
        self._sample_frequency = sample_frequency
        self._history: list[JurySentiment] = []
        self._sample_count = 0

    async def sample_sentiment(
        self,
        transcript: list[Turn],
        context: DebateContext,
    ) -> JurySentiment | None:
        """Poll jury for current sentiment.

        This triggers LLM calls for each juror - use sparingly.

        Args:
            transcript: Current debate transcript
            context: Current debate context

        Returns:
            JurySentiment snapshot, or None if skipped due to frequency
        """
        self._sample_count += 1

        # Skip if not at sample frequency
        if self._sample_count % self._sample_frequency != 0:
            return None

        # Get current round
        current_round = max(t.round for t in transcript) if transcript else 0

        # Poll each juror (this is expensive!)
        from artemis.core.jury import JurorEvaluation

        evaluations: list[JurorEvaluation] = []
        for juror in self._jury.jurors:
            evaluation = await juror.evaluate(transcript, context)
            evaluations.append(evaluation)

        # Aggregate into sentiment
        agent_leanings = self._compute_leanings(evaluations)
        perspectives_breakdown = self._compute_perspectives(evaluations)
        confidence = sum(e.confidence for e in evaluations) / len(evaluations)

        sentiment = JurySentiment(
            round=current_round,
            agent_leanings=agent_leanings,
            confidence=confidence,
            perspectives_breakdown=perspectives_breakdown,
        )

        self._history.append(sentiment)
        return sentiment

    def _compute_leanings(
        self,
        evaluations: list[Any],
    ) -> dict[str, float]:
        """Compute agent leanings from juror evaluations.

        Returns values in range [-1, 1] where:
        - Positive = winning
        - Negative = losing
        - 0 = neutral
        """
        if not evaluations:
            return {}

        # Collect all agents
        all_agents: set[str] = set()
        for e in evaluations:
            all_agents.update(e.agent_scores.keys())

        # Compute normalized leanings
        leanings = {}
        for agent in all_agents:
            scores = [e.agent_scores.get(agent, 0.0) for e in evaluations]
            avg_score = sum(scores) / len(scores) if scores else 0.0

            # Normalize to [-1, 1] based on distance from 0.5
            leanings[agent] = (avg_score - 0.5) * 2

        return leanings

    def _compute_perspectives(
        self,
        evaluations: list[Any],
    ) -> dict[str, dict[str, float]]:
        """Compute per-perspective breakdown."""
        breakdown: dict[str, dict[str, float]] = {}

        for e in evaluations:
            perspective_name = e.perspective.value if hasattr(e.perspective, "value") else str(e.perspective)
            breakdown[perspective_name] = e.agent_scores.copy()

        return breakdown

    def get_sentiment_history(self) -> list[JurySentiment]:
        """Return accumulated sentiment samples."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear accumulated history."""
        self._history = []
        self._sample_count = 0
