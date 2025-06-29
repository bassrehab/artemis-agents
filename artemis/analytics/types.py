"""Analytics data types for ARTEMIS debate analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MomentumPoint(BaseModel):
    """Momentum measurement at a specific point in time."""

    round: int
    agent: str
    score: float = Field(description="Current evaluation score (0-1)")
    momentum: float = Field(description="Rate of change (-1 to +1)")
    cumulative_advantage: float = Field(description="Running lead/deficit vs opponents")


class SwayEvent(BaseModel):
    """A significant shift in debate trajectory."""

    round: int
    turn_id: str
    agent: str
    sway_magnitude: float = Field(description="How much the tide shifted (0-1)")
    sway_direction: str = Field(description="'toward' or 'away_from' this agent")
    trigger_type: str = Field(description="'rebuttal', 'evidence', or 'rhetorical'")
    description: str


class TurningPoint(BaseModel):
    """A detected turning point in the debate."""

    round: int
    turn_id: str
    agent: str
    before_momentum: dict[str, float] = Field(description="Momentum per agent before")
    after_momentum: dict[str, float] = Field(description="Momentum per agent after")
    significance: float = Field(ge=0, le=1, description="How significant (0-1)")
    analysis: str = Field(description="Description of what caused the shift")


class JurySentiment(BaseModel):
    """Per-round jury sentiment snapshot."""

    round: int
    agent_leanings: dict[str, float] = Field(
        description="Agent -> lean score (-1 to +1, negative=losing)"
    )
    confidence: float = Field(ge=0, le=1)
    perspectives_breakdown: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Perspective -> agent -> score breakdown",
    )


class RoundMetrics(BaseModel):
    """Computed metrics for a single round."""

    round: int
    agent_scores: dict[str, float] = Field(description="Average score per agent this round")
    score_delta: dict[str, float] = Field(
        default_factory=dict,
        description="Change from previous round",
    )
    rebuttal_effectiveness: dict[str, float] = Field(
        default_factory=dict,
        description="Rebuttal effectiveness per agent (0-1)",
    )
    evidence_utilization: dict[str, float] = Field(
        default_factory=dict,
        description="Evidence use rate per agent (0-1)",
    )
    argument_level_distribution: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Agent -> level -> count",
    )


class DebateAnalytics(BaseModel):
    """Complete analytics package for a debate."""

    debate_id: str
    topic: str = ""
    agents: list[str]
    rounds: int

    # Momentum data
    momentum_history: list[MomentumPoint] = Field(default_factory=list)
    sway_events: list[SwayEvent] = Field(default_factory=list)
    turning_points: list[TurningPoint] = Field(default_factory=list)

    # Per-round metrics
    round_metrics: list[RoundMetrics] = Field(default_factory=list)

    # Optional jury pulse (if enabled during debate)
    jury_sentiment_history: list[JurySentiment] | None = None

    # Aggregate metrics
    final_momentum: dict[str, float] = Field(
        default_factory=dict,
        description="Final momentum per agent",
    )
    rebuttal_effectiveness_overall: dict[str, float] = Field(
        default_factory=dict,
        description="Overall rebuttal effectiveness per agent",
    )
    evidence_utilization_overall: dict[str, float] = Field(
        default_factory=dict,
        description="Overall evidence utilization per agent",
    )
    argument_diversity_index: dict[str, float] = Field(
        default_factory=dict,
        description="Argument level diversity per agent (0-1, higher=more balanced)",
    )
    predicted_winner_confidence: float | None = Field(
        default=None,
        description="Confidence in predicted winner (if computed)",
    )

    def get_momentum_trajectory(self, agent: str) -> list[float]:
        """Get momentum values over time for an agent."""
        return [
            mp.momentum
            for mp in self.momentum_history
            if mp.agent == agent
        ]

    def get_score_trajectory(self, agent: str) -> list[float]:
        """Get score values over time for an agent."""
        return [
            rm.agent_scores.get(agent, 0.0)
            for rm in self.round_metrics
        ]

    def get_leader_per_round(self) -> list[str]:
        """Get the leading agent after each round."""
        leaders = []
        for rm in self.round_metrics:
            if rm.agent_scores:
                leader = max(rm.agent_scores, key=rm.agent_scores.get)
                leaders.append(leader)
        return leaders

    def count_lead_changes(self) -> int:
        """Count how many times the lead changed hands."""
        leaders = self.get_leader_per_round()
        if len(leaders) < 2:
            return 0
        changes = sum(1 for i in range(1, len(leaders)) if leaders[i] != leaders[i - 1])
        return changes
