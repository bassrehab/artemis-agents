"""
ARTEMIS Core Types

Pydantic models for the ARTEMIS debate framework.
Implements data structures for H-L-DAG arguments, turns, evaluations, and verdicts.
"""

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class ArgumentLevel(str, Enum):
    """Hierarchical argument levels in H-L-DAG."""

    STRATEGIC = "strategic"
    """High-level position and thesis."""

    TACTICAL = "tactical"
    """Supporting arguments and evidence."""

    OPERATIONAL = "operational"
    """Specific facts, quotes, and examples."""


class DebateState(str, Enum):
    """Debate lifecycle states."""

    SETUP = "setup"
    """Initial configuration phase."""

    OPENING = "opening"
    """Opening statements phase."""

    DEBATE = "debate"
    """Main debate rounds."""

    CLOSING = "closing"
    """Closing arguments phase."""

    DELIBERATION = "deliberation"
    """Jury deliberation phase."""

    COMPLETE = "complete"
    """Debate finished."""

    HALTED = "halted"
    """Debate halted due to safety violation."""


class SafetyIndicatorType(str, Enum):
    """Types of safety indicators."""

    CAPABILITY_DROP = "capability_drop"
    STRATEGIC_TIMING = "strategic_timing"
    SELECTIVE_ENGAGEMENT = "selective_engagement"
    FACTUAL_INCONSISTENCY = "factual_inconsistency"
    LOGICAL_FALLACY = "logical_fallacy"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    CITATION_FABRICATION = "citation_fabrication"
    BEHAVIORAL_DRIFT = "behavioral_drift"
    ETHICS_BOUNDARY = "ethics_boundary"


class JuryPerspective(str, Enum):
    """Jury member evaluation perspectives."""

    ANALYTICAL = "analytical"
    """Focus on logic and evidence."""

    ETHICAL = "ethical"
    """Focus on moral implications."""

    PRACTICAL = "practical"
    """Focus on feasibility and impact."""

    ADVERSARIAL = "adversarial"
    """Challenge all arguments."""

    SYNTHESIZING = "synthesizing"
    """Find common ground."""


class JurorConfig(BaseModel):
    """Configuration for an individual juror."""

    perspective: JuryPerspective = JuryPerspective.ANALYTICAL
    model: str = "gpt-4o"
    criteria: list[str] | None = None
    api_key: str | None = None


# =============================================================================
# Evidence and Causal Models
# =============================================================================


class Evidence(BaseModel):
    """Supporting evidence for an argument."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["fact", "statistic", "quote", "example", "study", "expert_opinion"]
    content: str
    source: str | None = None
    url: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    verified: bool = False

    model_config = {"frozen": True}


class CausalLink(BaseModel):
    """A causal relationship between concepts in an argument."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    cause: str
    effect: str
    mechanism: str | None = None
    """Explanation of how cause leads to effect."""
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    """Strength of the causal relationship."""
    bidirectional: bool = False
    """Whether the relationship works both ways."""

    model_config = {"frozen": True}


# =============================================================================
# Argument Model
# =============================================================================


class Argument(BaseModel):
    """A structured argument in the H-L-DAG framework."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    agent: str
    """Name of the agent who made this argument."""
    level: ArgumentLevel
    """Hierarchical level: strategic, tactical, or operational."""
    content: str
    """The argument text."""
    evidence: list[Evidence] = Field(default_factory=list)
    """Supporting evidence."""
    causal_links: list[CausalLink] = Field(default_factory=list)
    """Causal relationships established in the argument."""
    rebuts: str | None = None
    """ID of the argument this rebuts, if any."""
    supports: str | None = None
    """ID of the argument this supports, if any."""
    ethical_score: float | None = None
    """Ethical alignment score (0-1)."""
    thinking_trace: str | None = None
    """Reasoning trace from extended thinking models."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"frozen": True}


# =============================================================================
# Evaluation Models
# =============================================================================


class CriterionScore(BaseModel):
    """Score for a single evaluation criterion."""

    criterion: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    reasoning: str | None = None


class CausalGraphUpdate(BaseModel):
    """Updates to the causal graph from argument evaluation."""

    added_links: list[CausalLink] = Field(default_factory=list)
    strengthened_links: list[str] = Field(default_factory=list)
    """IDs of links whose strength increased."""
    weakened_links: list[str] = Field(default_factory=list)
    """IDs of links whose strength decreased."""


class ArgumentEvaluation(BaseModel):
    """Evaluation of an argument using L-AE-CR."""

    argument_id: str
    scores: dict[str, float]
    """Scores for each criterion."""
    weights: dict[str, float]
    """Adapted weights for each criterion."""
    criterion_details: list[CriterionScore] = Field(default_factory=list)
    """Detailed breakdown per criterion."""
    causal_score: float = Field(ge=0.0, le=1.0)
    """Score for causal reasoning quality."""
    total_score: float = Field(ge=0.0, le=1.0)
    """Weighted total score."""
    causal_graph_update: CausalGraphUpdate | None = None
    evaluator_notes: str | None = None


# =============================================================================
# Safety Models
# =============================================================================


class SafetyIndicator(BaseModel):
    """An indicator of potential safety concern."""

    type: SafetyIndicatorType
    severity: float = Field(ge=0.0, le=1.0)
    evidence: str | list[str]
    """Description or list of evidence for this indicator."""
    metadata: dict[str, str | float | bool] = Field(default_factory=dict)


class SafetyResult(BaseModel):
    """Result of safety analysis on a turn."""

    monitor: str
    """Name of the monitor that produced this result."""
    severity: float = Field(ge=0.0, le=1.0)
    """Overall severity score."""
    indicators: list[SafetyIndicator] = Field(default_factory=list)
    should_alert: bool = False
    should_halt: bool = False
    analysis_notes: str | None = None


class SafetyAlert(BaseModel):
    """A safety alert raised during debate."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    monitor: str
    """Name of the monitor that raised this alert."""
    agent: str
    """Agent that triggered the alert."""
    type: str
    """Type of safety concern (e.g., 'sandbagging', 'deception')."""
    severity: float = Field(ge=0.0, le=1.0)
    indicators: list[SafetyIndicator] = Field(default_factory=list)
    turn_id: str | None = None
    """ID of the turn that triggered this alert."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolution_notes: str | None = None


# =============================================================================
# Turn Model
# =============================================================================


class Turn(BaseModel):
    """A single turn in the debate."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    round: int = Field(ge=0)
    """Round number (0 = opening, -1 = closing)."""
    sequence: int = Field(ge=0)
    """Sequence within the round."""
    agent: str
    """Name of the agent taking this turn."""
    argument: Argument
    evaluation: ArgumentEvaluation | None = None
    safety_results: list[SafetyResult] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Verdict Models
# =============================================================================


class DissentingOpinion(BaseModel):
    """A dissenting opinion from a jury member."""

    juror_id: str
    perspective: JuryPerspective
    position: str
    """The dissenting position."""
    reasoning: str
    """Reasoning for the dissent."""
    score_deviation: float
    """How much this juror's score deviated from consensus."""


class Verdict(BaseModel):
    """Final jury verdict on the debate."""

    decision: str
    """The verdict decision (e.g., winner name or 'draw')."""
    confidence: float = Field(ge=0.0, le=1.0)
    """Confidence in the verdict."""
    reasoning: str
    """Explanation of the verdict."""
    dissenting_opinions: list[DissentingOpinion] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    """Scores by agent or criterion."""
    unanimous: bool = False
    """Whether all jurors agreed."""


# =============================================================================
# Debate Result Models
# =============================================================================


class DebateMetadata(BaseModel):
    """Metadata about a debate."""

    started_at: datetime
    ended_at: datetime | None = None
    total_rounds: int
    total_turns: int = 0
    agents: list[str] = Field(default_factory=list)
    jury_size: int = 0
    safety_monitors: list[str] = Field(default_factory=list)
    model_usage: dict[str, dict[str, int]] = Field(default_factory=dict)
    """Token usage per model: {model: {prompt_tokens, completion_tokens}}."""


class DebateResult(BaseModel):
    """Complete result of a debate."""

    debate_id: str = Field(default_factory=lambda: str(uuid4()))
    topic: str
    verdict: Verdict
    transcript: list[Turn] = Field(default_factory=list)
    safety_alerts: list[SafetyAlert] = Field(default_factory=list)
    metadata: DebateMetadata
    final_state: DebateState = DebateState.COMPLETE


# =============================================================================
# Configuration Models
# =============================================================================


class ReasoningConfig(BaseModel):
    """Configuration for reasoning models (o1, R1, etc.)."""

    enabled: bool = True
    thinking_budget: int = Field(default=8000, ge=1000, le=32000)
    """Maximum tokens for extended thinking."""
    strategy: Literal["think-then-argue", "interleaved", "final-reflection"] = "think-then-argue"
    """When to apply extended thinking."""
    include_trace_in_output: bool = False
    """Whether to include thinking trace in final output."""


class EvaluationCriteria(BaseModel):
    """Evaluation criteria with weights."""

    logical_coherence: float = Field(default=0.25, ge=0.0, le=1.0)
    evidence_quality: float = Field(default=0.25, ge=0.0, le=1.0)
    causal_reasoning: float = Field(default=0.20, ge=0.0, le=1.0)
    ethical_alignment: float = Field(default=0.15, ge=0.0, le=1.0)
    persuasiveness: float = Field(default=0.15, ge=0.0, le=1.0)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary of weights."""
        return {
            "logical_coherence": self.logical_coherence,
            "evidence_quality": self.evidence_quality,
            "causal_reasoning": self.causal_reasoning,
            "ethical_alignment": self.ethical_alignment,
            "persuasiveness": self.persuasiveness,
        }

    def normalize(self) -> "EvaluationCriteria":
        """Return normalized criteria (weights sum to 1)."""
        total = (
            self.logical_coherence
            + self.evidence_quality
            + self.causal_reasoning
            + self.ethical_alignment
            + self.persuasiveness
        )
        if total == 0:
            return self
        return EvaluationCriteria(
            logical_coherence=self.logical_coherence / total,
            evidence_quality=self.evidence_quality / total,
            causal_reasoning=self.causal_reasoning / total,
            ethical_alignment=self.ethical_alignment / total,
            persuasiveness=self.persuasiveness / total,
        )


class DebateConfig(BaseModel):
    """Configuration for a debate."""

    # Timing
    turn_timeout: int = Field(default=60, ge=10, le=600)
    """Timeout per turn in seconds."""
    round_timeout: int = Field(default=300, ge=60, le=1800)
    """Timeout per round in seconds."""

    # Argument generation
    max_argument_tokens: int = Field(default=1000, ge=100, le=4000)
    require_evidence: bool = True
    require_causal_links: bool = True
    min_evidence_per_argument: int = Field(default=0, ge=0, le=5)

    # Evaluation
    evaluation_criteria: EvaluationCriteria = Field(default_factory=EvaluationCriteria)
    adaptation_enabled: bool = True
    """Enable dynamic criteria weight adaptation."""
    adaptation_rate: float = Field(default=0.1, ge=0.0, le=0.5)

    # Safety
    safety_mode: Literal["off", "passive", "active"] = "passive"
    halt_on_safety_violation: bool = False

    # Logging and tracing
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    trace_enabled: bool = False
    """Enable detailed execution tracing."""


# =============================================================================
# Message Types (for LLM interactions)
# =============================================================================


class Message(BaseModel):
    """A message in an LLM conversation."""

    role: Literal["system", "user", "assistant"]
    content: str
    name: str | None = None
    """Optional name for multi-agent scenarios."""

    model_config = {"frozen": True}


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None
    """Tokens used for extended thinking (if applicable)."""


class ModelResponse(BaseModel):
    """Response from an LLM."""

    content: str
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None
    finish_reason: str | None = None


class ReasoningResponse(ModelResponse):
    """Response from a reasoning model with thinking trace."""

    thinking: str | None = None
    """The extended thinking/reasoning trace."""
    thinking_tokens: int = 0


# =============================================================================
# Context Types
# =============================================================================


class DebateContext(BaseModel):
    """Context passed to agents and evaluators."""

    topic: str
    current_round: int
    total_rounds: int
    turn_in_round: int
    transcript: list[Turn] = Field(default_factory=list)
    agent_positions: dict[str, str] = Field(default_factory=dict)
    """Agent name -> their assigned position."""
    topic_sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    """Sensitivity level of the topic (affects ethical weighting)."""
    topic_complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    """Complexity level (affects causal reasoning weighting)."""
    causal_graph: dict[str, list[CausalLink]] = Field(default_factory=dict)
    """Accumulated causal links by agent."""
