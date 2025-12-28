"""
ARTEMIS Core Types

Pydantic models for the ARTEMIS debate framework.
Implements data structures for H-L-DAG arguments, turns, evaluations, and verdicts.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
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


class EvaluationMode(str, Enum):
    """Evaluation mode controlling accuracy vs cost tradeoff.

    QUALITY: LLM-native evaluation for all criteria. Highest accuracy, highest cost.
             Use for benchmarking and when accuracy is critical.

    BALANCED: Selective LLM use (jury verdict + key decisions).
              Good accuracy with moderate cost. Default for production.

    FAST: Heuristic-only evaluation. Lowest cost, maintains backwards compatibility.
          Use for development, testing, or cost-sensitive deployments.
    """

    QUALITY = "quality"
    """LLM-native evaluation for maximum accuracy."""

    BALANCED = "balanced"
    """Selective LLM use for good accuracy/cost balance."""

    FAST = "fast"
    """Heuristic-only for minimum cost."""


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
    evaluation_mode: EvaluationMode = EvaluationMode.BALANCED
    """Evaluation mode: QUALITY (LLM), BALANCED (selective LLM), FAST (heuristic)."""
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


class ContentType(str, Enum):
    """Type of content in a multimodal message."""

    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"


class ContentPart(BaseModel):
    """A part of multimodal message content.

    Supports text, images, and documents for multimodal debates.

    Example:
        ```python
        # Text content
        text_part = ContentPart(type=ContentType.TEXT, text="This is text")

        # Image from URL
        image_part = ContentPart(
            type=ContentType.IMAGE,
            url="https://example.com/chart.png",
            media_type="image/png",
        )

        # Image from base64 data
        image_data = ContentPart(
            type=ContentType.IMAGE,
            data=b"...",  # base64 encoded
            media_type="image/jpeg",
        )

        # PDF document
        doc_part = ContentPart(
            type=ContentType.DOCUMENT,
            data=b"...",
            media_type="application/pdf",
            filename="evidence.pdf",
        )
        ```
    """

    type: ContentType
    text: str | None = None
    """Text content (for TEXT type)."""

    data: bytes | None = None
    """Binary content for IMAGE/DOCUMENT (base64 encoded)."""

    url: str | None = None
    """URL reference for remote content."""

    media_type: str | None = None
    """MIME type (e.g., 'image/png', 'application/pdf')."""

    filename: str | None = None
    """Original filename for documents."""

    alt_text: str | None = None
    """Alternative text description for images."""

    @property
    def is_text(self) -> bool:
        """Check if this is text content."""
        return self.type == ContentType.TEXT

    @property
    def is_image(self) -> bool:
        """Check if this is image content."""
        return self.type == ContentType.IMAGE

    @property
    def is_document(self) -> bool:
        """Check if this is document content."""
        return self.type == ContentType.DOCUMENT

    def get_text(self) -> str:
        """Get text representation of content.

        Returns:
            Text content or description.
        """
        if self.text:
            return self.text
        if self.alt_text:
            return f"[Image: {self.alt_text}]"
        if self.filename:
            return f"[Document: {self.filename}]"
        return f"[{self.type.value}]"


class Message(BaseModel):
    """A message in an LLM conversation.

    Supports both simple text and multimodal content.

    Example:
        ```python
        # Simple text message
        msg = Message(role="user", content="What is AI?")

        # Multimodal message with image
        msg = Message(
            role="user",
            content="Analyze this chart",
            parts=[
                ContentPart(type=ContentType.TEXT, text="Analyze this chart"),
                ContentPart(
                    type=ContentType.IMAGE,
                    url="https://example.com/chart.png",
                    media_type="image/png",
                ),
            ],
        )
        ```
    """

    role: Literal["system", "user", "assistant"]
    content: str
    """Primary text content of the message."""

    name: str | None = None
    """Optional name for multi-agent scenarios."""

    parts: list[ContentPart] | None = None
    """Multimodal content parts (if any)."""

    @property
    def is_multimodal(self) -> bool:
        """Check if message contains multimodal content."""
        return bool(self.parts and any(p.type != ContentType.TEXT for p in self.parts))

    @property
    def text_content(self) -> str:
        """Get all text content from message.

        Combines primary content and text parts.
        """
        if not self.parts:
            return self.content

        texts = [self.content] if self.content else []
        for part in self.parts:
            if part.text:
                texts.append(part.text)
        return " ".join(texts)

    @property
    def images(self) -> list[ContentPart]:
        """Get all image parts."""
        if not self.parts:
            return []
        return [p for p in self.parts if p.type == ContentType.IMAGE]

    @property
    def documents(self) -> list[ContentPart]:
        """Get all document parts."""
        if not self.parts:
            return []
        return [p for p in self.parts if p.type == ContentType.DOCUMENT]

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
    agent_feedback: dict[str, str] = Field(default_factory=dict)
    """Performance feedback per agent (from FeedbackSynthesizer)."""


# =============================================================================
# Causal Analysis Types (CausalGraph v2)
# =============================================================================


class FallacyType(str, Enum):
    """Types of causal reasoning fallacies."""

    POST_HOC = "post_hoc"
    """A occurred before B, therefore A caused B."""

    FALSE_CAUSE = "false_cause"
    """Treating correlation as causation."""

    SLIPPERY_SLOPE = "slippery_slope"
    """Unwarranted chain of consequences."""

    CIRCULAR_REASONING = "circular_reasoning"
    """A because B because A."""

    APPEAL_TO_CONSEQUENCE = "appeal_to_consequence"
    """Something is true because of its consequences."""

    HASTY_GENERALIZATION = "hasty_generalization"
    """Conclusion from insufficient evidence."""

    SINGLE_CAUSE = "single_cause"
    """Ignoring multiple contributing factors."""


class CircularReasoningResult(BaseModel):
    """Result of circular reasoning detection."""

    cycle: list[str]
    """Node IDs forming the cycle."""
    argument_ids: list[str] = Field(default_factory=list)
    """Arguments involved in the cycle."""
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    """Severity of the circular reasoning (0-1)."""


class WeakLinkResult(BaseModel):
    """Result of weak link analysis."""

    source: str
    """Source node of the weak link."""
    target: str
    """Target node of the weak link."""
    strength: float
    """Current strength of the link."""
    weakness_score: float
    """Weakness score (1 - strength)."""
    attack_suggestions: list[str] = Field(default_factory=list)
    """Suggested attack strategies."""
    argument_ids: list[str] = Field(default_factory=list)
    """Arguments that established this link."""


class ContradictionResult(BaseModel):
    """Result of contradiction detection."""

    claim_a_source: str
    """Source of first claim."""
    claim_a_target: str
    """Target of first claim."""
    claim_a_type: str
    """Type of first claim (causes, prevents, etc.)."""
    claim_b_source: str
    """Source of second claim."""
    claim_b_target: str
    """Target of second claim."""
    claim_b_type: str
    """Type of second claim."""
    agents: list[str] = Field(default_factory=list)
    """Agents who made these claims."""
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    """Severity of the contradiction."""
    explanation: str | None = None
    """Explanation of why these are contradictory."""


class ArgumentStrengthScore(BaseModel):
    """Strength score for an argument based on causal support."""

    argument_id: str
    overall_score: float = Field(ge=0.0, le=1.0)
    """Overall strength score."""
    causal_support: float = Field(ge=0.0, le=1.0)
    """How well-supported by causal graph."""
    evidence_backing: float = Field(ge=0.0, le=1.0)
    """Evidence supporting causal claims."""
    vulnerability: float = Field(ge=0.0, le=1.0)
    """Exposure to attack (weak links)."""
    critical_dependencies: list[str] = Field(default_factory=list)
    """Node IDs this argument critically depends on."""


class CriticalNodeResult(BaseModel):
    """Result of critical node analysis."""

    node_id: str
    label: str
    centrality_score: float = Field(ge=0.0, le=1.0)
    """Betweenness centrality score."""
    dependent_arguments: list[str] = Field(default_factory=list)
    """Arguments that depend on this node."""
    impact_if_challenged: float = Field(ge=0.0, le=1.0)
    """Expected impact if this node is successfully challenged."""
    in_degree: int = 0
    """Number of incoming edges."""
    out_degree: int = 0
    """Number of outgoing edges."""


class ReasoningGap(BaseModel):
    """A gap in the causal reasoning chain."""

    start_node: str
    """Starting node of the gap."""
    end_node: str
    """Ending node (expected target)."""
    gap_type: str
    """Type of gap: 'missing_link', 'weak_chain', 'unsupported_conclusion'."""
    severity: float = Field(ge=0.0, le=1.0)
    """Severity of the gap."""
    suggested_bridges: list[str] = Field(default_factory=list)
    """Suggested intermediate nodes to bridge the gap."""


class FallacyResult(BaseModel):
    """Result of fallacy detection."""

    fallacy_type: FallacyType
    description: str
    """Human-readable description of the fallacy."""
    evidence: list[str] = Field(default_factory=list)
    """Evidence supporting the detection."""
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    """Severity of the fallacy."""
    affected_links: list[str] = Field(default_factory=list)
    """IDs of affected causal links."""
    affected_arguments: list[str] = Field(default_factory=list)
    """IDs of affected arguments."""


class AttackTarget(BaseModel):
    """A potential target for attacking opponent's argument."""

    source: str
    """Source node of vulnerable link."""
    target: str
    """Target node of vulnerable link."""
    vulnerability_score: float = Field(ge=0.0, le=1.0)
    """How vulnerable this link is to attack."""
    attack_strategies: list[str] = Field(default_factory=list)
    """Suggested attack strategies."""
    expected_impact: float = Field(ge=0.0, le=1.0)
    """Expected impact if attack succeeds."""
    priority: int = Field(default=0, ge=0)
    """Priority rank (0 = highest priority)."""


class ReinforcementSuggestion(BaseModel):
    """Suggestion for reinforcing a weak link in own argument."""

    source: str
    """Source node of link to reinforce."""
    target: str
    """Target node of link to reinforce."""
    current_strength: float = Field(ge=0.0, le=1.0)
    """Current strength of the link."""
    suggested_evidence: list[str] = Field(default_factory=list)
    """Types of evidence that would strengthen this link."""
    suggested_mechanisms: list[str] = Field(default_factory=list)
    """Mechanisms to explain the causal relationship."""
    priority: float = Field(ge=0.0, le=1.0)
    """Priority for reinforcement (higher = more urgent)."""


class CausalAnalysisResult(BaseModel):
    """Complete result of causal graph analysis."""

    has_circular_reasoning: bool = False
    circular_chains: list[CircularReasoningResult] = Field(default_factory=list)
    weak_links: list[WeakLinkResult] = Field(default_factory=list)
    contradictions: list[ContradictionResult] = Field(default_factory=list)
    critical_nodes: list[CriticalNodeResult] = Field(default_factory=list)
    reasoning_gaps: list[ReasoningGap] = Field(default_factory=list)
    fallacies: list[FallacyResult] = Field(default_factory=list)
    overall_coherence: float = Field(default=1.0, ge=0.0, le=1.0)
    """Overall coherence score of the causal graph."""
    argument_strengths: dict[str, ArgumentStrengthScore] = Field(default_factory=dict)
    """Strength scores by argument ID."""


class GraphSnapshot(BaseModel):
    """Snapshot of graph state at a point in time."""

    round: int
    turn: int
    node_count: int
    edge_count: int
    nodes: list[str] = Field(default_factory=list)
    """Node IDs at this point."""
    edges: list[tuple[str, str]] = Field(default_factory=list)
    """Edge pairs (source, target) at this point."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Streaming Types
# =============================================================================


class StreamEventType(str, Enum):
    """Types of streaming events during debate execution."""

    DEBATE_START = "debate_start"
    """Debate is starting."""

    ROUND_START = "round_start"
    """A new round is starting."""

    TURN_START = "turn_start"
    """An agent's turn is starting."""

    CHUNK = "chunk"
    """A chunk of argument content."""

    ARGUMENT_COMPLETE = "argument_complete"
    """An argument has been fully generated."""

    EVALUATION_START = "evaluation_start"
    """Evaluation of an argument is starting."""

    EVALUATION_COMPLETE = "evaluation_complete"
    """Evaluation of an argument is complete."""

    TURN_COMPLETE = "turn_complete"
    """An agent's turn is complete."""

    ROUND_COMPLETE = "round_complete"
    """A round is complete."""

    SAFETY_CHECK = "safety_check"
    """Safety monitor is checking."""

    SAFETY_ALERT = "safety_alert"
    """Safety monitor raised an alert."""

    VERDICT = "verdict"
    """Jury verdict is being delivered."""

    DEBATE_END = "debate_end"
    """Debate has ended."""

    ERROR = "error"
    """An error occurred."""


class StreamEvent(BaseModel):
    """An event emitted during streaming debate execution."""

    event_type: StreamEventType
    """Type of the event."""

    agent: str | None = None
    """Agent associated with this event (if applicable)."""

    content: str | None = None
    """Content chunk for CHUNK events."""

    argument: "Argument | None" = None
    """Complete argument for ARGUMENT_COMPLETE events."""

    turn: "Turn | None" = None
    """Complete turn for TURN_COMPLETE events."""

    evaluation: "ArgumentEvaluation | None" = None
    """Evaluation for EVALUATION_COMPLETE events."""

    safety_result: "SafetyResult | None" = None
    """Safety result for SAFETY_CHECK/SAFETY_ALERT events."""

    verdict: "Verdict | None" = None
    """Verdict for VERDICT events."""

    round_num: int | None = None
    """Round number for round-related events."""

    turn_num: int | None = None
    """Turn number within the round."""

    error: str | None = None
    """Error message for ERROR events."""

    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    """Additional metadata."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    """When the event occurred."""

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


# =============================================================================
# Hierarchical Debate Types
# =============================================================================


class HierarchyLevel(str, Enum):
    """Level in a hierarchical debate structure."""

    ROOT = "root"
    """Top-level parent debate."""

    BRANCH = "branch"
    """Intermediate sub-debate."""

    LEAF = "leaf"
    """Terminal sub-debate with no further decomposition."""


class SubDebateSpec(BaseModel):
    """Specification for a sub-debate within a hierarchical debate."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    """Unique identifier for the sub-debate."""

    aspect: str
    """The specific aspect or sub-topic to debate."""

    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    """Weight of this sub-debate in final aggregation."""

    description: str | None = None
    """Optional description of what this sub-debate should address."""

    rounds_override: int | None = None
    """Override number of rounds for this sub-debate."""

    agent_config_overrides: dict[str, dict] = Field(default_factory=dict)
    """Per-agent configuration overrides: {agent_name: {config_key: value}}."""


class HierarchicalContext(BaseModel):
    """Context passed to sub-debates in a hierarchical structure."""

    parent_topic: str
    """The parent debate's topic."""

    parent_verdict: "Verdict | None" = None
    """Parent's verdict if this is a refinement round."""

    sibling_verdicts: list["Verdict"] = Field(default_factory=list)
    """Verdicts from sibling sub-debates."""

    sibling_topics: list[str] = Field(default_factory=list)
    """Topics of sibling sub-debates."""

    depth: int = Field(default=0, ge=0)
    """Current depth in the hierarchy (0 = root)."""

    max_depth: int = Field(default=2, ge=1)
    """Maximum allowed depth."""

    path: list[str] = Field(default_factory=list)
    """Path from root to current node (topic names)."""


class CompoundVerdict(BaseModel):
    """Verdict from a hierarchical debate with sub-verdicts."""

    final_decision: str
    """The aggregated final decision."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Confidence in the aggregated verdict."""

    reasoning: str
    """Explanation of how the verdict was reached."""

    sub_verdicts: list["Verdict"] = Field(default_factory=list)
    """Verdicts from each sub-debate."""

    sub_topics: list[str] = Field(default_factory=list)
    """Topics of each sub-debate."""

    aggregation_method: str = "weighted_average"
    """Method used to aggregate sub-verdicts."""

    aggregation_weights: dict[str, float] = Field(default_factory=dict)
    """Weights used for each sub-debate."""

    depth: int = 0
    """Depth at which this verdict was produced."""


class DecompositionStrategy(str, Enum):
    """Strategy for decomposing topics."""

    LLM = "llm"
    """Use LLM to decompose topics."""

    RULE_BASED = "rule_based"
    """Use rule-based decomposition."""

    HYBRID = "hybrid"
    """Combine LLM and rules."""

    MANUAL = "manual"
    """Manually specified decomposition."""


class AggregationMethod(str, Enum):
    """Method for aggregating sub-verdicts."""

    WEIGHTED_AVERAGE = "weighted_average"
    """Weight verdicts by sub-debate importance."""

    MAJORITY_VOTE = "majority_vote"
    """Simple majority vote."""

    CONFIDENCE_WEIGHTED = "confidence_weighted"
    """Weight by verdict confidence."""

    UNANIMOUS = "unanimous"
    """Require unanimous agreement."""

    WEIGHTED_MAJORITY = "weighted_majority"
    """Majority vote with weights."""


# =============================================================================
# Formal Verification Types
# =============================================================================


class VerificationRuleType(str, Enum):
    """Type of verification rule."""

    CAUSAL_CHAIN = "causal_chain"
    """Verify causal reasoning chains are valid."""

    CITATION = "citation"
    """Verify citations and references."""

    LOGICAL_CONSISTENCY = "logical_consistency"
    """Check for logical contradictions."""

    EVIDENCE_SUPPORT = "evidence_support"
    """Verify claims are supported by evidence."""

    FALLACY_FREE = "fallacy_free"
    """Check for logical fallacies."""


class VerificationRule(BaseModel):
    """A single verification rule.

    Example:
        ```python
        rule = VerificationRule(
            rule_type=VerificationRuleType.CAUSAL_CHAIN,
            enabled=True,
            severity=1.0,
            config={"min_chain_length": 2},
        )
        ```
    """

    rule_type: VerificationRuleType
    enabled: bool = True
    severity: float = Field(default=1.0, ge=0.0, le=1.0)
    """Weight of this rule in the overall score."""
    config: dict[str, Any] = Field(default_factory=dict)
    """Rule-specific configuration."""


class VerificationSpec(BaseModel):
    """Specification for argument verification.

    Example:
        ```python
        spec = VerificationSpec(
            rules=[
                VerificationRule(rule_type=VerificationRuleType.CAUSAL_CHAIN),
                VerificationRule(rule_type=VerificationRuleType.CITATION),
            ],
            strict_mode=False,
            min_score=0.6,
        )
        ```
    """

    rules: list[VerificationRule] = Field(default_factory=list)
    strict_mode: bool = False
    """If True, fail on any violation."""
    min_score: float = Field(default=0.6, ge=0.0, le=1.0)
    """Minimum score to pass verification."""


class VerificationViolation(BaseModel):
    """A single verification violation."""

    description: str
    """Human-readable description."""
    location: str | None = None
    """Location in the argument (if applicable)."""
    severity: float = Field(default=1.0, ge=0.0, le=1.0)
    """Severity of the violation."""
    suggestion: str | None = None
    """Suggested fix."""


class VerificationResult(BaseModel):
    """Result of applying a single verification rule.

    Example:
        ```python
        result = VerificationResult(
            rule_type=VerificationRuleType.CAUSAL_CHAIN,
            passed=True,
            score=0.85,
            violations=[],
        )
        ```
    """

    rule_type: VerificationRuleType
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    violations: list[VerificationViolation] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class VerificationReport(BaseModel):
    """Complete verification report for an argument.

    Example:
        ```python
        report = VerificationReport(
            overall_passed=True,
            overall_score=0.85,
            results=[...],
            argument_id="arg-123",
        )
        ```
    """

    overall_passed: bool
    overall_score: float = Field(ge=0.0, le=1.0)
    results: list[VerificationResult] = Field(default_factory=list)
    argument_id: str
    summary: str = ""
    """Human-readable summary of verification results."""
