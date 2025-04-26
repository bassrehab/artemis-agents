"""
ARTEMIS LangGraph Integration

Provides LangGraph node for ARTEMIS debates.
Enables using structured multi-agent debates within LangGraph state machines.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, Field

from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.types import DebateConfig, DebateResult, Turn
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


class DebatePhase(str, Enum):
    """Phases of a LangGraph debate workflow."""

    SETUP = "setup"
    """Initial setup phase."""

    OPENING = "opening"
    """Opening statements."""

    DEBATE = "debate"
    """Main debate rounds."""

    CLOSING = "closing"
    """Closing arguments."""

    DELIBERATION = "deliberation"
    """Jury deliberation."""

    COMPLETE = "complete"
    """Debate finished."""


class DebateNodeState(TypedDict, total=False):
    """State schema for LangGraph debate node."""

    topic: str
    """The debate topic."""

    pro_position: str
    """Pro agent's position."""

    con_position: str
    """Con agent's position."""

    rounds: int
    """Total rounds."""

    current_round: int
    """Current round number."""

    phase: str
    """Current debate phase."""

    transcript: list[dict]
    """Debate transcript."""

    verdict: dict | None
    """Final verdict."""

    scores: dict[str, float]
    """Agent scores."""

    metadata: dict
    """Additional metadata."""


class DebateNodeConfig(BaseModel):
    """Configuration for debate node."""

    model: str = Field(default="gpt-4o", description="LLM model to use.")
    default_rounds: int = Field(default=3, ge=1, le=10)
    enable_safety: bool = Field(default=True)
    safety_monitors: list[Any] = Field(default_factory=list)


@dataclass
class DebateContext:
    """Context maintained across node executions."""

    debate: Debate | None = None
    agents: dict[str, Agent] = field(default_factory=dict)
    result: DebateResult | None = None


class ArtemisDebateNode:
    """
    LangGraph node for running ARTEMIS debates.

    Designed for integration with LangGraph state machines, providing
    flexible debate execution as nodes in a larger workflow.

    Example:
        >>> from langgraph.graph import StateGraph
        >>> from artemis.integrations.langgraph import ArtemisDebateNode
        >>>
        >>> node = ArtemisDebateNode(model="gpt-4o")
        >>> workflow = StateGraph(DebateNodeState)
        >>> workflow.add_node("debate", node.run_debate)
        >>> workflow.add_edge("start", "debate")

    Step-by-step execution:
        >>> workflow.add_node("setup", node.setup)
        >>> workflow.add_node("run_round", node.run_round)
        >>> workflow.add_node("finalize", node.finalize)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        config: DebateNodeConfig | None = None,
        debate_config: DebateConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the debate node.

        Args:
            model: LLM model to use.
            config: Node configuration.
            debate_config: Debate configuration.
            **kwargs: Additional configuration.
        """
        self.model = model
        self.node_config = config or DebateNodeConfig(model=model)
        self.debate_config = debate_config or DebateConfig()
        self.extra_config = kwargs

        # Context for multi-step execution
        self._contexts: dict[str, DebateContext] = {}

        logger.info(
            "ArtemisDebateNode initialized",
            model=model,
        )

    async def run_debate(self, state: DebateNodeState) -> DebateNodeState:
        """
        Run a complete debate as a single node.

        This is the simplest integration - runs the entire debate
        and returns updated state.

        Args:
            state: Current node state.

        Returns:
            Updated state with debate results.
        """
        topic = state.get("topic", "")
        if not topic:
            return {
                **state,
                "phase": DebatePhase.COMPLETE.value,
                "verdict": {"decision": "error", "reasoning": "No topic provided"},
            }

        logger.info("Running complete debate", topic=topic[:50])

        # Create agents
        pro_agent = Agent(
            name="pro_agent",
            model=self.model,
            position=state.get("pro_position", "supports the topic"),
        )
        con_agent = Agent(
            name="con_agent",
            model=self.model,
            position=state.get("con_position", "opposes the topic"),
        )

        # Create and run debate
        rounds = state.get("rounds", self.node_config.default_rounds)
        debate = Debate(
            topic=topic,
            agents=[pro_agent, con_agent],
            rounds=rounds,
            config=self.debate_config,
        )

        debate.assign_positions({
            "pro_agent": state.get("pro_position", "supports the topic"),
            "con_agent": state.get("con_position", "opposes the topic"),
        })

        result = await debate.run()

        # Format output
        return self._format_state(state, result)

    async def setup(self, state: DebateNodeState) -> DebateNodeState:
        """
        Setup phase - initialize debate.

        For step-by-step execution workflows.

        Args:
            state: Current node state.

        Returns:
            Updated state with debate initialized.
        """
        topic = state.get("topic", "")
        debate_id = f"debate_{id(state)}"

        logger.debug("Setting up debate", debate_id=debate_id, topic=topic[:50])

        # Create agents
        pro_agent = Agent(
            name="pro_agent",
            model=self.model,
            position=state.get("pro_position", "supports the topic"),
        )
        con_agent = Agent(
            name="con_agent",
            model=self.model,
            position=state.get("con_position", "opposes the topic"),
        )

        # Create debate
        rounds = state.get("rounds", self.node_config.default_rounds)
        debate = Debate(
            topic=topic,
            agents=[pro_agent, con_agent],
            rounds=rounds,
            config=self.debate_config,
        )

        debate.assign_positions({
            "pro_agent": state.get("pro_position", "supports the topic"),
            "con_agent": state.get("con_position", "opposes the topic"),
        })

        # Store context
        self._contexts[debate_id] = DebateContext(
            debate=debate,
            agents={"pro": pro_agent, "con": con_agent},
        )

        return {
            **state,
            "phase": DebatePhase.SETUP.value,
            "current_round": 0,
            "transcript": [],
            "metadata": {"debate_id": debate_id},
        }

    async def run_round(self, state: DebateNodeState) -> DebateNodeState:
        """
        Run a single debate round.

        For step-by-step execution workflows.

        Args:
            state: Current node state.

        Returns:
            Updated state with round results.
        """
        debate_id = state.get("metadata", {}).get("debate_id")
        if not debate_id or debate_id not in self._contexts:
            return {
                **state,
                "phase": DebatePhase.COMPLETE.value,
                "verdict": {"decision": "error", "reasoning": "No active debate"},
            }

        context = self._contexts[debate_id]
        debate = context.debate

        if debate is None:
            return {
                **state,
                "phase": DebatePhase.COMPLETE.value,
            }

        logger.debug(
            "Running debate round",
            debate_id=debate_id,
            round=debate.current_round,
        )

        # Run single round
        round_turns = await debate.run_single_round()

        # Update state
        current_round = debate.current_round
        phase = (
            DebatePhase.DEBATE.value
            if current_round < state.get("rounds", 3)
            else DebatePhase.CLOSING.value
        )

        transcript = state.get("transcript", [])
        transcript.extend([self._turn_to_dict(t) for t in round_turns])

        return {
            **state,
            "phase": phase,
            "current_round": current_round,
            "transcript": transcript,
        }

    async def finalize(self, state: DebateNodeState) -> DebateNodeState:
        """
        Finalize the debate and get verdict.

        For step-by-step execution workflows.

        Args:
            state: Current node state.

        Returns:
            Updated state with final verdict.
        """
        debate_id = state.get("metadata", {}).get("debate_id")
        if not debate_id or debate_id not in self._contexts:
            return {
                **state,
                "phase": DebatePhase.COMPLETE.value,
            }

        context = self._contexts[debate_id]
        debate = context.debate

        if debate is None:
            return {
                **state,
                "phase": DebatePhase.COMPLETE.value,
            }

        # Get final result
        result = await debate.run()
        context.result = result

        # Clean up
        del self._contexts[debate_id]

        return self._format_state(state, result)

    def _format_state(
        self,
        state: DebateNodeState,
        result: DebateResult,
    ) -> DebateNodeState:
        """Format debate result into state."""
        transcript = [self._turn_to_dict(t) for t in result.transcript]

        scores: dict[str, list[float]] = {}
        for turn in result.transcript:
            if turn.evaluation:
                if turn.agent not in scores:
                    scores[turn.agent] = []
                scores[turn.agent].append(turn.evaluation.total_score)

        avg_scores = {
            agent: (sum(s) / len(s) if s else 0.0)
            for agent, s in scores.items()
        }

        return {
            **state,
            "phase": DebatePhase.COMPLETE.value,
            "current_round": result.metadata.total_rounds,
            "transcript": transcript,
            "verdict": {
                "decision": result.verdict.decision,
                "confidence": result.verdict.confidence,
                "reasoning": result.verdict.reasoning,
                "unanimous": result.verdict.unanimous,
            },
            "scores": avg_scores,
            "metadata": {
                **state.get("metadata", {}),
                "total_turns": len(result.transcript),
                "started_at": (
                    result.metadata.started_at.isoformat()
                    if result.metadata.started_at
                    else None
                ),
                "ended_at": (
                    result.metadata.ended_at.isoformat()
                    if result.metadata.ended_at
                    else None
                ),
            },
        }

    def _turn_to_dict(self, turn: Turn) -> dict:
        """Convert Turn to dictionary."""
        return {
            "id": turn.id,
            "round": turn.round,
            "sequence": turn.sequence,
            "agent": turn.agent,
            "content": turn.argument.content,
            "level": turn.argument.level.value,
            "score": (
                turn.evaluation.total_score if turn.evaluation else None
            ),
            "timestamp": turn.timestamp.isoformat(),
        }

    def get_routing_function(self) -> Callable[[DebateNodeState], str]:
        """
        Get a routing function for conditional edges.

        Returns:
            Function that returns next node based on state.

        Example:
            >>> workflow.add_conditional_edges(
            ...     "run_round",
            ...     node.get_routing_function(),
            ...     {
            ...         "continue": "run_round",
            ...         "finalize": "finalize",
            ...     }
            ... )
        """
        def router(state: DebateNodeState) -> str:
            current = state.get("current_round", 0)
            total = state.get("rounds", 3)

            if current >= total:
                return "finalize"
            return "continue"

        return router

    def as_langgraph_node(self) -> Callable:
        """
        Get the main node function for LangGraph.

        Returns:
            Async function suitable for LangGraph node.
        """
        return self.run_debate

    def __repr__(self) -> str:
        return f"ArtemisDebateNode(model={self.model!r})"


def create_debate_workflow(
    model: str = "gpt-4o",
    step_by_step: bool = False,
) -> Any:
    """
    Create a LangGraph workflow for ARTEMIS debates.

    Args:
        model: LLM model to use.
        step_by_step: Whether to create step-by-step workflow.

    Returns:
        Compiled LangGraph workflow.

    Raises:
        ImportError: If langgraph is not installed.
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as e:
        raise ImportError(
            "langgraph is required for LangGraph integration. "
            "Install with: pip install langgraph"
        ) from e

    node = ArtemisDebateNode(model=model)
    workflow = StateGraph(DebateNodeState)

    if step_by_step:
        # Step-by-step workflow
        workflow.add_node("setup", node.setup)
        workflow.add_node("run_round", node.run_round)
        workflow.add_node("finalize", node.finalize)

        workflow.set_entry_point("setup")
        workflow.add_edge("setup", "run_round")
        workflow.add_conditional_edges(
            "run_round",
            node.get_routing_function(),
            {
                "continue": "run_round",
                "finalize": "finalize",
            },
        )
        workflow.add_edge("finalize", END)
    else:
        # Single-node workflow
        workflow.add_node("debate", node.run_debate)
        workflow.set_entry_point("debate")
        workflow.add_edge("debate", END)

    return workflow.compile()
