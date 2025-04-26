"""
ARTEMIS LangChain Integration

Provides LangChain Tool wrapper for ARTEMIS debates.
Enables using structured multi-agent debates within LangChain chains and agents.
"""

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.types import DebateConfig, DebateResult
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


class DebateInput(BaseModel):
    """Input schema for ARTEMIS debate tool."""

    topic: str = Field(
        ...,
        description="The debate topic or question to analyze.",
    )
    pro_position: str = Field(
        default="supports the topic",
        description="Position for the pro-side agent.",
    )
    con_position: str = Field(
        default="opposes the topic",
        description="Position for the con-side agent.",
    )
    rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of debate rounds (1-10).",
    )


class DebateOutput(BaseModel):
    """Output schema for ARTEMIS debate tool."""

    topic: str = Field(..., description="The debate topic.")
    verdict: str = Field(..., description="Final verdict (pro/con/draw).")
    confidence: float = Field(..., description="Confidence in verdict (0-1).")
    reasoning: str = Field(..., description="Reasoning for the verdict.")
    pro_score: float = Field(..., description="Average score for pro agent.")
    con_score: float = Field(..., description="Average score for con agent.")
    total_turns: int = Field(..., description="Total debate turns.")
    summary: str = Field(..., description="Brief summary of the debate.")


class ArtemisDebateTool:
    """
    LangChain-compatible tool for running ARTEMIS debates.

    Can be used as a standalone tool or integrated into LangChain agents
    and chains for structured multi-agent debate analysis.

    Example:
        >>> from artemis.integrations.langchain import ArtemisDebateTool
        >>> tool = ArtemisDebateTool(model="gpt-4o")
        >>> result = tool.invoke({
        ...     "topic": "Should AI be regulated?",
        ...     "rounds": 3,
        ... })
        >>> print(result.verdict)

    With LangChain:
        >>> from langchain.agents import initialize_agent
        >>> tools = [ArtemisDebateTool().as_langchain_tool()]
        >>> agent = initialize_agent(tools, llm, agent="zero-shot-react")
    """

    name: str = "artemis_debate"
    description: str = (
        "Conducts a structured multi-agent debate on a given topic. "
        "Two AI agents argue opposing positions through multiple rounds. "
        "A jury evaluates arguments and delivers a verdict. "
        "Use for complex decision-making requiring balanced analysis."
    )

    def __init__(
        self,
        model: str = "gpt-4o",
        default_rounds: int = 3,
        config: DebateConfig | None = None,
        safety_monitors: list | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ARTEMIS debate tool.

        Args:
            model: LLM model to use for agents.
            default_rounds: Default number of debate rounds.
            config: Optional debate configuration.
            safety_monitors: Optional safety monitor instances.
            **kwargs: Additional configuration.
        """
        self.model = model
        self.default_rounds = default_rounds
        self.config = config or DebateConfig()
        self.safety_monitors = safety_monitors or []
        self.extra_config = kwargs

        logger.info(
            "ArtemisDebateTool initialized",
            model=model,
            default_rounds=default_rounds,
        )

    def invoke(self, input_data: dict | DebateInput) -> DebateOutput:
        """
        Run a debate synchronously.

        Args:
            input_data: Debate parameters.

        Returns:
            DebateOutput with verdict and analysis.
        """
        if isinstance(input_data, dict):
            input_data = DebateInput(**input_data)

        return asyncio.run(self.ainvoke(input_data))

    async def ainvoke(self, input_data: dict | DebateInput) -> DebateOutput:
        """
        Run a debate asynchronously.

        Args:
            input_data: Debate parameters.

        Returns:
            DebateOutput with verdict and analysis.
        """
        if isinstance(input_data, dict):
            input_data = DebateInput(**input_data)

        logger.info(
            "Starting debate",
            topic=input_data.topic[:50],
            rounds=input_data.rounds,
        )

        # Create agents
        pro_agent = Agent(
            name="pro_agent",
            model=self.model,
            position=input_data.pro_position,
        )
        con_agent = Agent(
            name="con_agent",
            model=self.model,
            position=input_data.con_position,
        )

        # Create and run debate
        debate = Debate(
            topic=input_data.topic,
            agents=[pro_agent, con_agent],
            rounds=input_data.rounds or self.default_rounds,
            config=self.config,
            **self.extra_config,
        )

        # Add safety monitors if configured
        for monitor in self.safety_monitors:
            if hasattr(monitor, "process"):
                debate.add_safety_monitor(monitor.process)

        debate.assign_positions({
            "pro_agent": input_data.pro_position,
            "con_agent": input_data.con_position,
        })

        result = await debate.run()

        return self._format_output(result, input_data)

    def _format_output(
        self,
        result: DebateResult,
        input_data: DebateInput,
    ) -> DebateOutput:
        """Format debate result as output."""
        scores = self._calculate_scores(result)
        summary = self._generate_summary(result)

        return DebateOutput(
            topic=input_data.topic,
            verdict=result.verdict.decision,
            confidence=result.verdict.confidence,
            reasoning=result.verdict.reasoning,
            pro_score=scores.get("pro_agent", 0.0),
            con_score=scores.get("con_agent", 0.0),
            total_turns=len(result.transcript),
            summary=summary,
        )

    def _calculate_scores(self, result: DebateResult) -> dict[str, float]:
        """Calculate average scores per agent."""
        scores: dict[str, list[float]] = {}

        for turn in result.transcript:
            if turn.evaluation:
                if turn.agent not in scores:
                    scores[turn.agent] = []
                scores[turn.agent].append(turn.evaluation.total_score)

        return {
            agent: (sum(s) / len(s) if s else 0.0)
            for agent, s in scores.items()
        }

    def _generate_summary(self, result: DebateResult) -> str:
        """Generate a brief debate summary."""
        verdict = result.verdict
        transcript = result.transcript

        if not transcript:
            return "No debate occurred."

        opening_pro = next(
            (t for t in transcript if t.agent == "pro_agent" and t.round == 0),
            None,
        )
        opening_con = next(
            (t for t in transcript if t.agent == "con_agent" and t.round == 0),
            None,
        )

        parts = [f"Debate on: {result.topic}"]

        if opening_pro:
            parts.append(f"Pro argued: {opening_pro.argument.content[:100]}...")
        if opening_con:
            parts.append(f"Con argued: {opening_con.argument.content[:100]}...")

        parts.append(
            f"After {len(transcript)} turns, verdict: {verdict.decision} "
            f"({verdict.confidence:.0%} confidence)"
        )

        return " ".join(parts)

    def as_langchain_tool(self) -> Any:
        """
        Convert to a LangChain Tool.

        Returns:
            LangChain Tool instance.

        Raises:
            ImportError: If langchain is not installed.
        """
        try:
            from langchain_core.tools import StructuredTool
        except ImportError as e:
            raise ImportError(
                "langchain-core is required for LangChain integration. "
                "Install with: pip install langchain-core"
            ) from e

        return StructuredTool.from_function(
            func=self.invoke,
            coroutine=self.ainvoke,
            name=self.name,
            description=self.description,
            args_schema=DebateInput,
        )

    def as_openai_function(self) -> dict:
        """
        Convert to OpenAI function calling format.

        Returns:
            OpenAI function definition dict.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The debate topic or question.",
                    },
                    "pro_position": {
                        "type": "string",
                        "description": "Position for the pro-side agent.",
                    },
                    "con_position": {
                        "type": "string",
                        "description": "Position for the con-side agent.",
                    },
                    "rounds": {
                        "type": "integer",
                        "description": "Number of debate rounds.",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["topic"],
            },
        }

    def __repr__(self) -> str:
        return (
            f"ArtemisDebateTool(model={self.model!r}, "
            f"default_rounds={self.default_rounds})"
        )


class QuickDebate:
    """
    Simplified interface for quick debates.

    Example:
        >>> result = QuickDebate.run("Should we use AI in healthcare?")
        >>> print(f"Verdict: {result.verdict}")
    """

    @staticmethod
    def run(
        topic: str,
        rounds: int = 3,
        model: str = "gpt-4o",
    ) -> DebateOutput:
        """
        Run a quick debate on a topic.

        Args:
            topic: The debate topic.
            rounds: Number of rounds.
            model: LLM model to use.

        Returns:
            DebateOutput with verdict.
        """
        tool = ArtemisDebateTool(model=model, default_rounds=rounds)
        return tool.invoke({"topic": topic, "rounds": rounds})

    @staticmethod
    async def arun(
        topic: str,
        rounds: int = 3,
        model: str = "gpt-4o",
    ) -> DebateOutput:
        """
        Run a quick debate asynchronously.

        Args:
            topic: The debate topic.
            rounds: Number of rounds.
            model: LLM model to use.

        Returns:
            DebateOutput with verdict.
        """
        tool = ArtemisDebateTool(model=model, default_rounds=rounds)
        return await tool.ainvoke({"topic": topic, "rounds": rounds})
