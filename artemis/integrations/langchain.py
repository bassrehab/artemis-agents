"""LangChain tool wrapper for ARTEMIS debates."""

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.types import DebateConfig, DebateResult
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


class AgentConfig(BaseModel):
    """Configuration for a debate agent."""

    name: str = Field(..., description="Unique name for the agent.")
    role: str = Field(..., description="Role description for the agent.")
    position: str = Field(
        default="", description="Position the agent will argue for."
    )
    model: str | None = Field(
        default=None, description="Model override for this agent."
    )


class DebateInput(BaseModel):
    """Input schema for ARTEMIS debate tool."""

    topic: str = Field(
        ...,
        description="The debate topic or question to analyze.",
    )
    agents: list[AgentConfig] | None = Field(
        default=None,
        description="List of agent configurations. If not provided, uses default pro/con agents.",
    )
    rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of debate rounds (1-10).",
    )
    # Backward compatibility for simple pro/con debates
    pro_position: str | None = Field(
        default=None,
        description="Position for the pro-side agent (used if agents not provided).",
    )
    con_position: str | None = Field(
        default=None,
        description="Position for the con-side agent (used if agents not provided).",
    )


class DebateOutput(BaseModel):
    """Output schema for ARTEMIS debate tool."""

    topic: str = Field(..., description="The debate topic.")
    verdict: str = Field(..., description="Final verdict (winner agent name or draw).")
    confidence: float = Field(..., description="Confidence in verdict (0-1).")
    reasoning: str = Field(..., description="Reasoning for the verdict.")
    agent_scores: dict[str, float] = Field(
        default_factory=dict, description="Scores for each agent."
    )
    total_turns: int = Field(..., description="Total debate turns.")
    summary: str = Field(..., description="Brief summary of the debate.")


class ArtemisDebateTool:
    """LangChain-compatible tool for running ARTEMIS debates."""

    name: str = "artemis_debate"
    description: str = (
        "Conducts a structured multi-agent debate on a given topic. "
        "Multiple AI agents argue different positions through multiple rounds. "
        "A jury evaluates arguments and delivers a verdict. "
        "Use for complex decision-making requiring balanced analysis."
    )

    def __init__(
        self,
        model: str = "gpt-4o",
        default_rounds: int = 3,
        agents: list[Agent] | None = None,
        config: DebateConfig | None = None,
        safety_monitors: list | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.default_rounds = default_rounds
        self.default_agents = agents
        self.config = config or DebateConfig()
        self.safety_monitors = safety_monitors or []
        self.extra_config = kwargs

        logger.info(
            "ArtemisDebateTool initialized",
            model=model,
            default_rounds=default_rounds,
            pre_configured_agents=len(agents) if agents else 0,
        )

    def invoke(self, input_data):
        """Run a debate synchronously."""
        if isinstance(input_data, dict):
            input_data = DebateInput(**input_data)

        return asyncio.run(self.ainvoke(input_data))

    async def ainvoke(self, input_data):
        """Run a debate asynchronously."""
        if isinstance(input_data, dict):
            input_data = DebateInput(**input_data)

        logger.info(
            "Starting debate",
            topic=input_data.topic[:50],
            rounds=input_data.rounds,
        )

        # Create or use agents
        agents = self._create_agents(input_data)
        positions = self._get_positions(input_data, agents)

        # Create and run debate
        debate = Debate(
            topic=input_data.topic,
            agents=agents,
            rounds=input_data.rounds or self.default_rounds,
            config=self.config,
            **self.extra_config,
        )

        # Add safety monitors if configured
        for monitor in self.safety_monitors:
            if hasattr(monitor, "process"):
                debate.add_safety_monitor(monitor.process)

        # Assign positions
        debate.assign_positions(positions)

        result = await debate.run()

        return self._format_output(result, input_data)

    def _create_agents(self, input_data: DebateInput) -> list[Agent]:
        """Create agents from input or use defaults."""
        # Use pre-configured agents if available
        if self.default_agents:
            return self.default_agents

        # Create from input agent configs
        if input_data.agents:
            return [
                Agent(
                    name=agent_config.name,
                    role=agent_config.role,
                    model=agent_config.model or self.model,
                )
                for agent_config in input_data.agents
            ]

        # Fall back to default pro/con agents
        return [
            Agent(
                name="pro_agent",
                role="Debate advocate for the proposition",
                model=self.model,
            ),
            Agent(
                name="con_agent",
                role="Debate advocate against the proposition",
                model=self.model,
            ),
        ]

    def _get_positions(
        self, input_data: DebateInput, agents: list[Agent]
    ) -> dict[str, str]:
        """Get positions mapping for agents."""
        # From input agent configs
        if input_data.agents:
            return {
                agent_config.name: agent_config.position
                for agent_config in input_data.agents
                if agent_config.position
            }

        # From simple pro/con positions
        return {
            "pro_agent": input_data.pro_position or "supports the topic",
            "con_agent": input_data.con_position or "opposes the topic",
        }

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
            agent_scores=scores,
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

        # Get opening turns
        opening_turns = [t for t in transcript if t.round == 0]

        parts = [f"Debate on: {result.topic}"]

        for turn in opening_turns[:2]:  # First two opening statements
            content = turn.argument.content
            parts.append(f"{turn.agent} argued: {content[:100]}...")

        parts.append(
            f"After {len(transcript)} turns, verdict: {verdict.decision} "
            f"({verdict.confidence:.0%} confidence)"
        )

        return " ".join(parts)

    def as_langchain_tool(self):
        """Convert to a LangChain Tool."""
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

    def as_openai_function(self):
        """Convert to OpenAI function calling format."""
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
                    "agents": {
                        "type": "array",
                        "description": "List of agent configurations.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "role": {"type": "string"},
                                "position": {"type": "string"},
                            },
                            "required": ["name", "role"],
                        },
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
    """Simplified interface for quick debates."""

    @staticmethod
    def run(topic, rounds=3, model="gpt-4o", agents=None):
        """Run a quick debate on a topic."""
        tool = ArtemisDebateTool(model=model, default_rounds=rounds)
        return tool.invoke({"topic": topic, "rounds": rounds, "agents": agents})

    @staticmethod
    async def arun(topic, rounds=3, model="gpt-4o", agents=None):
        """Run a quick debate asynchronously."""
        tool = ArtemisDebateTool(model=model, default_rounds=rounds)
        return await tool.ainvoke({"topic": topic, "rounds": rounds, "agents": agents})
