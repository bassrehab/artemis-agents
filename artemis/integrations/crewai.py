"""
ARTEMIS CrewAI Integration

Provides CrewAI Tool wrapper for ARTEMIS debates.
Enables using structured multi-agent debates within CrewAI crews.
"""

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


class DebateToolInput(BaseModel):
    """Input schema for ARTEMIS debate tool in CrewAI."""

    topic: str = Field(
        ...,
        description="The debate topic or question to analyze through structured argumentation.",
    )
    agents: list[AgentConfig] | None = Field(
        default=None,
        description="List of agent configurations. If not provided, uses default pro/con agents.",
    )
    rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of debate rounds (1-10). More rounds provide deeper analysis.",
    )
    # Backward compatibility for simple pro/con debates
    pro_position: str | None = Field(
        default=None,
        description="The position that the pro-side agent will argue for (used if agents not provided).",
    )
    con_position: str | None = Field(
        default=None,
        description="The position that the con-side agent will argue for (used if agents not provided).",
    )


class DebateToolOutput(BaseModel):
    """Output schema for ARTEMIS debate tool."""

    topic: str
    verdict: str
    confidence: float
    reasoning: str
    agent_scores: dict[str, float]
    key_arguments: list[str]
    recommendation: str


class ArtemisCrewTool:
    """
    CrewAI-compatible tool for running ARTEMIS debates.

    Integrates structured multi-agent debates into CrewAI workflows,
    enabling crews to leverage adversarial analysis for decision-making.

    Supports configurable agents - either simple pro/con or multi-agent.

    Example (simple pro/con):
        >>> from crewai import Agent, Task, Crew
        >>> from artemis.integrations.crewai import ArtemisCrewTool
        >>>
        >>> debate_tool = ArtemisCrewTool(model="gpt-4o")
        >>>
        >>> analyst = Agent(
        ...     role="Decision Analyst",
        ...     tools=[debate_tool.as_crewai_tool()],
        ... )

    Example (multi-agent):
        >>> result = tool.run(
        ...     topic="How should we approach AI safety?",
        ...     agents=[
        ...         {"name": "researcher", "role": "AI Researcher", "position": "focus on alignment"},
        ...         {"name": "ethicist", "role": "Ethicist", "position": "focus on ethics"},
        ...     ],
        ... )

    Direct usage:
        >>> tool = ArtemisCrewTool()
        >>> result = tool.run(
        ...     topic="Should we adopt microservices architecture?",
        ...     rounds=3,
        ... )
    """

    name: str = "artemis_structured_debate"
    description: str = (
        "Conducts a structured multi-agent debate on a given topic. "
        "Multiple AI agents argue different positions through multiple rounds, "
        "with a jury evaluating arguments and delivering a verdict. "
        "Use this tool when you need balanced analysis of complex decisions, "
        "policy questions, or trade-off evaluations. Returns a verdict with "
        "confidence score, key arguments from all sides, and a recommendation."
    )

    def __init__(
        self,
        model: str = "gpt-4o",
        default_rounds: int = 3,
        agents: list[Agent] | None = None,
        config: DebateConfig | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ARTEMIS CrewAI tool.

        Args:
            model: LLM model to use for debate agents.
            default_rounds: Default number of debate rounds.
            agents: Pre-configured agents (optional).
            config: Optional debate configuration.
            verbose: Whether to log detailed progress.
            **kwargs: Additional configuration.
        """
        self.model = model
        self.default_rounds = default_rounds
        self.default_agents = agents
        self.config = config or DebateConfig()
        self.verbose = verbose
        self.extra_config = kwargs

        logger.info(
            "ArtemisCrewTool initialized",
            model=model,
            default_rounds=default_rounds,
            pre_configured_agents=len(agents) if agents else 0,
        )

    def run(
        self,
        topic: str,
        agents: list[dict] | None = None,
        pro_position: str | None = None,
        con_position: str | None = None,
        rounds: int | None = None,
    ) -> str:
        """
        Run a debate and return formatted result string.

        This is the main entry point for CrewAI tool execution.

        Args:
            topic: The debate topic.
            agents: List of agent configurations (optional).
            pro_position: Pro-side position (for simple pro/con, if agents not provided).
            con_position: Con-side position (for simple pro/con, if agents not provided).
            rounds: Number of rounds (uses default if not specified).

        Returns:
            Formatted string with debate results.
        """
        result = asyncio.run(
            self._run_debate(topic, agents, pro_position, con_position, rounds)
        )
        return self._format_result_string(result)

    async def arun(
        self,
        topic: str,
        agents: list[dict] | None = None,
        pro_position: str | None = None,
        con_position: str | None = None,
        rounds: int | None = None,
    ) -> str:
        """
        Run a debate asynchronously.

        Args:
            topic: The debate topic.
            agents: List of agent configurations (optional).
            pro_position: Pro-side position (for simple pro/con).
            con_position: Con-side position (for simple pro/con).
            rounds: Number of rounds.

        Returns:
            Formatted string with debate results.
        """
        result = await self._run_debate(topic, agents, pro_position, con_position, rounds)
        return self._format_result_string(result)

    def _create_agents(
        self,
        agent_configs: list[dict] | None,
        pro_position: str | None,
        con_position: str | None,
    ) -> list[Agent]:
        """Create agents from configs or use defaults."""
        # Use pre-configured agents if available
        if self.default_agents:
            return self.default_agents

        # Create from input agent configs
        if agent_configs:
            return [
                Agent(
                    name=cfg["name"],
                    role=cfg["role"],
                    model=cfg.get("model") or self.model,
                )
                for cfg in agent_configs
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
        self,
        agent_configs: list[dict] | None,
        pro_position: str | None,
        con_position: str | None,
    ) -> dict[str, str]:
        """Get positions mapping for agents."""
        # From input agent configs
        if agent_configs:
            return {
                cfg["name"]: cfg.get("position", "")
                for cfg in agent_configs
                if cfg.get("position")
            }

        # From simple pro/con positions
        return {
            "pro_agent": pro_position or "supports the proposition",
            "con_agent": con_position or "opposes the proposition",
        }

    async def _run_debate(
        self,
        topic: str,
        agent_configs: list[dict] | None,
        pro_position: str | None,
        con_position: str | None,
        rounds: int | None,
    ) -> DebateToolOutput:
        """Execute the debate and return structured output."""
        if self.verbose:
            logger.info("Starting debate", topic=topic[:50])

        # Create agents
        agents = self._create_agents(agent_configs, pro_position, con_position)
        positions = self._get_positions(agent_configs, pro_position, con_position)

        # Create and run debate
        debate = Debate(
            topic=topic,
            agents=agents,
            rounds=rounds or self.default_rounds,
            config=self.config,
            **self.extra_config,
        )

        debate.assign_positions(positions)

        result = await debate.run()

        return self._build_output(result, topic)

    def _build_output(
        self,
        result: DebateResult,
        topic: str,
    ) -> DebateToolOutput:
        """Build structured output from debate result."""
        # Calculate scores
        scores = self._calculate_scores(result)

        # Extract key arguments
        key_arguments = self._extract_key_arguments(result)

        # Generate recommendation
        recommendation = self._generate_recommendation(result)

        return DebateToolOutput(
            topic=topic,
            verdict=result.verdict.decision,
            confidence=result.verdict.confidence,
            reasoning=result.verdict.reasoning,
            agent_scores=scores,
            key_arguments=key_arguments,
            recommendation=recommendation,
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

    def _extract_key_arguments(self, result: DebateResult) -> list[str]:
        """Extract key arguments from the debate."""
        key_args = []

        # Get opening statements
        for turn in result.transcript:
            if turn.round == 0:  # Opening statements
                content = turn.argument.content
                # Truncate to key point
                key_point = content[:200] + "..." if len(content) > 200 else content
                key_args.append(f"[{turn.agent}] {key_point}")

        # Get highest-scored arguments
        scored_turns = [
            (turn, turn.evaluation.total_score)
            for turn in result.transcript
            if turn.evaluation and turn.round > 0
        ]
        scored_turns.sort(key=lambda x: x[1], reverse=True)

        for turn, score in scored_turns[:2]:  # Top 2 arguments
            content = turn.argument.content
            key_point = content[:150] + "..." if len(content) > 150 else content
            key_args.append(f"[{turn.agent}] (score: {score:.2f}) {key_point}")

        return key_args

    def _generate_recommendation(self, result: DebateResult) -> str:
        """Generate a recommendation based on debate outcome."""
        verdict = result.verdict
        confidence = verdict.confidence

        if confidence > 0.7:
            strength = "strongly "
        elif confidence > 0.5:
            strength = ""
        else:
            strength = "cautiously "

        if verdict.decision in ["draw", "tie", "inconclusive"]:
            return f"The debate resulted in a balanced outcome. Both positions have merit. {verdict.reasoning[:100]}"
        else:
            return f"Based on the debate analysis, the evidence {strength}supports '{verdict.decision}'. {verdict.reasoning[:100]}"

    def _format_result_string(self, output: DebateToolOutput) -> str:
        """Format output as a string for CrewAI consumption."""
        lines = [
            f"DEBATE ANALYSIS: {output.topic}",
            "",
            f"VERDICT: {output.verdict.upper()}",
            f"CONFIDENCE: {output.confidence:.0%}",
            "",
            "AGENT SCORES:",
        ]

        for agent, score in output.agent_scores.items():
            lines.append(f"  {agent}: {score:.2f}")

        lines.extend([
            "",
            "KEY ARGUMENTS:",
        ])

        for arg in output.key_arguments:
            lines.append(f"  - {arg}")

        lines.extend([
            "",
            "REASONING:",
            f"  {output.reasoning}",
            "",
            "RECOMMENDATION:",
            f"  {output.recommendation}",
        ])

        return "\n".join(lines)

    def as_crewai_tool(self) -> Any:
        """
        Convert to a CrewAI Tool.

        Returns:
            CrewAI Tool instance.

        Raises:
            ImportError: If crewai is not installed.
        """
        try:
            from crewai.tools import BaseTool
        except ImportError as e:
            raise ImportError(
                "crewai is required for CrewAI integration. "
                "Install with: pip install crewai"
            ) from e

        tool_instance = self

        class ArtemisDebateTool(BaseTool):
            name: str = tool_instance.name
            description: str = tool_instance.description
            args_schema: type[BaseModel] = DebateToolInput

            def _run(
                self,
                topic: str,
                agents: list[dict] | None = None,
                pro_position: str | None = None,
                con_position: str | None = None,
                rounds: int = 3,
            ) -> str:
                return tool_instance.run(topic, agents, pro_position, con_position, rounds)

            async def _arun(
                self,
                topic: str,
                agents: list[dict] | None = None,
                pro_position: str | None = None,
                con_position: str | None = None,
                rounds: int = 3,
            ) -> str:
                return await tool_instance.arun(
                    topic, agents, pro_position, con_position, rounds
                )

        return ArtemisDebateTool()

    def as_function(self) -> dict:
        """
        Export as a function definition for function calling.

        Returns:
            Function definition dict.
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
                        "default": 3,
                    },
                },
                "required": ["topic"],
            },
        }

    def __repr__(self) -> str:
        return (
            f"ArtemisCrewTool(model={self.model!r}, "
            f"default_rounds={self.default_rounds})"
        )


class DebateAnalyzer:
    """
    High-level analyzer for complex decisions using debates.

    Provides a simplified interface for running multiple debates
    to analyze complex multi-faceted decisions.

    Example:
        >>> analyzer = DebateAnalyzer()
        >>> results = analyzer.analyze_decision(
        ...     decision="Should we migrate to cloud infrastructure?",
        ...     aspects=["cost", "security", "scalability"],
        ... )
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        rounds_per_debate: int = 2,
    ) -> None:
        """
        Initialize the analyzer.

        Args:
            model: LLM model to use.
            rounds_per_debate: Rounds for each sub-debate.
        """
        self.model = model
        self.rounds = rounds_per_debate
        self._tool = ArtemisCrewTool(model=model, default_rounds=rounds_per_debate)

    def analyze_decision(
        self,
        decision: str,
        aspects: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze a decision from multiple aspects.

        Args:
            decision: The main decision to analyze.
            aspects: Specific aspects to debate (optional).

        Returns:
            Analysis results with verdicts per aspect.
        """
        return asyncio.run(self._analyze_async(decision, aspects))

    async def _analyze_async(
        self,
        decision: str,
        aspects: list[str] | None,
    ) -> dict[str, Any]:
        """Async implementation of decision analysis."""
        if not aspects:
            # Single comprehensive debate
            result = await self._tool._run_debate(
                topic=decision,
                agent_configs=None,
                pro_position="recommends this course of action",
                con_position="recommends against this course of action",
                rounds=self.rounds,
            )
            return {
                "decision": decision,
                "overall_verdict": result.verdict,
                "confidence": result.confidence,
                "recommendation": result.recommendation,
            }

        # Multi-aspect analysis
        results = {}
        for aspect in aspects:
            topic = f"{decision} - focusing on {aspect}"
            result = await self._tool._run_debate(
                topic=topic,
                agent_configs=None,
                pro_position=f"argues {aspect} supports this decision",
                con_position=f"argues {aspect} opposes this decision",
                rounds=self.rounds,
            )
            results[aspect] = {
                "verdict": result.verdict,
                "confidence": result.confidence,
                "key_points": result.key_arguments[:2],
            }

        # Aggregate results
        pro_count = sum(1 for r in results.values() if r["verdict"] == "pro")
        con_count = sum(1 for r in results.values() if r["verdict"] == "con")

        return {
            "decision": decision,
            "aspects_analyzed": aspects,
            "aspect_results": results,
            "overall_verdict": "pro" if pro_count > con_count else "con",
            "verdict_distribution": {"pro": pro_count, "con": con_count},
        }
