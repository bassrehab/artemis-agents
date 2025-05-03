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


class DebateToolInput(BaseModel):
    """Input schema for ARTEMIS debate tool in CrewAI."""

    topic: str = Field(
        ...,
        description="The debate topic or question to analyze through structured argumentation.",
    )
    pro_position: str = Field(
        default="supports the proposition",
        description="The position that the pro-side agent will argue for.",
    )
    con_position: str = Field(
        default="opposes the proposition",
        description="The position that the con-side agent will argue for.",
    )
    rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of debate rounds (1-10). More rounds provide deeper analysis.",
    )


class DebateToolOutput(BaseModel):
    """Output schema for ARTEMIS debate tool."""

    topic: str
    verdict: str
    confidence: float
    reasoning: str
    pro_score: float
    con_score: float
    key_arguments: list[str]
    recommendation: str


class ArtemisCrewTool:
    """
    CrewAI-compatible tool for running ARTEMIS debates.

    Integrates structured multi-agent debates into CrewAI workflows,
    enabling crews to leverage adversarial analysis for decision-making.

    Example:
        >>> from crewai import Agent, Task, Crew
        >>> from artemis.integrations.crewai import ArtemisCrewTool
        >>>
        >>> debate_tool = ArtemisCrewTool(model="gpt-4o")
        >>>
        >>> analyst = Agent(
        ...     role="Decision Analyst",
        ...     tools=[debate_tool.as_crewai_tool()],
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
        "Two AI agents argue opposing positions through multiple rounds, "
        "with a jury evaluating arguments and delivering a verdict. "
        "Use this tool when you need balanced analysis of complex decisions, "
        "policy questions, or trade-off evaluations. Returns a verdict with "
        "confidence score, key arguments from both sides, and a recommendation."
    )

    def __init__(
        self,
        model: str = "gpt-4o",
        default_rounds: int = 3,
        config: DebateConfig | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ARTEMIS CrewAI tool.

        Args:
            model: LLM model to use for debate agents.
            default_rounds: Default number of debate rounds.
            config: Optional debate configuration.
            verbose: Whether to log detailed progress.
            **kwargs: Additional configuration.
        """
        self.model = model
        self.default_rounds = default_rounds
        self.config = config or DebateConfig()
        self.verbose = verbose
        self.extra_config = kwargs

        logger.info(
            "ArtemisCrewTool initialized",
            model=model,
            default_rounds=default_rounds,
        )

    def run(
        self,
        topic: str,
        pro_position: str = "supports the proposition",
        con_position: str = "opposes the proposition",
        rounds: int | None = None,
    ) -> str:
        """
        Run a debate and return formatted result string.

        This is the main entry point for CrewAI tool execution.

        Args:
            topic: The debate topic.
            pro_position: Pro-side position.
            con_position: Con-side position.
            rounds: Number of rounds (uses default if not specified).

        Returns:
            Formatted string with debate results.
        """
        result = asyncio.run(
            self._run_debate(topic, pro_position, con_position, rounds)
        )
        return self._format_result_string(result)

    async def arun(
        self,
        topic: str,
        pro_position: str = "supports the proposition",
        con_position: str = "opposes the proposition",
        rounds: int | None = None,
    ) -> str:
        """
        Run a debate asynchronously.

        Args:
            topic: The debate topic.
            pro_position: Pro-side position.
            con_position: Con-side position.
            rounds: Number of rounds.

        Returns:
            Formatted string with debate results.
        """
        result = await self._run_debate(topic, pro_position, con_position, rounds)
        return self._format_result_string(result)

    async def _run_debate(
        self,
        topic: str,
        pro_position: str,
        con_position: str,
        rounds: int | None,
    ) -> DebateToolOutput:
        """Execute the debate and return structured output."""
        if self.verbose:
            logger.info("Starting debate", topic=topic[:50])

        # Create agents
        pro_agent = Agent(
            name="pro_agent",
            model=self.model,
            position=pro_position,
        )
        con_agent = Agent(
            name="con_agent",
            model=self.model,
            position=con_position,
        )

        # Create and run debate
        debate = Debate(
            topic=topic,
            agents=[pro_agent, con_agent],
            rounds=rounds or self.default_rounds,
            config=self.config,
            **self.extra_config,
        )

        debate.assign_positions({
            "pro_agent": pro_position,
            "con_agent": con_position,
        })

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
            pro_score=scores.get("pro_agent", 0.0),
            con_score=scores.get("con_agent", 0.0),
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

        if verdict.decision == "pro":
            strength = "strongly " if confidence > 0.7 else ""
            return f"Based on the debate analysis, the evidence {strength}supports the proposition. {verdict.reasoning[:100]}"
        elif verdict.decision == "con":
            strength = "strongly " if confidence > 0.7 else ""
            return f"Based on the debate analysis, the evidence {strength}opposes the proposition. {verdict.reasoning[:100]}"
        else:
            return f"The debate resulted in a balanced outcome. Both positions have merit. {verdict.reasoning[:100]}"

    def _format_result_string(self, output: DebateToolOutput) -> str:
        """Format output as a string for CrewAI consumption."""
        lines = [
            f"DEBATE ANALYSIS: {output.topic}",
            "",
            f"VERDICT: {output.verdict.upper()}",
            f"CONFIDENCE: {output.confidence:.0%}",
            "",
            "SCORES:",
            f"  Pro Position: {output.pro_score:.2f}",
            f"  Con Position: {output.con_score:.2f}",
            "",
            "KEY ARGUMENTS:",
        ]

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
                pro_position: str = "supports the proposition",
                con_position: str = "opposes the proposition",
                rounds: int = 3,
            ) -> str:
                return tool_instance.run(topic, pro_position, con_position, rounds)

            async def _arun(
                self,
                topic: str,
                pro_position: str = "supports the proposition",
                con_position: str = "opposes the proposition",
                rounds: int = 3,
            ) -> str:
                return await tool_instance.arun(
                    topic, pro_position, con_position, rounds
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
                    "pro_position": {
                        "type": "string",
                        "description": "The position for the pro-side agent.",
                        "default": "supports the proposition",
                    },
                    "con_position": {
                        "type": "string",
                        "description": "The position for the con-side agent.",
                        "default": "opposes the proposition",
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
