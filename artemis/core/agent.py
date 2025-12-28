"""Debate agent with H-L-DAG argument generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from artemis.core.argument import ArgumentParser
from artemis.core.causal import CausalExtractor, CausalGraph
from artemis.core.evidence import EvidenceExtractor
from artemis.core.prompts.hdag import (
    build_closing_prompt,
    build_generation_prompt,
    build_opening_prompt,
    build_rebuttal_prompt,
)
from artemis.core.types import (
    Argument,
    ArgumentLevel,
    DebateContext,
    Message,
    ReasoningConfig,
)
from artemis.exceptions import ArgumentGenerationError
from artemis.utils.logging import get_logger

if TYPE_CHECKING:
    from artemis.models.base import BaseModel

logger = get_logger(__name__)


class DebateStrategy(str, Enum):
    """Strategies for argument generation based on debate context."""

    ESTABLISH = "establish"  # Establish core position (early rounds)
    REINFORCE = "reinforce"  # Reinforce with evidence (middle rounds)
    COUNTER = "counter"  # Counter opponent arguments
    SYNTHESIZE = "synthesize"  # Synthesize and conclude (late rounds)
    ADAPT = "adapt"  # Adapt to opponent's strategy


@dataclass
class OpponentModel:
    """Tracks opponent debate patterns."""

    name: str
    arguments: list[Argument] = field(default_factory=list)
    primary_themes: list[str] = field(default_factory=list)
    evidence_types_used: list[str] = field(default_factory=list)
    causal_claims: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)

    def update(self, argument: Argument):
        self.arguments.append(argument)
        # Extract themes from content (simplified)
        words = argument.content.lower().split()
        for word in words:
            if len(word) > 6 and word not in self.primary_themes:
                self.primary_themes.append(word)
                if len(self.primary_themes) > 10:
                    self.primary_themes.pop(0)

    @property
    def argument_count(self) -> int:
        """Number of arguments observed."""
        return len(self.arguments)


@dataclass
class StrategyContext:
    """Context for strategy selection decisions."""

    current_round: int
    total_rounds: int
    own_argument_count: int
    opponent_argument_count: int
    is_winning: bool | None = None  # Based on evaluations if available
    opponent_last_strong: bool = False  # Was opponent's last argument strong?


class Agent:
    """Debate agent with H-L-DAG argument generation."""

    def __init__(
        self,
        name: str,
        role: str,
        model: str | BaseModel,
        position: str | None = None,
        persona: str | None = None,
        reasoning: ReasoningConfig | None = None,
        api_key: str | None = None,
        **model_kwargs: Any,
    ) -> None:
        self.name = name
        self.role = role
        self.position = position
        self.persona = persona
        self.reasoning = reasoning or ReasoningConfig(enabled=False)

        # Initialize the model (lazy import to avoid circular dependency)
        from artemis.models.base import BaseModel as ModelBase
        from artemis.models.base import ModelRegistry

        if isinstance(model, ModelBase):
            self._model = model
        else:
            self._model = ModelRegistry.create(model, api_key=api_key, **model_kwargs)

        # Argument parser for extracting structure
        self._parser = ArgumentParser()

        # Evidence and causal extractors
        self._evidence_extractor = EvidenceExtractor()
        self._causal_extractor = CausalExtractor()

        # Track generated arguments
        self._argument_history: list[Argument] = []

        # Causal graph built from arguments
        self._causal_graph = CausalGraph()

        # Opponent models for strategic adaptation
        self._opponent_models: dict[str, OpponentModel] = {}

        # Current strategy
        self._current_strategy = DebateStrategy.ESTABLISH

        logger.debug(
            "Agent initialized",
            name=name,
            role=role,
            model=self._model.model,
            reasoning_enabled=self.reasoning.enabled,
        )

    @property
    def model(self):
        return self._model

    @property
    def argument_history(self):
        return self._argument_history.copy()

    @property
    def causal_graph(self):
        return self._causal_graph

    @property
    def current_strategy(self):
        return self._current_strategy

    def set_position(self, position: str) -> None:
        self.position = position

    def select_strategy(self, context: DebateContext) -> DebateStrategy:
        """Select strategy based on debate context."""
        # HACK: progress calc is a bit rough
        progress = context.current_round / max(context.total_rounds, 1)

        # Count opponent arguments
        opponent_args = 0
        if context.transcript:
            for turn in context.transcript:
                if turn.agent != self.name:
                    opponent_args += 1

        # Strategy selection logic
        if progress < 0.2:
            # Early: Establish position
            strategy = DebateStrategy.ESTABLISH
        elif progress < 0.5:
            # Middle early: Reinforce with evidence
            strategy = DebateStrategy.REINFORCE
        elif progress < 0.8:
            # Middle late: Counter and adapt
            if opponent_args > len(self._argument_history):
                strategy = DebateStrategy.COUNTER
            else:
                strategy = DebateStrategy.ADAPT
        else:
            # Late: Synthesize
            strategy = DebateStrategy.SYNTHESIZE

        self._current_strategy = strategy

        logger.debug(
            "Strategy selected",
            agent=self.name,
            strategy=strategy.value,
            progress=progress,
        )

        return strategy

    def observe_opponent(self, argument: Argument) -> None:
        """Observe and model an opponent's argument."""
        if argument.agent == self.name:
            return  # Don't model self

        # Get or create opponent model
        if argument.agent not in self._opponent_models:
            self._opponent_models[argument.agent] = OpponentModel(name=argument.agent)

        model = self._opponent_models[argument.agent]
        model.update(argument)

        # Extract and analyze evidence used
        extracted = self._evidence_extractor.extract(argument.content)
        for ev in extracted:
            if ev.evidence_type.value not in model.evidence_types_used:
                model.evidence_types_used.append(ev.evidence_type.value)

        # Extract causal claims
        causal_links = self._causal_extractor.extract(argument.content)
        for link, _ in causal_links:
            claim = f"{link.cause} -> {link.effect}"
            if claim not in model.causal_claims:
                model.causal_claims.append(claim)

        logger.debug(
            "Opponent observed",
            agent=self.name,
            opponent=argument.agent,
            total_arguments=model.argument_count,
        )

    def get_opponent_model(self, opponent_name: str):
        return self._opponent_models.get(opponent_name)

    def extract_evidence(self, text: str):
        return self._evidence_extractor.extract(text)

    def build_strategy_instructions(self, strategy):
        instructions = {
            DebateStrategy.ESTABLISH: (
                "Focus on clearly establishing your core position. "
                "State your main thesis and preview key supporting points. "
                "Be confident and authoritative."
            ),
            DebateStrategy.REINFORCE: (
                "Reinforce your position with strong evidence. "
                "Cite studies, statistics, and expert opinions. "
                "Build multiple lines of supporting evidence."
            ),
            DebateStrategy.COUNTER: (
                "Address and refute the opponent's strongest arguments. "
                "Identify logical weaknesses in their claims. "
                "Provide counter-evidence where possible."
            ),
            DebateStrategy.SYNTHESIZE: (
                "Synthesize the debate into a compelling conclusion. "
                "Summarize your strongest points and why they prevail. "
                "Address key objections and show why your position is stronger."
            ),
            DebateStrategy.ADAPT: (
                "Adapt your approach based on the opponent's strategy. "
                "Fill gaps in the argument they haven't addressed. "
                "Introduce new angles they haven't considered."
            ),
        }
        return instructions.get(strategy, "")

    async def generate_argument(
        self,
        context: DebateContext,
        level: ArgumentLevel = ArgumentLevel.TACTICAL,
        additional_instructions: str | None = None,
        use_strategy: bool = True,
    ) -> Argument:
        """Generate an argument at the specified H-L-DAG level."""
        # Observe opponent arguments from transcript
        if context.transcript:
            for turn in context.transcript:
                if turn.agent != self.name:
                    self.observe_opponent(turn.argument)

        # Select strategy based on context
        strategy = self.select_strategy(context) if use_strategy else None

        logger.info(
            "Generating argument",
            agent=self.name,
            level=level.value,
            round=context.current_round,
            strategy=strategy.value if strategy else None,
        )

        try:
            # Build strategy instructions
            strategy_instructions = ""
            if strategy:
                strategy_instructions = self.build_strategy_instructions(strategy)

            # Combine with any additional instructions
            combined_instructions = strategy_instructions
            if additional_instructions:
                combined_instructions = f"{strategy_instructions}\n\n{additional_instructions}"

            # Build prompts
            system_prompt, user_prompt = build_generation_prompt(
                context=context,
                agent_name=self.name,
                role=self.role,
                level=level,
                persona=self.persona,
                additional_instructions=combined_instructions if combined_instructions else None,
            )

            # Generate response
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]

            thinking_trace: str | None = None
            content: str

            if self.reasoning.enabled and self._model.supports_reasoning:
                reasoning_response = await self._model.generate_with_reasoning(
                    messages=messages,
                    thinking_budget=self.reasoning.thinking_budget,
                )
                content = reasoning_response.content
                thinking_trace = reasoning_response.thinking
            else:
                response = await self._model.generate(messages=messages)
                content = response.content

            # Parse the response into structured argument
            argument = self._parser.parse(
                content=content,
                agent=self.name,
                level=level,
            )

            # Add thinking trace if available
            if thinking_trace and self.reasoning.include_trace_in_output:
                argument = Argument(
                    id=argument.id,
                    agent=argument.agent,
                    level=argument.level,
                    content=argument.content,
                    evidence=argument.evidence,
                    causal_links=argument.causal_links,
                    rebuts=argument.rebuts,
                    supports=argument.supports,
                    ethical_score=argument.ethical_score,
                    thinking_trace=thinking_trace,
                    timestamp=argument.timestamp,
                )

            # Track in history
            self._argument_history.append(argument)

            # Update causal graph with extracted links
            for link in argument.causal_links:
                self._causal_graph.add_link(link, argument_id=argument.id)

            logger.info(
                "Argument generated",
                agent=self.name,
                argument_id=argument.id,
                level=level.value,
                strategy=strategy.value if strategy else None,
                evidence_count=len(argument.evidence),
                causal_links_count=len(argument.causal_links),
            )

            return argument

        except Exception as e:
            logger.error(
                "Argument generation failed",
                agent=self.name,
                level=level.value,
                error=str(e),
            )
            raise ArgumentGenerationError(
                message=f"Failed to generate {level.value} argument: {e}",
                agent_name=self.name,
            ) from e

    async def generate_opening(self, context: DebateContext) -> Argument:
        """Generate an opening statement."""
        logger.info("Generating opening statement", agent=self.name)

        try:
            system_prompt, user_prompt = build_opening_prompt(
                context=context,
                agent_name=self.name,
                role=self.role,
                persona=self.persona,
            )

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]

            response = await self._model.generate(messages=messages)

            argument = self._parser.parse(
                content=response.content,
                agent=self.name,
                level=ArgumentLevel.STRATEGIC,
            )

            self._argument_history.append(argument)
            return argument

        except Exception as e:
            raise ArgumentGenerationError(
                message=f"Failed to generate opening statement: {e}",
                agent_name=self.name,
            ) from e

    async def generate_closing(self, context: DebateContext) -> Argument:
        """Generate a closing statement."""
        logger.info("Generating closing statement", agent=self.name)

        try:
            system_prompt, user_prompt = build_closing_prompt(
                context=context,
                agent_name=self.name,
                role=self.role,
                persona=self.persona,
            )

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]

            response = await self._model.generate(messages=messages)

            argument = self._parser.parse(
                content=response.content,
                agent=self.name,
                level=ArgumentLevel.STRATEGIC,
            )

            self._argument_history.append(argument)
            return argument

        except Exception as e:
            raise ArgumentGenerationError(
                message=f"Failed to generate closing statement: {e}",
                agent_name=self.name,
            ) from e

    async def generate_rebuttal(
        self,
        context: DebateContext,
        target_argument: Argument,
        level: ArgumentLevel = ArgumentLevel.TACTICAL,
    ) -> Argument:
        """Generate a rebuttal to a specific argument."""
        logger.info(
            "Generating rebuttal",
            agent=self.name,
            target_id=target_argument.id,
            target_agent=target_argument.agent,
        )

        try:
            system_prompt, user_prompt = build_rebuttal_prompt(
                context=context,
                agent_name=self.name,
                role=self.role,
                target_argument=target_argument.content,
                level=level,
                persona=self.persona,
            )

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]

            response = await self._model.generate(messages=messages)

            # Parse and link to target
            argument = self._parser.parse(
                content=response.content,
                agent=self.name,
                level=level,
            )

            # Create new argument with rebuttal link
            argument = Argument(
                id=argument.id,
                agent=argument.agent,
                level=argument.level,
                content=argument.content,
                evidence=argument.evidence,
                causal_links=argument.causal_links,
                rebuts=target_argument.id,
                supports=argument.supports,
                ethical_score=argument.ethical_score,
                thinking_trace=argument.thinking_trace,
                timestamp=argument.timestamp,
            )

            self._argument_history.append(argument)
            return argument

        except Exception as e:
            raise ArgumentGenerationError(
                message=f"Failed to generate rebuttal: {e}",
                agent_name=self.name,
            ) from e

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, role={self.role!r}, model={self._model.model!r})"
