"""
ARTEMIS MCP Tool Definitions

Defines the MCP tools exposed by the ARTEMIS server.
Follows the Model Context Protocol specification.
"""

from typing import Any

from pydantic import BaseModel, Field


class DebateStartInput(BaseModel):
    """Input schema for starting a new debate."""

    topic: str = Field(
        ...,
        description="The debate topic or question to analyze.",
    )
    pro_position: str = Field(
        default="supports the proposition",
        description="The position the pro-side will argue.",
    )
    con_position: str = Field(
        default="opposes the proposition",
        description="The position the con-side will argue.",
    )
    rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of debate rounds.",
    )
    model: str = Field(
        default="gpt-4o",
        description="LLM model to use for agents.",
    )


class DebateStartOutput(BaseModel):
    """Output schema for debate start."""

    debate_id: str = Field(description="Unique identifier for the debate session.")
    topic: str = Field(description="The debate topic.")
    status: str = Field(description="Current debate status.")
    agents: list[str] = Field(description="List of agent names.")


class AddRoundInput(BaseModel):
    """Input schema for adding a debate round."""

    debate_id: str = Field(
        ...,
        description="The debate session ID.",
    )


class AddRoundOutput(BaseModel):
    """Output schema for adding a round."""

    debate_id: str
    round_number: int
    pro_argument: str
    con_argument: str
    pro_score: float
    con_score: float
    status: str


class GetVerdictInput(BaseModel):
    """Input schema for getting the verdict."""

    debate_id: str = Field(
        ...,
        description="The debate session ID.",
    )


class GetVerdictOutput(BaseModel):
    """Output schema for verdict."""

    debate_id: str
    verdict: str
    confidence: float
    reasoning: str
    winner: str | None
    final_scores: dict[str, float]


class GetTranscriptInput(BaseModel):
    """Input schema for getting debate transcript."""

    debate_id: str = Field(
        ...,
        description="The debate session ID.",
    )
    include_evaluations: bool = Field(
        default=True,
        description="Whether to include evaluation scores.",
    )


class TranscriptTurn(BaseModel):
    """A single turn in the debate transcript."""

    round: int
    agent: str
    position: str
    argument: str
    score: float | None = None


class GetTranscriptOutput(BaseModel):
    """Output schema for transcript."""

    debate_id: str
    topic: str
    turns: list[TranscriptTurn]
    status: str


class ListDebatesOutput(BaseModel):
    """Output for listing active debates."""

    debates: list[dict[str, Any]]


class AnalyzeTopicInput(BaseModel):
    """Input for quick topic analysis."""

    topic: str = Field(
        ...,
        description="The topic to analyze.",
    )
    depth: str = Field(
        default="moderate",
        description="Analysis depth: 'quick', 'moderate', or 'thorough'.",
    )


class AnalyzeTopicOutput(BaseModel):
    """Output for topic analysis."""

    topic: str
    pro_points: list[str]
    con_points: list[str]
    key_considerations: list[str]
    recommended_rounds: int


# Tool definitions following MCP specification
ARTEMIS_TOOLS: list[dict[str, Any]] = [
    {
        "name": "artemis_debate_start",
        "description": (
            "Start a new structured multi-agent debate. Two AI agents will argue "
            "opposing positions on the given topic through multiple rounds. "
            "Returns a debate ID that can be used to add rounds and get verdicts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The debate topic or question to analyze.",
                },
                "pro_position": {
                    "type": "string",
                    "description": "The position the pro-side will argue.",
                    "default": "supports the proposition",
                },
                "con_position": {
                    "type": "string",
                    "description": "The position the con-side will argue.",
                    "default": "opposes the proposition",
                },
                "rounds": {
                    "type": "integer",
                    "description": "Number of debate rounds (1-10).",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                },
                "model": {
                    "type": "string",
                    "description": "LLM model to use for agents.",
                    "default": "gpt-4o",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "artemis_add_round",
        "description": (
            "Add a round to an existing debate. Each round involves both agents "
            "presenting arguments, with evaluation scores for each. Use this to "
            "incrementally build the debate before requesting a verdict."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "debate_id": {
                    "type": "string",
                    "description": "The debate session ID from artemis_debate_start.",
                },
            },
            "required": ["debate_id"],
        },
    },
    {
        "name": "artemis_get_verdict",
        "description": (
            "Get the jury's verdict for a debate. The verdict includes the winning "
            "position, confidence score, reasoning, and final scores for each agent. "
            "Call this after all desired rounds have been completed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "debate_id": {
                    "type": "string",
                    "description": "The debate session ID.",
                },
            },
            "required": ["debate_id"],
        },
    },
    {
        "name": "artemis_get_transcript",
        "description": (
            "Get the full transcript of a debate, including all arguments made "
            "by both agents across all rounds, optionally with evaluation scores."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "debate_id": {
                    "type": "string",
                    "description": "The debate session ID.",
                },
                "include_evaluations": {
                    "type": "boolean",
                    "description": "Whether to include evaluation scores.",
                    "default": True,
                },
            },
            "required": ["debate_id"],
        },
    },
    {
        "name": "artemis_list_debates",
        "description": ("List all active debate sessions with their current status."),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "artemis_analyze_topic",
        "description": (
            "Quick analysis of a topic without running a full debate. "
            "Identifies key pro and con points, considerations, and suggests "
            "an appropriate number of rounds for a full debate."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to analyze.",
                },
                "depth": {
                    "type": "string",
                    "description": "Analysis depth: 'quick', 'moderate', or 'thorough'.",
                    "enum": ["quick", "moderate", "thorough"],
                    "default": "moderate",
                },
            },
            "required": ["topic"],
        },
    },
]


def get_tool_by_name(name: str) -> dict[str, Any] | None:
    """Get a tool definition by name."""
    for tool in ARTEMIS_TOOLS:
        if tool["name"] == name:
            return tool
    return None


def list_tool_names() -> list[str]:
    """List all available tool names."""
    return [tool["name"] for tool in ARTEMIS_TOOLS]
