"""
ARTEMIS MCP Server

Model Context Protocol server implementation for ARTEMIS debates.
Exposes structured multi-agent debate capabilities via MCP.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from artemis.core.agent import Agent
from artemis.core.debate import Debate
from artemis.core.types import DebateConfig, DebateResult
from artemis.mcp.tools import (
    ARTEMIS_TOOLS,
    AddRoundOutput,
    AnalyzeTopicOutput,
    DebateStartOutput,
    GetTranscriptOutput,
    GetVerdictOutput,
    ListDebatesOutput,
    TranscriptTurn,
)
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DebateSession:
    """Represents an active debate session."""

    debate_id: str
    topic: str
    debate: Debate
    created_at: datetime
    status: str = "active"
    current_round: int = 0
    result: DebateResult | None = None


@dataclass
class ServerState:
    """Server state management."""

    sessions: dict[str, DebateSession] = field(default_factory=dict)
    max_sessions: int = 100

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than max_age_hours."""
        now = datetime.now()
        to_remove = []
        for session_id, session in self.sessions.items():
            age = (now - session.created_at).total_seconds() / 3600
            if age > max_age_hours:
                to_remove.append(session_id)

        for session_id in to_remove:
            del self.sessions[session_id]

        return len(to_remove)


class ArtemisMCPServer:
    """
    MCP Server for ARTEMIS debates.

    Implements the Model Context Protocol to expose structured
    multi-agent debate capabilities.

    Example:
        >>> server = ArtemisMCPServer()
        >>> await server.start(port=8080)

    Or as a stdio server:
        >>> server = ArtemisMCPServer()
        >>> await server.run_stdio()
    """

    def __init__(
        self,
        default_model: str = "gpt-4o",
        max_sessions: int = 100,
        config: DebateConfig | None = None,
    ) -> None:
        """
        Initialize the MCP server.

        Args:
            default_model: Default LLM model for debates.
            max_sessions: Maximum concurrent debate sessions.
            config: Default debate configuration.
        """
        self.default_model = default_model
        self.config = config or DebateConfig()
        self.state = ServerState(max_sessions=max_sessions)

        logger.info(
            "ArtemisMCPServer initialized",
            default_model=default_model,
            max_sessions=max_sessions,
        )

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Get available MCP tools."""
        return ARTEMIS_TOOLS

    async def handle_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Handle an MCP tool call.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            Tool result as dict.

        Raises:
            ValueError: If tool is unknown.
        """
        handlers = {
            "artemis_debate_start": self._handle_debate_start,
            "artemis_add_round": self._handle_add_round,
            "artemis_get_verdict": self._handle_get_verdict,
            "artemis_get_transcript": self._handle_get_transcript,
            "artemis_list_debates": self._handle_list_debates,
            "artemis_analyze_topic": self._handle_analyze_topic,
        }

        handler = handlers.get(name)
        if not handler:
            raise ValueError(f"Unknown tool: {name}")

        logger.debug("Handling tool call", tool=name)
        result = await handler(arguments)
        return result.model_dump() if hasattr(result, "model_dump") else result

    async def _handle_debate_start(
        self,
        arguments: dict[str, Any],
    ) -> DebateStartOutput:
        """Start a new debate session."""
        topic = arguments["topic"]
        pro_position = arguments.get("pro_position", "supports the proposition")
        con_position = arguments.get("con_position", "opposes the proposition")
        rounds = arguments.get("rounds", 3)
        model = arguments.get("model", self.default_model)

        # Check session limit
        if len(self.state.sessions) >= self.state.max_sessions:
            self.state.cleanup_old_sessions()
            if len(self.state.sessions) >= self.state.max_sessions:
                raise ValueError("Maximum session limit reached")

        # Create debate
        debate_id = str(uuid.uuid4())[:8]

        pro_agent = Agent(
            name="pro_agent",
            model=model,
            position=pro_position,
        )
        con_agent = Agent(
            name="con_agent",
            model=model,
            position=con_position,
        )

        debate = Debate(
            topic=topic,
            agents=[pro_agent, con_agent],
            rounds=rounds,
            config=self.config,
        )

        debate.assign_positions({
            "pro_agent": pro_position,
            "con_agent": con_position,
        })

        session = DebateSession(
            debate_id=debate_id,
            topic=topic,
            debate=debate,
            created_at=datetime.now(),
        )
        self.state.sessions[debate_id] = session

        logger.info(
            "Debate started",
            debate_id=debate_id,
            topic=topic[:50],
        )

        return DebateStartOutput(
            debate_id=debate_id,
            topic=topic,
            status="active",
            agents=["pro_agent", "con_agent"],
        )

    async def _handle_add_round(
        self,
        arguments: dict[str, Any],
    ) -> AddRoundOutput:
        """Add a round to an existing debate."""
        debate_id = arguments["debate_id"]
        session = self._get_session(debate_id)

        if session.status != "active":
            raise ValueError(f"Debate {debate_id} is not active")

        # Run a single round
        session.current_round += 1
        result = await session.debate.run_round()

        # Get the latest arguments
        pro_arg = ""
        con_arg = ""
        pro_score = 0.0
        con_score = 0.0

        for turn in result.transcript[-2:]:  # Last two turns
            if turn.agent == "pro_agent":
                pro_arg = turn.argument.content[:500]
                pro_score = turn.evaluation.total_score if turn.evaluation else 0.0
            else:
                con_arg = turn.argument.content[:500]
                con_score = turn.evaluation.total_score if turn.evaluation else 0.0

        return AddRoundOutput(
            debate_id=debate_id,
            round_number=session.current_round,
            pro_argument=pro_arg,
            con_argument=con_arg,
            pro_score=pro_score,
            con_score=con_score,
            status="active",
        )

    async def _handle_get_verdict(
        self,
        arguments: dict[str, Any],
    ) -> GetVerdictOutput:
        """Get the verdict for a debate."""
        debate_id = arguments["debate_id"]
        session = self._get_session(debate_id)

        # Run remaining rounds if needed
        if session.status == "active":
            result = await session.debate.run()
            session.result = result
            session.status = "completed"
        elif session.result is None:
            raise ValueError(f"Debate {debate_id} has no result")

        result = session.result
        verdict = result.verdict

        # Calculate final scores
        scores: dict[str, list[float]] = {}
        for turn in result.transcript:
            if turn.evaluation:
                if turn.agent not in scores:
                    scores[turn.agent] = []
                scores[turn.agent].append(turn.evaluation.total_score)

        final_scores = {
            agent: sum(s) / len(s) if s else 0.0
            for agent, s in scores.items()
        }

        return GetVerdictOutput(
            debate_id=debate_id,
            verdict=verdict.decision,
            confidence=verdict.confidence,
            reasoning=verdict.reasoning,
            winner=verdict.decision if verdict.decision in ("pro", "con") else None,
            final_scores=final_scores,
        )

    async def _handle_get_transcript(
        self,
        arguments: dict[str, Any],
    ) -> GetTranscriptOutput:
        """Get the transcript of a debate."""
        debate_id = arguments["debate_id"]
        include_evaluations = arguments.get("include_evaluations", True)

        session = self._get_session(debate_id)

        # Get transcript from debate or result
        if session.result:
            transcript = session.result.transcript
        else:
            transcript = session.debate.get_transcript()

        turns = []
        for turn in transcript:
            score = None
            if include_evaluations and turn.evaluation:
                score = turn.evaluation.total_score

            turns.append(TranscriptTurn(
                round=turn.round,
                agent=turn.agent,
                position=session.debate.get_position(turn.agent),
                argument=turn.argument.content,
                score=score,
            ))

        return GetTranscriptOutput(
            debate_id=debate_id,
            topic=session.topic,
            turns=turns,
            status=session.status,
        )

    async def _handle_list_debates(
        self,
        _arguments: dict[str, Any],
    ) -> ListDebatesOutput:
        """List all active debates."""
        debates = []
        for session_id, session in self.state.sessions.items():
            debates.append({
                "debate_id": session_id,
                "topic": session.topic[:100],
                "status": session.status,
                "current_round": session.current_round,
                "created_at": session.created_at.isoformat(),
            })

        return ListDebatesOutput(debates=debates)

    async def _handle_analyze_topic(
        self,
        arguments: dict[str, Any],
    ) -> AnalyzeTopicOutput:
        """Quick topic analysis without full debate."""
        topic = arguments["topic"]
        depth = arguments.get("depth", "moderate")

        # Use a single agent to analyze the topic
        from artemis.models import create_model

        model = create_model(self.default_model)

        prompt = self._build_analysis_prompt(topic, depth)
        messages = [{"role": "user", "content": prompt}]

        response = await model.generate(messages)
        analysis = self._parse_analysis(response.content)

        return AnalyzeTopicOutput(
            topic=topic,
            pro_points=analysis.get("pro_points", []),
            con_points=analysis.get("con_points", []),
            key_considerations=analysis.get("considerations", []),
            recommended_rounds=analysis.get("recommended_rounds", 3),
        )

    def _get_session(self, debate_id: str) -> DebateSession:
        """Get a debate session by ID."""
        session = self.state.sessions.get(debate_id)
        if not session:
            raise ValueError(f"Debate {debate_id} not found")
        return session

    def _build_analysis_prompt(self, topic: str, depth: str) -> str:
        """Build the topic analysis prompt."""
        depth_instructions = {
            "quick": "Briefly identify 2-3 key points for each side.",
            "moderate": "Identify 3-5 points for each side with brief explanations.",
            "thorough": "Provide detailed analysis with 5+ points per side.",
        }

        return f"""Analyze this topic for debate: {topic}

{depth_instructions.get(depth, depth_instructions["moderate"])}

Respond in JSON format:
{{
    "pro_points": ["point1", "point2", ...],
    "con_points": ["point1", "point2", ...],
    "considerations": ["consideration1", ...],
    "recommended_rounds": <number 1-10>
}}"""

    def _parse_analysis(self, content: str) -> dict[str, Any]:
        """Parse the analysis response."""
        try:
            # Try to extract JSON from response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass

        # Fallback to defaults
        return {
            "pro_points": ["Pro arguments identified"],
            "con_points": ["Con arguments identified"],
            "considerations": ["Analysis inconclusive"],
            "recommended_rounds": 3,
        }

    async def run_stdio(self) -> None:
        """
        Run the server in stdio mode.

        Reads JSON-RPC requests from stdin and writes responses to stdout.
        """
        import sys

        logger.info("Starting MCP server in stdio mode")

        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                request = json.loads(line)
                response = await self._handle_jsonrpc(request)

                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                    "id": None,
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()

            except Exception as e:
                logger.error("Server error", error=str(e))
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": str(e)},
                    "id": None,
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()

    async def _handle_jsonrpc(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a JSON-RPC request."""
        request_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        try:
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "tools/list":
                result = {"tools": self.tools}
            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {})
                content = await self.handle_tool_call(tool_name, tool_args)
                result = {
                    "content": [
                        {"type": "text", "text": json.dumps(content, indent=2)}
                    ]
                }
            else:
                raise ValueError(f"Unknown method: {method}")

            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id,
            }

        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(e)},
                "id": request_id,
            }

    async def _handle_initialize(
        self,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle the initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": "artemis-mcp-server",
                "version": "0.1.0",
            },
        }

    async def start(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        """
        Start the HTTP server.

        Args:
            host: Host to bind to.
            port: Port to listen on.
        """
        try:
            from aiohttp import web
        except ImportError as e:
            raise ImportError(
                "aiohttp is required for HTTP server mode. "
                "Install with: pip install aiohttp"
            ) from e

        app = web.Application()
        app.router.add_post("/mcp", self._handle_http_request)
        app.router.add_get("/health", self._handle_health)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, host, port)
        logger.info(f"Starting HTTP server on {host}:{port}")
        await site.start()

        # Keep running
        await asyncio.Event().wait()

    async def _handle_http_request(self, request: Any) -> Any:
        """Handle an HTTP request."""
        from aiohttp import web

        try:
            body = await request.json()
            response = await self._handle_jsonrpc(body)
            return web.json_response(response)
        except Exception as e:
            return web.json_response(
                {"error": str(e)},
                status=500,
            )

    async def _handle_health(self, _request: Any) -> Any:
        """Health check endpoint."""
        from aiohttp import web

        return web.json_response({
            "status": "healthy",
            "sessions": len(self.state.sessions),
        })


async def create_mcp_server(
    model: str = "gpt-4o",
    **kwargs: Any,
) -> ArtemisMCPServer:
    """
    Factory function to create an MCP server.

    Args:
        model: Default LLM model.
        **kwargs: Additional server configuration.

    Returns:
        Configured MCP server instance.
    """
    return ArtemisMCPServer(default_model=model, **kwargs)
