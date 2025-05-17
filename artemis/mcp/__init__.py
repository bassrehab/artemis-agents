"""
ARTEMIS MCP Module

Model Context Protocol server implementation.
Exposes ARTEMIS capabilities as MCP tools for any compatible client.
"""

from artemis.mcp.server import ArtemisMCPServer, create_mcp_server
from artemis.mcp.sessions import (
    SessionManager,
    SessionMetadata,
    SessionSnapshot,
    SessionStore,
    create_session_manager,
    generate_session_id,
)
from artemis.mcp.tools import (
    ARTEMIS_TOOLS,
    AddRoundOutput,
    AnalyzeTopicInput,
    AnalyzeTopicOutput,
    DebateStartInput,
    DebateStartOutput,
    GetTranscriptInput,
    GetTranscriptOutput,
    GetVerdictInput,
    GetVerdictOutput,
    ListDebatesOutput,
    get_tool_by_name,
    list_tool_names,
)

__all__ = [
    # Server
    "ArtemisMCPServer",
    "create_mcp_server",
    # Sessions
    "SessionManager",
    "SessionStore",
    "SessionMetadata",
    "SessionSnapshot",
    "create_session_manager",
    "generate_session_id",
    # Tools
    "ARTEMIS_TOOLS",
    "get_tool_by_name",
    "list_tool_names",
    # Input/Output schemas
    "DebateStartInput",
    "DebateStartOutput",
    "AddRoundOutput",
    "GetVerdictInput",
    "GetVerdictOutput",
    "GetTranscriptInput",
    "GetTranscriptOutput",
    "ListDebatesOutput",
    "AnalyzeTopicInput",
    "AnalyzeTopicOutput",
]
