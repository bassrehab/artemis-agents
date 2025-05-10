"""
ARTEMIS MCP Module

Model Context Protocol server implementation.
Exposes ARTEMIS capabilities as MCP tools for any compatible client.
"""

from artemis.mcp.server import ArtemisMCPServer, create_mcp_server
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
