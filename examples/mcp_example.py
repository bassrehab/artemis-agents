"""
ARTEMIS MCP Server Example

Demonstrates programmatic usage of the ARTEMIS MCP server.

This example shows how to:
1. Create and configure an MCP server
2. Start debates via tool calls
3. Manage sessions
4. Run topic analysis

Usage:
    python examples/mcp_example.py
"""

import asyncio
from datetime import datetime

from artemis.mcp import (
    ArtemisMCPServer,
    SessionManager,
    SessionStore,
    create_mcp_server,
)


async def basic_tool_usage() -> None:
    """Demonstrate basic MCP tool calls."""
    print("\n=== Basic MCP Tool Usage ===\n")

    # Create server instance
    server = ArtemisMCPServer(
        default_model="gpt-4o",
        max_sessions=50,
    )

    # List available tools
    print("Available tools:")
    for tool in server.tools:
        print(f"  - {tool['name']}: {tool['description'][:60]}...")

    # Start a mock debate (would need real API keys)
    print("\nTool schemas defined for:")
    print("  - artemis_debate_start")
    print("  - artemis_add_round")
    print("  - artemis_get_verdict")
    print("  - artemis_get_transcript")
    print("  - artemis_list_debates")
    print("  - artemis_analyze_topic")


async def session_management_example() -> None:
    """Demonstrate session management features."""
    print("\n=== Session Management ===\n")

    # Create memory-only store (no persistence)
    store = SessionStore(
        storage_path=None,  # Memory only
        cache_enabled=True,
        max_cache_size=100,
    )

    # Create session manager
    manager = SessionManager(
        store=store,
        auto_cleanup=False,  # Manual cleanup for demo
    )

    # Create sessions
    session1_id = await manager.create_session(
        topic="Should AI be open source?",
        model="gpt-4o",
        rounds=3,
        tags=["technology", "policy"],
    )
    print(f"Created session 1: {session1_id}")

    session2_id = await manager.create_session(
        topic="Is remote work more productive?",
        model="gpt-4o",
        rounds=2,
        tags=["workplace", "productivity"],
    )
    print(f"Created session 2: {session2_id}")

    # List sessions
    sessions = await manager.list_sessions()
    print(f"\nActive sessions: {len(sessions)}")
    for session in sessions:
        print(f"  - {session.session_id}: {session.topic[:40]}...")

    # Simulate round data
    round_data = [
        {"agent": "pro_agent", "argument": "Pro argument for round 1"},
        {"agent": "con_agent", "argument": "Con argument for round 1"},
    ]
    await manager.update_round(session1_id, round_data)
    print(f"\nUpdated session {session1_id} with round 1 data")

    # Get session details
    snapshot = await manager.get_session(session1_id)
    if snapshot:
        print(f"Session status: {snapshot.metadata.status}")
        print(f"Rounds completed: {snapshot.metadata.rounds_completed}")
        print(f"Transcript entries: {len(snapshot.transcript)}")

    # Complete session with verdict
    verdict = {
        "decision": "pro",
        "confidence": 0.72,
        "reasoning": "Pro arguments demonstrated stronger evidence.",
    }
    await manager.complete_session(session1_id, verdict)
    print(f"\nCompleted session {session1_id} with verdict: {verdict['decision']}")

    # Archive session
    await manager.archive_session(session1_id)
    print(f"Archived session {session1_id}")

    # List by status
    active = await manager.list_sessions(status="active")
    archived = await manager.list_sessions(status="archived")
    print("\nSession counts:")
    print(f"  Active: {len(active)}")
    print(f"  Archived: {len(archived)}")

    # Cleanup
    await manager.delete_session(session2_id)
    print(f"\nDeleted session {session2_id}")


async def server_creation_example() -> None:
    """Demonstrate server factory usage."""
    print("\n=== Server Factory ===\n")

    # Use factory function
    server = await create_mcp_server(
        model="gpt-4-turbo",
        max_sessions=25,
    )

    print(f"Created server with model: {server.default_model}")
    print(f"Max sessions: {server.state.max_sessions}")
    print(f"Tools available: {len(server.tools)}")


async def jsonrpc_example() -> None:
    """Demonstrate JSON-RPC protocol handling."""
    print("\n=== JSON-RPC Protocol ===\n")

    server = ArtemisMCPServer()

    # Initialize request
    init_request = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {},
        "id": 1,
    }
    init_response = await server._handle_jsonrpc(init_request)
    print("Initialize response:")
    print(f"  Server: {init_response['result']['serverInfo']['name']}")
    print(f"  Version: {init_response['result']['serverInfo']['version']}")
    print(f"  Protocol: {init_response['result']['protocolVersion']}")

    # List tools request
    tools_request = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": 2,
    }
    tools_response = await server._handle_jsonrpc(tools_request)
    print(f"\nTools available: {len(tools_response['result']['tools'])}")

    # Call tool request (list debates)
    call_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "artemis_list_debates",
            "arguments": {},
        },
        "id": 3,
    }
    call_response = await server._handle_jsonrpc(call_request)
    print("\nList debates response received")
    print(f"  Content type: {call_response['result']['content'][0]['type']}")


async def main() -> None:
    """Run all examples."""
    print("=" * 60)
    print("ARTEMIS MCP Server Examples")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    await basic_tool_usage()
    await session_management_example()
    await server_creation_example()
    await jsonrpc_example()

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
