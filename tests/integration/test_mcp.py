"""
Integration tests for the ARTEMIS MCP server.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from artemis.mcp.server import ArtemisMCPServer, ServerState
from artemis.mcp.sessions import (
    SessionManager,
    SessionMetadata,
    SessionSnapshot,
    SessionStore,
    generate_session_id,
)
from artemis.mcp.tools import (
    ARTEMIS_TOOLS,
    DebateStartInput,
    DebateStartOutput,
    get_tool_by_name,
    list_tool_names,
)

# ============================================================================
# Tool Definition Tests
# ============================================================================


class TestToolDefinitions:
    """Tests for MCP tool definitions."""

    def test_all_tools_defined(self) -> None:
        """All expected tools should be defined."""
        expected_tools = [
            "artemis_debate_start",
            "artemis_add_round",
            "artemis_get_verdict",
            "artemis_get_transcript",
            "artemis_list_debates",
            "artemis_analyze_topic",
        ]
        tool_names = list_tool_names()
        for expected in expected_tools:
            assert expected in tool_names

    def test_tool_has_required_fields(self) -> None:
        """Each tool should have required fields."""
        for tool in ARTEMIS_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert isinstance(tool["description"], str)
            assert len(tool["description"]) > 10

    def test_get_tool_by_name(self) -> None:
        """Should get tool by name."""
        tool = get_tool_by_name("artemis_debate_start")
        assert tool is not None
        assert tool["name"] == "artemis_debate_start"

    def test_get_nonexistent_tool(self) -> None:
        """Should return None for unknown tool."""
        tool = get_tool_by_name("nonexistent_tool")
        assert tool is None

    def test_input_schema_structure(self) -> None:
        """Input schemas should have proper structure."""
        tool = get_tool_by_name("artemis_debate_start")
        assert tool is not None
        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "topic" in schema["properties"]
        assert "required" in schema
        assert "topic" in schema["required"]


class TestInputOutputSchemas:
    """Tests for input/output Pydantic models."""

    def test_debate_start_input_defaults(self) -> None:
        """DebateStartInput should have proper defaults."""
        input_data = DebateStartInput(topic="Test topic")
        assert input_data.topic == "Test topic"
        assert input_data.pro_position == "supports the proposition"
        assert input_data.con_position == "opposes the proposition"
        assert input_data.rounds == 3
        assert input_data.model == "gpt-4o"

    def test_debate_start_input_validation(self) -> None:
        """DebateStartInput should validate rounds."""
        # Valid range
        input_data = DebateStartInput(topic="Test", rounds=5)
        assert input_data.rounds == 5

        # Invalid range
        with pytest.raises(ValueError):
            DebateStartInput(topic="Test", rounds=15)

    def test_debate_start_output(self) -> None:
        """DebateStartOutput should serialize correctly."""
        output = DebateStartOutput(
            debate_id="abc123",
            topic="Test topic",
            status="active",
            agents=["pro_agent", "con_agent"],
        )
        data = output.model_dump()
        assert data["debate_id"] == "abc123"
        assert data["status"] == "active"
        assert len(data["agents"]) == 2


# ============================================================================
# Session Store Tests
# ============================================================================


class TestSessionStore:
    """Tests for the session store."""

    @pytest.fixture
    def temp_storage(self) -> Path:
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_metadata(self) -> SessionMetadata:
        """Create sample session metadata."""
        now = datetime.now()
        return SessionMetadata(
            session_id="test123",
            topic="Test debate topic",
            model="gpt-4o",
            rounds_total=3,
            rounds_completed=0,
            status="active",
            created_at=now,
            updated_at=now,
        )

    @pytest.fixture
    def sample_snapshot(self, sample_metadata: SessionMetadata) -> SessionSnapshot:
        """Create sample session snapshot."""
        return SessionSnapshot(
            metadata=sample_metadata,
            transcript=[],
            verdict=None,
        )

    @pytest.mark.asyncio
    async def test_save_and_load_memory(
        self,
        sample_snapshot: SessionSnapshot,
    ) -> None:
        """Should save and load from memory cache."""
        store = SessionStore(storage_path=None, cache_enabled=True)
        await store.save(sample_snapshot)
        loaded = await store.load("test123")
        assert loaded is not None
        assert loaded.metadata.session_id == "test123"
        assert loaded.metadata.topic == "Test debate topic"

    @pytest.mark.asyncio
    async def test_save_and_load_file(
        self,
        temp_storage: Path,
        sample_snapshot: SessionSnapshot,
    ) -> None:
        """Should save and load from file storage."""
        store = SessionStore(storage_path=temp_storage, cache_enabled=False)
        await store.save(sample_snapshot)

        # Verify file exists
        file_path = temp_storage / "test123.json"
        assert file_path.exists()

        # Load and verify
        loaded = await store.load("test123")
        assert loaded is not None
        assert loaded.metadata.session_id == "test123"

    @pytest.mark.asyncio
    async def test_delete_session(
        self,
        sample_snapshot: SessionSnapshot,
    ) -> None:
        """Should delete session from store."""
        store = SessionStore(storage_path=None, cache_enabled=True)
        await store.save(sample_snapshot)

        deleted = await store.delete("test123")
        assert deleted is True

        loaded = await store.load("test123")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self) -> None:
        """Should return False when deleting nonexistent session."""
        store = SessionStore(storage_path=None)
        deleted = await store.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_sessions(self) -> None:
        """Should list all sessions."""
        store = SessionStore(storage_path=None, cache_enabled=True)

        # Create multiple sessions
        for i in range(5):
            metadata = SessionMetadata(
                session_id=f"session{i}",
                topic=f"Topic {i}",
                model="gpt-4o",
                rounds_total=3,
                rounds_completed=0,
                status="active" if i % 2 == 0 else "completed",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            snapshot = SessionSnapshot(metadata=metadata, transcript=[])
            await store.save(snapshot)

        # List all
        sessions = await store.list_sessions()
        assert len(sessions) == 5

        # Filter by status
        active = await store.list_sessions(status="active")
        assert len(active) == 3

        completed = await store.list_sessions(status="completed")
        assert len(completed) == 2

    @pytest.mark.asyncio
    async def test_cache_eviction(self) -> None:
        """Should evict oldest entries when cache is full."""
        store = SessionStore(storage_path=None, cache_enabled=True, max_cache_size=3)

        # Add 5 sessions
        for i in range(5):
            metadata = SessionMetadata(
                session_id=f"session{i}",
                topic=f"Topic {i}",
                model="gpt-4o",
                rounds_total=3,
                rounds_completed=0,
                status="active",
                created_at=datetime.now() + timedelta(seconds=i),
                updated_at=datetime.now() + timedelta(seconds=i),
            )
            snapshot = SessionSnapshot(metadata=metadata, transcript=[])
            await store.save(snapshot)

        # Should only have 3 in cache
        assert len(store._cache) == 3

    @pytest.mark.asyncio
    async def test_cleanup_old_sessions(
        self,
        temp_storage: Path,
    ) -> None:
        """Should clean up old sessions."""
        store = SessionStore(storage_path=temp_storage, cache_enabled=True)

        # Create old and new sessions
        old_metadata = SessionMetadata(
            session_id="old_session",
            topic="Old topic",
            model="gpt-4o",
            rounds_total=3,
            rounds_completed=0,
            status="completed",
            created_at=datetime.now() - timedelta(hours=48),
            updated_at=datetime.now() - timedelta(hours=48),
        )
        new_metadata = SessionMetadata(
            session_id="new_session",
            topic="New topic",
            model="gpt-4o",
            rounds_total=3,
            rounds_completed=0,
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        await store.save(SessionSnapshot(metadata=old_metadata, transcript=[]))
        await store.save(SessionSnapshot(metadata=new_metadata, transcript=[]))

        # Cleanup sessions older than 24 hours
        cleaned = await store.cleanup(max_age_hours=24)
        assert cleaned == 1

        # Old session should be gone
        old_loaded = await store.load("old_session")
        assert old_loaded is None

        # New session should remain
        new_loaded = await store.load("new_session")
        assert new_loaded is not None


class TestSessionMetadata:
    """Tests for SessionMetadata."""

    def test_to_dict(self) -> None:
        """Should serialize to dictionary."""
        now = datetime.now()
        metadata = SessionMetadata(
            session_id="test123",
            topic="Test",
            model="gpt-4o",
            rounds_total=3,
            rounds_completed=1,
            status="active",
            created_at=now,
            updated_at=now,
            tags=["tag1", "tag2"],
        )
        data = metadata.to_dict()
        assert data["session_id"] == "test123"
        assert data["tags"] == ["tag1", "tag2"]
        assert "created_at" in data

    def test_from_dict(self) -> None:
        """Should deserialize from dictionary."""
        data = {
            "session_id": "test123",
            "topic": "Test",
            "model": "gpt-4o",
            "rounds_total": 3,
            "rounds_completed": 1,
            "status": "active",
            "created_at": "2025-05-10T10:00:00",
            "updated_at": "2025-05-10T10:30:00",
        }
        metadata = SessionMetadata.from_dict(data)
        assert metadata.session_id == "test123"
        assert isinstance(metadata.created_at, datetime)


# ============================================================================
# Session Manager Tests
# ============================================================================


class TestSessionManager:
    """Tests for the session manager."""

    @pytest.fixture
    def manager(self) -> SessionManager:
        """Create session manager with memory-only store."""
        store = SessionStore(storage_path=None)
        return SessionManager(store=store, auto_cleanup=False)

    @pytest.mark.asyncio
    async def test_create_session(self, manager: SessionManager) -> None:
        """Should create a new session."""
        session_id = await manager.create_session(
            topic="Test debate",
            model="gpt-4o",
            rounds=3,
        )
        assert session_id is not None
        assert len(session_id) == 8

    @pytest.mark.asyncio
    async def test_get_session(self, manager: SessionManager) -> None:
        """Should get session by ID."""
        session_id = await manager.create_session(
            topic="Test debate",
            model="gpt-4o",
        )
        snapshot = await manager.get_session(session_id)
        assert snapshot is not None
        assert snapshot.metadata.topic == "Test debate"

    @pytest.mark.asyncio
    async def test_update_round(self, manager: SessionManager) -> None:
        """Should update session with round data."""
        session_id = await manager.create_session(
            topic="Test debate",
            model="gpt-4o",
        )

        round_data = [
            {"agent": "pro_agent", "argument": "Pro argument 1"},
            {"agent": "con_agent", "argument": "Con argument 1"},
        ]
        await manager.update_round(session_id, round_data)

        snapshot = await manager.get_session(session_id)
        assert snapshot is not None
        assert snapshot.metadata.rounds_completed == 1
        assert len(snapshot.transcript) == 2

    @pytest.mark.asyncio
    async def test_complete_session(self, manager: SessionManager) -> None:
        """Should complete session with verdict."""
        session_id = await manager.create_session(
            topic="Test debate",
            model="gpt-4o",
        )

        verdict = {
            "decision": "pro",
            "confidence": 0.75,
            "reasoning": "Pro arguments were stronger.",
        }
        await manager.complete_session(session_id, verdict)

        snapshot = await manager.get_session(session_id)
        assert snapshot is not None
        assert snapshot.metadata.status == "completed"
        assert snapshot.verdict == verdict

    @pytest.mark.asyncio
    async def test_archive_session(self, manager: SessionManager) -> None:
        """Should archive a session."""
        session_id = await manager.create_session(
            topic="Test debate",
            model="gpt-4o",
        )
        await manager.archive_session(session_id)

        snapshot = await manager.get_session(session_id)
        assert snapshot is not None
        assert snapshot.metadata.status == "archived"

    @pytest.mark.asyncio
    async def test_list_sessions(self, manager: SessionManager) -> None:
        """Should list all sessions."""
        for i in range(3):
            await manager.create_session(
                topic=f"Debate {i}",
                model="gpt-4o",
            )

        sessions = await manager.list_sessions()
        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_delete_session(self, manager: SessionManager) -> None:
        """Should delete a session."""
        session_id = await manager.create_session(
            topic="Test debate",
            model="gpt-4o",
        )
        deleted = await manager.delete_session(session_id)
        assert deleted is True

        snapshot = await manager.get_session(session_id)
        assert snapshot is None


class TestGenerateSessionId:
    """Tests for session ID generation."""

    def test_generates_unique_ids(self) -> None:
        """Should generate unique session IDs."""
        ids = {generate_session_id() for _ in range(100)}
        assert len(ids) == 100

    def test_id_format(self) -> None:
        """Session ID should be 8 characters."""
        session_id = generate_session_id()
        assert len(session_id) == 8


# ============================================================================
# MCP Server Tests
# ============================================================================


class TestServerState:
    """Tests for server state management."""

    def test_initial_state(self) -> None:
        """Server state should start empty."""
        state = ServerState()
        assert len(state.sessions) == 0
        assert state.max_sessions == 100

    def test_cleanup_old_sessions(self) -> None:
        """Should clean up old sessions."""
        from artemis.mcp.server import DebateSession

        state = ServerState()

        # Add old and new sessions
        old_session = MagicMock(spec=DebateSession)
        old_session.created_at = datetime.now() - timedelta(hours=48)

        new_session = MagicMock(spec=DebateSession)
        new_session.created_at = datetime.now()

        state.sessions["old"] = old_session
        state.sessions["new"] = new_session

        cleaned = state.cleanup_old_sessions(max_age_hours=24)
        assert cleaned == 1
        assert "old" not in state.sessions
        assert "new" in state.sessions


class TestArtemisMCPServer:
    """Tests for the MCP server."""

    @pytest.fixture
    def server(self) -> ArtemisMCPServer:
        """Create MCP server instance."""
        return ArtemisMCPServer(default_model="gpt-4o", max_sessions=10)

    def test_initialization(self, server: ArtemisMCPServer) -> None:
        """Should initialize with defaults."""
        assert server.default_model == "gpt-4o"
        assert server.state.max_sessions == 10

    def test_tools_property(self, server: ArtemisMCPServer) -> None:
        """Should expose available tools."""
        tools = server.tools
        assert len(tools) > 0
        tool_names = [t["name"] for t in tools]
        assert "artemis_debate_start" in tool_names

    @pytest.mark.asyncio
    async def test_handle_unknown_tool(self, server: ArtemisMCPServer) -> None:
        """Should raise error for unknown tool."""
        with pytest.raises(ValueError, match="Unknown tool"):
            await server.handle_tool_call("unknown_tool", {})

    @pytest.mark.asyncio
    async def test_handle_list_debates_empty(
        self,
        server: ArtemisMCPServer,
    ) -> None:
        """Should return empty list when no debates."""
        result = await server.handle_tool_call("artemis_list_debates", {})
        assert "debates" in result
        assert len(result["debates"]) == 0

    @pytest.mark.asyncio
    async def test_handle_debate_start(self, server: ArtemisMCPServer) -> None:
        """Should start a new debate."""
        with patch("artemis.mcp.server.Agent"), \
             patch("artemis.mcp.server.Debate") as mock_debate:
            mock_debate.return_value.assign_positions = MagicMock()

            result = await server.handle_tool_call(
                "artemis_debate_start",
                {"topic": "Test debate topic"},
            )

            assert "debate_id" in result
            assert result["topic"] == "Test debate topic"
            assert result["status"] == "active"
            assert len(server.state.sessions) == 1

    @pytest.mark.asyncio
    async def test_handle_debate_start_with_options(
        self,
        server: ArtemisMCPServer,
    ) -> None:
        """Should start debate with custom options."""
        with patch("artemis.mcp.server.Agent"), \
             patch("artemis.mcp.server.Debate") as mock_debate:
            mock_debate.return_value.assign_positions = MagicMock()

            result = await server.handle_tool_call(
                "artemis_debate_start",
                {
                    "topic": "Custom topic",
                    "pro_position": "supports A",
                    "con_position": "supports B",
                    "rounds": 5,
                    "model": "gpt-4-turbo",
                },
            )

            assert result["topic"] == "Custom topic"

    @pytest.mark.asyncio
    async def test_session_limit_enforcement(
        self,
        server: ArtemisMCPServer,
    ) -> None:
        """Should enforce session limit."""
        server.state.max_sessions = 2

        with patch("artemis.mcp.server.Agent"), \
             patch("artemis.mcp.server.Debate") as mock_debate:
            mock_debate.return_value.assign_positions = MagicMock()

            # Create 2 sessions
            for i in range(2):
                await server.handle_tool_call(
                    "artemis_debate_start",
                    {"topic": f"Topic {i}"},
                )

            # Third should fail
            with pytest.raises(ValueError, match="Maximum session limit"):
                await server.handle_tool_call(
                    "artemis_debate_start",
                    {"topic": "Topic 3"},
                )

    @pytest.mark.asyncio
    async def test_handle_jsonrpc_initialize(
        self,
        server: ArtemisMCPServer,
    ) -> None:
        """Should handle initialize request."""
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {},
            "id": 1,
        }
        response = await server._handle_jsonrpc(request)
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["serverInfo"]["name"] == "artemis-mcp-server"

    @pytest.mark.asyncio
    async def test_handle_jsonrpc_tools_list(
        self,
        server: ArtemisMCPServer,
    ) -> None:
        """Should handle tools/list request."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2,
        }
        response = await server._handle_jsonrpc(request)
        assert response["id"] == 2
        assert "result" in response
        assert "tools" in response["result"]

    @pytest.mark.asyncio
    async def test_handle_jsonrpc_tools_call(
        self,
        server: ArtemisMCPServer,
    ) -> None:
        """Should handle tools/call request."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "artemis_list_debates",
                "arguments": {},
            },
            "id": 3,
        }
        response = await server._handle_jsonrpc(request)
        assert response["id"] == 3
        assert "result" in response
        assert "content" in response["result"]

    @pytest.mark.asyncio
    async def test_handle_jsonrpc_unknown_method(
        self,
        server: ArtemisMCPServer,
    ) -> None:
        """Should return error for unknown method."""
        request = {
            "jsonrpc": "2.0",
            "method": "unknown/method",
            "params": {},
            "id": 4,
        }
        response = await server._handle_jsonrpc(request)
        assert response["id"] == 4
        assert "error" in response


class TestAnalyzeTopic:
    """Tests for topic analysis."""

    @pytest.fixture
    def server(self) -> ArtemisMCPServer:
        """Create MCP server instance."""
        return ArtemisMCPServer(default_model="gpt-4o")

    def test_build_analysis_prompt_quick(self, server: ArtemisMCPServer) -> None:
        """Should build quick analysis prompt."""
        prompt = server._build_analysis_prompt("AI ethics", "quick")
        assert "AI ethics" in prompt
        assert "2-3 key points" in prompt

    def test_build_analysis_prompt_moderate(self, server: ArtemisMCPServer) -> None:
        """Should build moderate analysis prompt."""
        prompt = server._build_analysis_prompt("AI ethics", "moderate")
        assert "3-5 points" in prompt

    def test_build_analysis_prompt_thorough(self, server: ArtemisMCPServer) -> None:
        """Should build thorough analysis prompt."""
        prompt = server._build_analysis_prompt("AI ethics", "thorough")
        assert "5+ points" in prompt

    def test_parse_analysis_valid_json(self, server: ArtemisMCPServer) -> None:
        """Should parse valid JSON response."""
        content = """
        Here's my analysis:
        {
            "pro_points": ["Point 1", "Point 2"],
            "con_points": ["Con 1"],
            "considerations": ["Consider this"],
            "recommended_rounds": 4
        }
        """
        result = server._parse_analysis(content)
        assert len(result["pro_points"]) == 2
        assert result["recommended_rounds"] == 4

    def test_parse_analysis_invalid_json(self, server: ArtemisMCPServer) -> None:
        """Should return defaults for invalid JSON."""
        content = "This is not valid JSON at all"
        result = server._parse_analysis(content)
        assert "pro_points" in result
        assert "con_points" in result
        assert result["recommended_rounds"] == 3


# ============================================================================
# Integration Tests
# ============================================================================


class TestMCPIntegration:
    """Integration tests for full MCP workflows."""

    @pytest.fixture
    def server(self) -> ArtemisMCPServer:
        """Create MCP server instance."""
        return ArtemisMCPServer(default_model="gpt-4o", max_sessions=100)

    @pytest.mark.asyncio
    async def test_full_debate_workflow(self, server: ArtemisMCPServer) -> None:
        """Test complete debate workflow via MCP."""
        with patch("artemis.mcp.server.Agent"), \
             patch("artemis.mcp.server.Debate") as mock_debate_class:
            # Setup mock debate
            mock_debate = MagicMock()
            mock_debate.assign_positions = MagicMock()
            mock_debate.get_transcript.return_value = []
            mock_debate.get_position.return_value = "pro position"
            mock_debate_class.return_value = mock_debate

            # Step 1: Start debate
            start_result = await server.handle_tool_call(
                "artemis_debate_start",
                {"topic": "Should AI be regulated?"},
            )
            debate_id = start_result["debate_id"]
            assert debate_id is not None

            # Step 2: List debates
            list_result = await server.handle_tool_call("artemis_list_debates", {})
            assert len(list_result["debates"]) == 1
            assert list_result["debates"][0]["debate_id"] == debate_id

            # Step 3: Get transcript (empty)
            transcript_result = await server.handle_tool_call(
                "artemis_get_transcript",
                {"debate_id": debate_id},
            )
            assert transcript_result["debate_id"] == debate_id
            assert len(transcript_result["turns"]) == 0

    @pytest.mark.asyncio
    async def test_multiple_debates(self, server: ArtemisMCPServer) -> None:
        """Test running multiple debates concurrently."""
        with patch("artemis.mcp.server.Agent"), \
             patch("artemis.mcp.server.Debate") as mock_debate_class:
            mock_debate = MagicMock()
            mock_debate.assign_positions = MagicMock()
            mock_debate_class.return_value = mock_debate

            # Start multiple debates
            topics = [
                "Topic A",
                "Topic B",
                "Topic C",
            ]
            debate_ids = []

            for topic in topics:
                result = await server.handle_tool_call(
                    "artemis_debate_start",
                    {"topic": topic},
                )
                debate_ids.append(result["debate_id"])

            # Verify all debates exist
            list_result = await server.handle_tool_call("artemis_list_debates", {})
            assert len(list_result["debates"]) == 3

            # Verify unique IDs
            assert len(set(debate_ids)) == 3
