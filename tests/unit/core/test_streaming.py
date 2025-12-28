"""Tests for streaming debate functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artemis.core.streaming import (
    ConsoleStreamCallback,
    StreamCallback,
    StreamingDebate,
)
from artemis.core.types import (
    Argument,
    ArgumentEvaluation,
    ArgumentLevel,
    DebateConfig,
    StreamEvent,
    StreamEventType,
    Verdict,
)


class MockStreamCallback(StreamCallback):
    """A mock callback that collects events."""

    def __init__(self) -> None:
        self.events: list[StreamEvent] = []

    async def on_event(self, event: StreamEvent) -> None:
        self.events.append(event)


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = MagicMock()
    agent.name = "TestAgent"
    agent.set_position = MagicMock()
    agent.observe_opponent = MagicMock()
    agent.generate_argument = AsyncMock(
        return_value=Argument(
            agent="TestAgent",
            level=ArgumentLevel.STRATEGIC,
            content="Test argument content",
            evidence=[],
            causal_links=[],
        )
    )
    return agent


@pytest.fixture
def mock_agents(mock_agent):
    """Create two mock agents."""
    agent1 = MagicMock()
    agent1.name = "Agent1"
    agent1.set_position = MagicMock()
    agent1.observe_opponent = MagicMock()
    agent1.generate_argument = AsyncMock(
        return_value=Argument(
            agent="Agent1",
            level=ArgumentLevel.STRATEGIC,
            content="Agent1 argument",
            evidence=[],
            causal_links=[],
        )
    )

    agent2 = MagicMock()
    agent2.name = "Agent2"
    agent2.set_position = MagicMock()
    agent2.observe_opponent = MagicMock()
    agent2.generate_argument = AsyncMock(
        return_value=Argument(
            agent="Agent2",
            level=ArgumentLevel.STRATEGIC,
            content="Agent2 argument",
            evidence=[],
            causal_links=[],
        )
    )

    return [agent1, agent2]


class TestStreamEventType:
    """Tests for StreamEventType enum."""

    def test_event_types_exist(self):
        """All required event types should exist."""
        assert StreamEventType.DEBATE_START
        assert StreamEventType.ROUND_START
        assert StreamEventType.TURN_START
        assert StreamEventType.CHUNK
        assert StreamEventType.ARGUMENT_COMPLETE
        assert StreamEventType.EVALUATION_START
        assert StreamEventType.EVALUATION_COMPLETE
        assert StreamEventType.TURN_COMPLETE
        assert StreamEventType.ROUND_COMPLETE
        assert StreamEventType.SAFETY_CHECK
        assert StreamEventType.SAFETY_ALERT
        assert StreamEventType.VERDICT
        assert StreamEventType.DEBATE_END
        assert StreamEventType.ERROR

    def test_event_types_are_strings(self):
        """Event types should have string values."""
        assert StreamEventType.CHUNK.value == "chunk"
        assert StreamEventType.DEBATE_START.value == "debate_start"


class TestStreamEvent:
    """Tests for StreamEvent model."""

    def test_create_chunk_event(self):
        """Should create a chunk event."""
        event = StreamEvent(
            event_type=StreamEventType.CHUNK,
            agent="TestAgent",
            content="Test content",
            round_num=1,
            turn_num=0,
        )
        assert event.event_type == StreamEventType.CHUNK
        assert event.agent == "TestAgent"
        assert event.content == "Test content"
        assert event.round_num == 1

    def test_create_debate_start_event(self):
        """Should create a debate start event."""
        event = StreamEvent(
            event_type=StreamEventType.DEBATE_START,
            metadata={"topic": "Test topic"},
        )
        assert event.event_type == StreamEventType.DEBATE_START
        assert event.metadata["topic"] == "Test topic"

    def test_event_has_timestamp(self):
        """Events should have timestamps."""
        event = StreamEvent(event_type=StreamEventType.CHUNK)
        assert event.timestamp is not None


class TestStreamCallback:
    """Tests for StreamCallback."""

    def test_mock_callback_collects_events(self):
        """Mock callback should collect events."""
        callback = MockStreamCallback()
        event = StreamEvent(event_type=StreamEventType.CHUNK, content="test")
        asyncio.run(callback.on_event(event))
        assert len(callback.events) == 1
        assert callback.events[0].content == "test"


class TestConsoleStreamCallback:
    """Tests for ConsoleStreamCallback."""

    def test_create_console_callback(self):
        """Should create console callback."""
        callback = ConsoleStreamCallback()
        assert callback.verbose is False

    def test_create_verbose_callback(self):
        """Should create verbose console callback."""
        callback = ConsoleStreamCallback(verbose=True)
        assert callback.verbose is True

    @patch("builtins.print")
    def test_prints_chunk_content(self, mock_print):
        """Should print chunk content."""
        callback = ConsoleStreamCallback()
        event = StreamEvent(
            event_type=StreamEventType.CHUNK,
            content="Test content",
        )
        asyncio.run(callback.on_event(event))
        mock_print.assert_called_once_with("Test content", end="", flush=True)


@pytest.fixture
def mock_jury():
    """Create a mock jury."""
    jury = MagicMock()
    jury.deliberate = AsyncMock(
        return_value=Verdict(
            decision="Agent1",
            confidence=0.8,
            reasoning="Test reasoning",
        )
    )
    jury.__len__ = MagicMock(return_value=3)
    return jury


@pytest.fixture
def mock_evaluator():
    """Create a mock evaluator."""
    evaluator = AsyncMock()
    evaluator.evaluate_argument = AsyncMock(
        return_value=ArgumentEvaluation(
            argument_id="test",
            scores={"logical_coherence": 0.8},
            weights={"logical_coherence": 1.0},
            causal_score=0.7,
            total_score=0.8,
        )
    )
    return evaluator


class TestStreamingDebate:
    """Tests for StreamingDebate."""

    def test_create_streaming_debate(self, mock_agents, mock_jury, mock_evaluator):
        """Should create a streaming debate."""
        debate = StreamingDebate(
            topic="Test topic",
            agents=mock_agents,
            rounds=3,
            jury=mock_jury,
            evaluator=mock_evaluator,
        )
        assert debate.topic == "Test topic"
        assert debate.total_rounds == 3
        assert len(debate.agents) == 2

    def test_requires_at_least_two_agents(self, mock_agent):
        """Should require at least two agents."""
        with pytest.raises(ValueError, match="need at least 2 agents"):
            StreamingDebate(
                topic="Test topic",
                agents=[mock_agent],
            )

    def test_add_callback(self, mock_agents, mock_jury, mock_evaluator):
        """Should add callbacks."""
        debate = StreamingDebate(
            topic="Test topic",
            agents=mock_agents,
            jury=mock_jury,
            evaluator=mock_evaluator,
        )
        callback = MockStreamCallback()
        debate.add_callback(callback)
        assert callback in debate._callbacks

    def test_assign_positions(self, mock_agents, mock_jury, mock_evaluator):
        """Should assign positions to agents."""
        debate = StreamingDebate(
            topic="Test topic",
            agents=mock_agents,
            jury=mock_jury,
            evaluator=mock_evaluator,
        )
        debate.assign_positions({"Agent1": "Pro", "Agent2": "Con"})
        assert debate._agent_positions["Agent1"] == "Pro"
        assert debate._agent_positions["Agent2"] == "Con"

    @pytest.mark.asyncio
    async def test_run_streaming_yields_events(self, mock_agents, mock_jury, mock_evaluator):
        """Should yield streaming events during debate."""
        debate = StreamingDebate(
            topic="Test topic",
            agents=mock_agents,
            rounds=1,
            evaluator=mock_evaluator,
            jury=mock_jury,
        )

        events: list[StreamEvent] = []
        async for event in debate.run_streaming():
            events.append(event)

        # Should have debate start event
        assert any(e.event_type == StreamEventType.DEBATE_START for e in events)

        # Should have round events
        assert any(e.event_type == StreamEventType.ROUND_START for e in events)
        assert any(e.event_type == StreamEventType.ROUND_COMPLETE for e in events)

        # Should have turn events
        assert any(e.event_type == StreamEventType.TURN_START for e in events)
        assert any(e.event_type == StreamEventType.TURN_COMPLETE for e in events)

        # Should have argument events
        assert any(e.event_type == StreamEventType.CHUNK for e in events)
        assert any(e.event_type == StreamEventType.ARGUMENT_COMPLETE for e in events)

        # Should have evaluation events
        assert any(e.event_type == StreamEventType.EVALUATION_START for e in events)
        assert any(e.event_type == StreamEventType.EVALUATION_COMPLETE for e in events)

        # Should have verdict and debate end
        assert any(e.event_type == StreamEventType.VERDICT for e in events)
        assert any(e.event_type == StreamEventType.DEBATE_END for e in events)

    @pytest.mark.asyncio
    async def test_callbacks_receive_events(self, mock_agents, mock_jury, mock_evaluator):
        """Callbacks should receive all events."""
        callback = MockStreamCallback()

        debate = StreamingDebate(
            topic="Test topic",
            agents=mock_agents,
            rounds=1,
            evaluator=mock_evaluator,
            jury=mock_jury,
            callbacks=[callback],
        )

        # Consume all events
        async for _ in debate.run_streaming():
            pass

        # Callback should have received events
        assert len(callback.events) > 0
        assert any(e.event_type == StreamEventType.DEBATE_START for e in callback.events)

    @pytest.mark.asyncio
    async def test_chunk_events_contain_content(self, mock_agents, mock_jury, mock_evaluator):
        """Chunk events should contain argument content."""
        debate = StreamingDebate(
            topic="Test topic",
            agents=mock_agents,
            rounds=1,
            evaluator=mock_evaluator,
            jury=mock_jury,
        )

        events: list[StreamEvent] = []
        async for event in debate.run_streaming():
            events.append(event)

        chunk_events = [e for e in events if e.event_type == StreamEventType.CHUNK]
        assert len(chunk_events) > 0
        for chunk in chunk_events:
            assert chunk.content is not None
            assert len(chunk.content) > 0

    @pytest.mark.asyncio
    async def test_debate_end_contains_metadata(self, mock_agents, mock_jury, mock_evaluator):
        """Debate end event should contain metadata."""
        debate = StreamingDebate(
            topic="Test topic",
            agents=mock_agents,
            rounds=1,
            evaluator=mock_evaluator,
            jury=mock_jury,
        )

        events: list[StreamEvent] = []
        async for event in debate.run_streaming():
            events.append(event)

        end_events = [e for e in events if e.event_type == StreamEventType.DEBATE_END]
        assert len(end_events) == 1
        assert "duration_seconds" in end_events[0].metadata
        assert "total_turns" in end_events[0].metadata

    def test_repr(self, mock_agents, mock_jury, mock_evaluator):
        """Should have a useful repr."""
        debate = StreamingDebate(
            topic="Test topic for debate",
            agents=mock_agents,
            jury=mock_jury,
            evaluator=mock_evaluator,
        )
        repr_str = repr(debate)
        assert "StreamingDebate" in repr_str
        assert "Test topic" in repr_str
