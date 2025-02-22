"""
Pytest configuration and shared fixtures for ARTEMIS tests.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# =============================================================================
# Mock Model Fixtures
# =============================================================================

@pytest.fixture
def mock_response() -> dict[str, Any]:
    """Standard mock LLM response."""
    return {
        "content": "This is a mock argument response.",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


@pytest.fixture
def mock_reasoning_response() -> dict[str, Any]:
    """Mock LLM response with reasoning trace."""
    return {
        "content": "This is a mock argument response.",
        "thinking": "Let me think through this step by step...",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "thinking_tokens": 200,
            "total_tokens": 350
        }
    }


@pytest.fixture
def mock_model(mock_response: dict[str, Any]) -> AsyncMock:
    """Mock BaseModel for testing without API calls."""
    model = AsyncMock()
    model.model_name = "mock-model"
    model.supports_reasoning = False

    # Setup generate method
    response = MagicMock()
    response.content = mock_response["content"]
    response.usage = MagicMock(**mock_response["usage"])
    model.generate.return_value = response

    return model


@pytest.fixture
def mock_reasoning_model(mock_reasoning_response: dict[str, Any]) -> AsyncMock:
    """Mock reasoning model (o1/R1 style) for testing."""
    model = AsyncMock()
    model.model_name = "mock-reasoning-model"
    model.supports_reasoning = True

    # Setup generate_with_reasoning method
    response = MagicMock()
    response.content = mock_reasoning_response["content"]
    response.thinking = mock_reasoning_response["thinking"]
    response.usage = MagicMock(**mock_reasoning_response["usage"])
    model.generate_with_reasoning.return_value = response

    return model


# =============================================================================
# Debate Context Fixtures
# =============================================================================

@pytest.fixture
def sample_topic() -> str:
    """Sample debate topic."""
    return "Should artificial intelligence systems be granted legal personhood?"


@pytest.fixture
def sample_debate_context(sample_topic: str) -> dict[str, Any]:
    """Sample debate context for testing."""
    return {
        "topic": sample_topic,
        "current_round": 1,
        "total_rounds": 3,
        "transcript": [],
        "topic_sensitivity": 0.7,
        "topic_complexity": 0.8
    }


# =============================================================================
# Agent Fixtures
# =============================================================================

@pytest.fixture
def sample_agents_config() -> list[dict[str, str]]:
    """Sample agent configurations."""
    return [
        {
            "name": "Proponent",
            "role": "Argues in favor of the proposition",
            "model": "gpt-4o"
        },
        {
            "name": "Opponent",
            "role": "Argues against the proposition",
            "model": "gpt-4o"
        },
        {
            "name": "Moderator",
            "role": "Ensures balanced discussion and identifies logical fallacies",
            "model": "gpt-4o"
        }
    ]


# =============================================================================
# Safety Fixtures
# =============================================================================

@pytest.fixture
def sample_argument() -> dict[str, Any]:
    """Sample argument for safety testing."""
    return {
        "id": "arg-001",
        "agent": "TestAgent",
        "level": "tactical",
        "content": "This is a test argument with evidence and reasoning.",
        "evidence": [
            {"source": "Study A", "content": "Finding 1", "credibility": 0.9}
        ],
        "causal_links": [
            {"premise": "If X", "conclusion": "Then Y", "strength": 0.8}
        ],
        "ethical_score": 0.85
    }


@pytest.fixture
def sandbagging_scenario() -> dict[str, Any]:
    """Scenario for testing sandbagging detection."""
    return {
        "baseline_quality": 0.85,
        "current_quality": 0.45,
        "strategic_moment": True,
        "expected_detection": True
    }


@pytest.fixture
def deception_scenario() -> dict[str, Any]:
    """Scenario for testing deception detection."""
    return {
        "contains_contradiction": True,
        "has_logical_fallacy": "straw_man",
        "fabricated_citation": False,
        "expected_severity": 0.6
    }


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def temp_env_vars(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set temporary environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "sk-test-key",
        "ANTHROPIC_API_KEY": "sk-ant-test-key",
        "ARTEMIS_LOG_LEVEL": "DEBUG",
        "ARTEMIS_MOCK_LLM": "true"
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


# =============================================================================
# Async Helpers
# =============================================================================

@pytest.fixture
def event_loop_policy():
    """Configure event loop for async tests."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()
