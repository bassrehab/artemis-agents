"""
ARTEMIS Integrations Module

Framework integrations for popular AI/ML libraries:
- LangChain Tool
- LangGraph Node
- CrewAI Tool
"""

from artemis.integrations.langchain import (
    ArtemisDebateTool,
    DebateInput,
    DebateOutput,
    QuickDebate,
)
from artemis.integrations.langgraph import (
    ArtemisDebateNode,
    DebateNodeConfig,
    DebateNodeState,
    DebatePhase,
    create_debate_workflow,
)

# from artemis.integrations.crewai import ArtemisCrewTool

__all__ = [
    # LangChain
    "ArtemisDebateTool",
    "DebateInput",
    "DebateOutput",
    "QuickDebate",
    # LangGraph
    "ArtemisDebateNode",
    "DebateNodeConfig",
    "DebateNodeState",
    "DebatePhase",
    "create_debate_workflow",
]
