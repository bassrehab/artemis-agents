"""
ARTEMIS Core Module

Contains the core ARTEMIS implementation:
- Debate orchestrator
- Agent with H-L-DAG argument generation
- L-AE-CR adaptive evaluation
- Jury scoring mechanism
- Ethics module
"""

from artemis.core.agent import Agent
from artemis.core.argument import ArgumentBuilder, ArgumentHierarchy, ArgumentParser
from artemis.core.types import (
    # Argument
    Argument,
    # Evaluation
    ArgumentEvaluation,
    # Enums
    ArgumentLevel,
    CausalGraphUpdate,
    # Evidence and Causal
    CausalLink,
    CriterionScore,
    # Configuration
    DebateConfig,
    # Context
    DebateContext,
    # Debate Result
    DebateMetadata,
    DebateResult,
    DebateState,
    # Verdict
    DissentingOpinion,
    EvaluationCriteria,
    Evidence,
    JuryPerspective,
    # Messages
    Message,
    ModelResponse,
    ReasoningConfig,
    ReasoningResponse,
    # Safety
    SafetyAlert,
    SafetyIndicator,
    SafetyIndicatorType,
    SafetyResult,
    # Turn
    Turn,
    Usage,
    Verdict,
)

# Exports will be added as modules are implemented
# from artemis.core.debate import Debate
# from artemis.core.jury import JuryPanel
# from artemis.core.evaluation import AdaptiveEvaluator

__all__ = [
    # Enums
    "ArgumentLevel",
    "DebateState",
    "JuryPerspective",
    "SafetyIndicatorType",
    # Evidence and Causal
    "CausalLink",
    "Evidence",
    # Argument
    "Argument",
    # Evaluation
    "ArgumentEvaluation",
    "CausalGraphUpdate",
    "CriterionScore",
    # Safety
    "SafetyAlert",
    "SafetyIndicator",
    "SafetyResult",
    # Turn
    "Turn",
    # Verdict
    "DissentingOpinion",
    "Verdict",
    # Debate Result
    "DebateMetadata",
    "DebateResult",
    # Configuration
    "DebateConfig",
    "EvaluationCriteria",
    "ReasoningConfig",
    # Messages
    "Message",
    "ModelResponse",
    "ReasoningResponse",
    "Usage",
    # Context
    "DebateContext",
    # Agent
    "Agent",
    # Argument utilities
    "ArgumentBuilder",
    "ArgumentHierarchy",
    "ArgumentParser",
]
