"""
ARTEMIS Core Module

Contains the core ARTEMIS implementation:
- Debate orchestrator
- Agent with H-L-DAG argument generation
- L-AE-CR adaptive evaluation
- Jury scoring mechanism
- Ethics module
"""

from artemis.core.agent import Agent, DebateStrategy, OpponentModel, StrategyContext
from artemis.core.argument import ArgumentBuilder, ArgumentHierarchy, ArgumentParser
from artemis.core.causal import (
    CausalEdge,
    CausalExtractor,
    CausalGraph,
    CausalNode,
    LinkType,
)
from artemis.core.debate import Debate, DebateError, DebateHaltedError
from artemis.core.disagreement import DisagreementAnalyzer, DisagreementType
from artemis.core.ethics import (
    EthicalConcern,
    EthicalFramework,
    EthicsEvaluator,
    EthicsResult,
    FrameworkScore,
    StakeholderImpact,
    StakeholderType,
)
from artemis.core.evaluation import (
    AdaptationConfig,
    AdaptiveEvaluator,
    CriterionEvaluator,
    EvaluationDimension,
    RoundEvaluator,
    TopicAnalysis,
)
from artemis.core.evidence import (
    CredibilityLevel,
    EvidenceExtractor,
    EvidenceLinker,
    EvidenceType,
    ExtractedEvidence,
)
from artemis.core.feedback import FeedbackSummary, FeedbackSynthesizer
from artemis.core.jury import (
    ConsensusResult,
    JurorEvaluation,
    JuryConfig,
    JuryMember,
    JuryPanel,
)
from artemis.core.llm_evaluation import (
    EvaluatorFactory,
    LLMCriterionEvaluator,
)
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
    EvaluationMode,
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

__all__ = [
    # Enums
    "ArgumentLevel",
    "DebateState",
    "EvaluationMode",
    "JuryPerspective",
    "SafetyIndicatorType",
    "DebateStrategy",
    "EvidenceType",
    "CredibilityLevel",
    "LinkType",
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
    "OpponentModel",
    "StrategyContext",
    # Argument utilities
    "ArgumentBuilder",
    "ArgumentHierarchy",
    "ArgumentParser",
    # Evidence extraction
    "EvidenceExtractor",
    "EvidenceLinker",
    "ExtractedEvidence",
    # Causal reasoning
    "CausalExtractor",
    "CausalGraph",
    "CausalNode",
    "CausalEdge",
    # Evaluation
    "AdaptiveEvaluator",
    "AdaptationConfig",
    "CriterionEvaluator",
    "EvaluationDimension",
    "EvaluatorFactory",
    "LLMCriterionEvaluator",
    "RoundEvaluator",
    "TopicAnalysis",
    # Jury
    "JuryMember",
    "JuryPanel",
    "JuryConfig",
    "JurorEvaluation",
    "ConsensusResult",
    # Debate
    "Debate",
    "DebateError",
    "DebateHaltedError",
    # Ethics
    "EthicsEvaluator",
    "EthicsResult",
    "EthicalFramework",
    "EthicalConcern",
    "FrameworkScore",
    "StakeholderImpact",
    "StakeholderType",
    # Feedback
    "FeedbackSummary",
    "FeedbackSynthesizer",
    # Disagreement Analysis
    "DisagreementAnalyzer",
    "DisagreementType",
]
