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
from artemis.core.causal_analysis import (
    CausalAnalyzer,
    CausalFallacyDetector,
)
from artemis.core.causal_strategy import (
    CausalStrategy,
    DefensePriority,
    OpponentStrategyProfile,
    PredictedTarget,
    RebuttalSuggestion,
    VulnerableClaim,
)
from artemis.core.causal_visualization import (
    CausalVisualizer,
    create_snapshot,
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
from artemis.core.streaming import (
    ConsoleStreamCallback,
    StreamCallback,
    StreamingDebate,
)
from artemis.core.aggregation import (
    ConfidenceWeightedAggregator,
    MajorityVoteAggregator,
    UnanimousAggregator,
    VerdictAggregator,
    WeightedAverageAggregator,
    WeightedMajorityAggregator,
    create_aggregator,
)
from artemis.core.decomposition import (
    HybridDecomposer,
    LLMTopicDecomposer,
    ManualDecomposer,
    RuleBasedDecomposer,
    TopicDecomposer,
)
from artemis.core.hierarchical import HierarchicalDebate
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
    # Causal Analysis Types (v2)
    ArgumentStrengthScore,
    AttackTarget,
    CausalAnalysisResult,
    CircularReasoningResult,
    ContradictionResult,
    CriticalNodeResult,
    FallacyResult,
    FallacyType,
    GraphSnapshot,
    ReasoningGap,
    ReinforcementSuggestion,
    WeakLinkResult,
    # Streaming Types
    StreamEvent,
    StreamEventType,
    # Hierarchical Debate Types
    AggregationMethod,
    CompoundVerdict,
    DecompositionStrategy,
    HierarchicalContext,
    HierarchyLevel,
    SubDebateSpec,
    # Multimodal Types
    ContentPart,
    ContentType,
)
from artemis.core.multimodal_evidence import (
    DocumentProcessor,
    ExtractedContent,
    ExtractionType,
    ImageAnalyzer,
    MultimodalEvidenceExtractor,
)
from artemis.core.verification import (
    ArgumentVerifier,
    CausalChainRule,
    CitationRule,
    CitationValidator,
    EvidenceSupportRule,
    FallacyFreeRule,
    LogicalConsistencyRule,
    VerificationError,
    VerificationRuleBase,
)
from artemis.core.types import (
    VerificationReport,
    VerificationResult,
    VerificationRule,
    VerificationRuleType,
    VerificationSpec,
    VerificationViolation,
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
    # Causal Analysis (v2)
    "CausalAnalyzer",
    "CausalFallacyDetector",
    "CausalStrategy",
    "CausalVisualizer",
    "DefensePriority",
    "OpponentStrategyProfile",
    "PredictedTarget",
    "RebuttalSuggestion",
    "VulnerableClaim",
    "create_snapshot",
    # Causal Analysis Types (v2)
    "ArgumentStrengthScore",
    "AttackTarget",
    "CausalAnalysisResult",
    "CircularReasoningResult",
    "ContradictionResult",
    "CriticalNodeResult",
    "FallacyResult",
    "FallacyType",
    "GraphSnapshot",
    "ReasoningGap",
    "ReinforcementSuggestion",
    "WeakLinkResult",
    # Streaming
    "StreamingDebate",
    "StreamCallback",
    "ConsoleStreamCallback",
    "StreamEvent",
    "StreamEventType",
    # Hierarchical Debates
    "HierarchicalDebate",
    "TopicDecomposer",
    "ManualDecomposer",
    "RuleBasedDecomposer",
    "LLMTopicDecomposer",
    "HybridDecomposer",
    "VerdictAggregator",
    "WeightedAverageAggregator",
    "MajorityVoteAggregator",
    "ConfidenceWeightedAggregator",
    "UnanimousAggregator",
    "WeightedMajorityAggregator",
    "create_aggregator",
    # Hierarchical Types
    "HierarchyLevel",
    "SubDebateSpec",
    "HierarchicalContext",
    "CompoundVerdict",
    "DecompositionStrategy",
    "AggregationMethod",
    # Multimodal
    "ContentType",
    "ContentPart",
    "MultimodalEvidenceExtractor",
    "ExtractedContent",
    "ExtractionType",
    "DocumentProcessor",
    "ImageAnalyzer",
    # Verification
    "ArgumentVerifier",
    "VerificationError",
    "VerificationRuleBase",
    "CausalChainRule",
    "CitationRule",
    "LogicalConsistencyRule",
    "EvidenceSupportRule",
    "FallacyFreeRule",
    "CitationValidator",
    # Verification Types
    "VerificationRuleType",
    "VerificationRule",
    "VerificationSpec",
    "VerificationResult",
    "VerificationReport",
    "VerificationViolation",
]
