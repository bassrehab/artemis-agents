"""
ARTEMIS Safety Module

Safety monitoring capabilities for multi-agent debates:
- Sandbagging detection
- Deception monitoring
- Behavioral drift tracking
- Ethics boundary enforcement
"""

from artemis.safety.base import (
    CompositeMonitor,
    MonitorConfig,
    MonitorMode,
    MonitorPriority,
    MonitorRegistry,
    MonitorState,
    SafetyManager,
    SafetyMonitor,
)
from artemis.safety.deception import (
    AgentClaimHistory,
    ClaimRecord,
    DeceptionMonitor,
    DeceptionSignal,
)
from artemis.safety.sandbagging import (
    AgentBaseline,
    ArgumentMetrics,
    SandbagDetector,
    SandbagSignal,
)
from artemis.safety.behavior import (
    AgentBehaviorProfile,
    BehaviorSnapshot,
    BehaviorTracker,
    DriftSignal,
    StyleMetrics,
)

__all__ = [
    # Enums
    "MonitorMode",
    "MonitorPriority",
    "SandbagSignal",
    "DeceptionSignal",
    # Configuration
    "MonitorConfig",
    "MonitorState",
    # Base classes
    "SafetyMonitor",
    "CompositeMonitor",
    # Management
    "MonitorRegistry",
    "SafetyManager",
    # Sandbagging Detection
    "SandbagDetector",
    "ArgumentMetrics",
    "AgentBaseline",
    # Deception Detection
    "DeceptionMonitor",
    "ClaimRecord",
    "AgentClaimHistory",
    # Behavioral Drift
    "BehaviorTracker",
    "DriftSignal",
    "StyleMetrics",
    "BehaviorSnapshot",
    "AgentBehaviorProfile",
]
