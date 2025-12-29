"""ARTEMIS Steering Vectors Package.

Provides runtime behavior modification for debate agents through
steering vectors - multi-dimensional style controls.
"""

from artemis.steering.analyzer import SteeringEffectivenessAnalyzer
from artemis.steering.controller import SteeringController
from artemis.steering.formatter import SteeringFormatter
from artemis.steering.presets import PRESET_VECTORS, get_preset
from artemis.steering.vectors import (
    SteeringConfig,
    SteeringMode,
    SteeringVector,
)

__all__ = [
    # Core types
    "SteeringVector",
    "SteeringConfig",
    "SteeringMode",
    # Controller
    "SteeringController",
    # Formatter
    "SteeringFormatter",
    # Analyzer
    "SteeringEffectivenessAnalyzer",
    # Presets
    "PRESET_VECTORS",
    "get_preset",
]
