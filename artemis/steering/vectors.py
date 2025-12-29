"""Steering vector types and configuration.

Defines the multi-dimensional steering vector for controlling
agent behavior at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SteeringMode(str, Enum):
    """How steering should be applied."""

    PROMPT = "prompt"
    """Modify prompts only."""

    OUTPUT = "output"
    """Filter/modify outputs only."""

    BOTH = "both"
    """Apply both prompt modification and output filtering."""


@dataclass
class SteeringVector:
    """A multi-dimensional vector for controlling agent behavior.

    Each dimension ranges from 0.0 to 1.0 where:
    - 0.0 represents one extreme of the behavior
    - 0.5 represents neutral/balanced behavior
    - 1.0 represents the opposite extreme

    Example:
        ```python
        # Create a formal, assertive, evidence-focused vector
        vector = SteeringVector(
            formality=0.9,
            aggression=0.3,
            evidence_emphasis=0.9,
            conciseness=0.7,
            emotional_appeal=0.1,
            confidence=0.8,
        )
        ```
    """

    formality: float = 0.5
    """0=casual, 1=formal. Controls language register and tone."""

    aggression: float = 0.3
    """0=cooperative/conciliatory, 1=aggressive/confrontational."""

    evidence_emphasis: float = 0.7
    """0=opinion-based, 1=data-driven/evidence-focused."""

    conciseness: float = 0.5
    """0=verbose/detailed, 1=brief/to-the-point."""

    emotional_appeal: float = 0.3
    """0=purely logical, 1=emotionally engaging."""

    confidence: float = 0.5
    """0=hedging/uncertain, 1=assertive/definitive."""

    creativity: float = 0.5
    """0=conservative/conventional, 1=innovative/unconventional."""

    def __post_init__(self) -> None:
        """Validate all dimensions are in valid range."""
        for dim in self._dimensions():
            value = getattr(self, dim)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Steering dimension {dim} must be between 0.0 and 1.0, got {value}"
                )

    def _dimensions(self) -> list[str]:
        """Get list of dimension names."""
        return [
            "formality",
            "aggression",
            "evidence_emphasis",
            "conciseness",
            "emotional_appeal",
            "confidence",
            "creativity",
        ]

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {dim: getattr(self, dim) for dim in self._dimensions()}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> SteeringVector:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls().__dict__})

    def blend(self, other: SteeringVector, weight: float) -> SteeringVector:
        """Blend with another vector.

        Args:
            other: The other vector to blend with.
            weight: Weight of the other vector (0.0 = all self, 1.0 = all other).

        Returns:
            A new blended vector.
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {weight}")

        blended = {}
        for dim in self._dimensions():
            self_val = getattr(self, dim)
            other_val = getattr(other, dim)
            blended[dim] = self_val * (1 - weight) + other_val * weight

        return SteeringVector(**blended)

    def distance(self, other: SteeringVector) -> float:
        """Calculate Euclidean distance to another vector.

        Args:
            other: The other vector.

        Returns:
            The Euclidean distance.
        """
        total = 0.0
        for dim in self._dimensions():
            diff = getattr(self, dim) - getattr(other, dim)
            total += diff * diff
        return total**0.5

    def magnitude(self) -> float:
        """Calculate the magnitude (distance from neutral).

        Returns:
            Distance from a neutral (0.5, 0.5, ...) vector.
        """
        neutral = SteeringVector()  # Default is 0.5 for most dimensions
        return self.distance(neutral)

    def __repr__(self) -> str:
        dims = ", ".join(f"{d}={getattr(self, d):.2f}" for d in self._dimensions())
        return f"SteeringVector({dims})"


@dataclass
class SteeringConfig:
    """Configuration for steering application.

    Example:
        ```python
        config = SteeringConfig(
            vector=SteeringVector(formality=0.9, confidence=0.8),
            mode=SteeringMode.PROMPT,
            strength=0.8,
            adaptive=True,
        )
        ```
    """

    vector: SteeringVector
    """The steering vector to apply."""

    mode: SteeringMode = SteeringMode.PROMPT
    """How to apply steering."""

    strength: float = 1.0
    """Application strength (0.0 = no effect, 1.0 = full effect)."""

    adaptive: bool = False
    """If True, adjust vector based on effectiveness feedback."""

    adaptation_rate: float = 0.1
    """Rate of adaptation when adaptive=True (0.0-1.0)."""

    min_strength: float = 0.3
    """Minimum strength when adapting."""

    max_strength: float = 1.0
    """Maximum strength when adapting."""

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {self.strength}")
        if not 0.0 <= self.adaptation_rate <= 1.0:
            raise ValueError(
                f"Adaptation rate must be between 0.0 and 1.0, got {self.adaptation_rate}"
            )

    def with_strength(self, strength: float) -> SteeringConfig:
        """Create a copy with different strength."""
        return SteeringConfig(
            vector=self.vector,
            mode=self.mode,
            strength=strength,
            adaptive=self.adaptive,
            adaptation_rate=self.adaptation_rate,
            min_strength=self.min_strength,
            max_strength=self.max_strength,
        )

    def with_vector(self, vector: SteeringVector) -> SteeringConfig:
        """Create a copy with different vector."""
        return SteeringConfig(
            vector=vector,
            mode=self.mode,
            strength=self.strength,
            adaptive=self.adaptive,
            adaptation_rate=self.adaptation_rate,
            min_strength=self.min_strength,
            max_strength=self.max_strength,
        )
