"""Steering controller for applying and adapting steering vectors.

Manages the application of steering vectors to agent prompts and
optionally adapts the vector based on effectiveness feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from artemis.steering.vectors import SteeringConfig, SteeringMode, SteeringVector
from artemis.steering.formatter import SteeringFormatter

if TYPE_CHECKING:
    from artemis.steering.analyzer import StyleMetrics


@dataclass
class SteeringApplication:
    """Record of a steering application."""

    timestamp: datetime
    vector: SteeringVector
    strength: float
    prompt_before: str
    prompt_after: str
    output: str | None = None
    metrics: "StyleMetrics | None" = None


class SteeringController:
    """Controls the application of steering vectors to agent behavior."""

    def __init__(
        self,
        config: SteeringConfig,
        formatter: SteeringFormatter | None = None,
    ) -> None:
        """Initialize the controller."""
        self.config = config
        self.formatter = formatter or SteeringFormatter()
        self._history: list[SteeringApplication] = []
        self._current_application: SteeringApplication | None = None

    @property
    def vector(self) -> SteeringVector:
        """Current steering vector."""
        return self.config.vector

    @property
    def strength(self) -> float:
        """Current application strength."""
        return self.config.strength

    def apply_to_prompt(self, prompt: str) -> str:
        """Apply steering to a prompt."""
        if self.config.mode == SteeringMode.OUTPUT:
            # Output mode doesn't modify prompts
            return prompt

        instructions = self.formatter.format_instructions(
            self.config.vector,
            self.config.strength,
        )

        if not instructions:
            return prompt

        # Prepend instructions to prompt
        modified = f"{instructions}\n\n{prompt}"

        # Record application
        self._current_application = SteeringApplication(
            timestamp=datetime.utcnow(),
            vector=self.config.vector,
            strength=self.config.strength,
            prompt_before=prompt,
            prompt_after=modified,
        )

        return modified

    def apply_to_system_prompt(self, system_prompt: str) -> str:
        """Apply steering to a system prompt."""
        if self.config.mode == SteeringMode.OUTPUT:
            return system_prompt

        addon = self.formatter.format_system_prompt_addon(
            self.config.vector,
            self.config.strength,
        )

        if not addon:
            return system_prompt

        return system_prompt + addon

    def record_output(self, output: str) -> None:
        """Record an output for effectiveness analysis."""
        if self._current_application:
            self._current_application.output = output
            self._history.append(self._current_application)
            self._current_application = None

            # Adapt if enabled
            if self.config.adaptive:
                self._adapt(output)

    def _adapt(self, output: str) -> None:
        # Import here to avoid circular import
        from artemis.steering.analyzer import SteeringEffectivenessAnalyzer

        analyzer = SteeringEffectivenessAnalyzer()
        metrics = analyzer.analyze_output(output)

        # Calculate effectiveness (how close output is to target)
        target_vector = self.config.vector
        effectiveness = analyzer.calculate_effectiveness(metrics, target_vector)

        # Record metrics
        if self._history:
            self._history[-1].metrics = metrics

        # Adjust strength based on effectiveness
        if effectiveness < 0.5:
            # Low effectiveness - increase strength
            new_strength = min(
                self.config.max_strength,
                self.config.strength + self.config.adaptation_rate,
            )
        elif effectiveness > 0.8:
            # High effectiveness - can reduce strength slightly
            new_strength = max(
                self.config.min_strength,
                self.config.strength - self.config.adaptation_rate * 0.5,
            )
        else:
            # Acceptable effectiveness - maintain
            new_strength = self.config.strength

        self.config = self.config.with_strength(new_strength)

    def get_effectiveness_report(self) -> dict:
        """Generate an effectiveness report."""
        if not self._history:
            return {
                "applications": 0,
                "average_effectiveness": None,
                "current_strength": self.config.strength,
            }

        # Import here to avoid circular import
        from artemis.steering.analyzer import SteeringEffectivenessAnalyzer

        analyzer = SteeringEffectivenessAnalyzer()
        effectivenesses = []

        for app in self._history:
            if app.metrics:
                eff = analyzer.calculate_effectiveness(app.metrics, app.vector)
                effectivenesses.append(eff)

        avg_effectiveness = (
            sum(effectivenesses) / len(effectivenesses)
            if effectivenesses
            else None
        )

        return {
            "applications": len(self._history),
            "average_effectiveness": avg_effectiveness,
            "current_strength": self.config.strength,
            "strength_history": [app.strength for app in self._history],
            "vector": self.config.vector.to_dict(),
        }

    def reset_history(self) -> None:
        # lol just nuke everything
        self._history = []
        self._current_application = None

    def set_vector(self, vector: SteeringVector) -> None:
        """Update the steering vector."""
        self.config = self.config.with_vector(vector)

    def set_strength(self, strength: float) -> None:
        """Update the application strength."""
        self.config = self.config.with_strength(strength)

    def __repr__(self) -> str:
        return (
            f"SteeringController(strength={self.config.strength:.2f}, "
            f"mode={self.config.mode.value}, applications={len(self._history)})"
        )
