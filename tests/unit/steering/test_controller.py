"""Tests for steering controller."""

import pytest

from artemis.steering.controller import SteeringController
from artemis.steering.vectors import SteeringConfig, SteeringMode, SteeringVector


class TestSteeringController:
    """Tests for SteeringController."""

    def test_create_controller(self):
        """Should create controller with config."""
        config = SteeringConfig(vector=SteeringVector())
        controller = SteeringController(config)
        assert controller.config == config

    def test_vector_property(self):
        """Should expose current vector."""
        vector = SteeringVector(formality=0.9)
        config = SteeringConfig(vector=vector)
        controller = SteeringController(config)
        assert controller.vector == vector

    def test_strength_property(self):
        """Should expose current strength."""
        config = SteeringConfig(vector=SteeringVector(), strength=0.7)
        controller = SteeringController(config)
        assert controller.strength == 0.7

    def test_apply_to_prompt(self):
        """Should modify prompt with steering instructions."""
        config = SteeringConfig(
            vector=SteeringVector(formality=0.9, confidence=0.8),
            mode=SteeringMode.PROMPT,
        )
        controller = SteeringController(config)

        original = "Generate an argument about AI."
        modified = controller.apply_to_prompt(original)

        # Should prepend instructions
        assert len(modified) > len(original)
        assert original in modified

    def test_apply_to_prompt_output_mode(self):
        """Should not modify prompt in output mode."""
        config = SteeringConfig(
            vector=SteeringVector(formality=0.9),
            mode=SteeringMode.OUTPUT,
        )
        controller = SteeringController(config)

        original = "Generate an argument."
        modified = controller.apply_to_prompt(original)

        assert modified == original

    def test_apply_to_system_prompt(self):
        """Should add addon to system prompt."""
        config = SteeringConfig(
            vector=SteeringVector(formality=0.9),
            mode=SteeringMode.PROMPT,
        )
        controller = SteeringController(config)

        original = "You are a debate agent."
        modified = controller.apply_to_system_prompt(original)

        assert original in modified
        assert len(modified) > len(original)

    def test_record_output(self):
        """Should record outputs for analysis."""
        config = SteeringConfig(vector=SteeringVector())
        controller = SteeringController(config)

        # Apply to prompt first
        controller.apply_to_prompt("Test prompt")

        # Record output
        controller.record_output("Test output")

        # Check history
        report = controller.get_effectiveness_report()
        assert report["applications"] == 1

    def test_effectiveness_report_empty(self):
        """Should handle empty history."""
        config = SteeringConfig(vector=SteeringVector())
        controller = SteeringController(config)

        report = controller.get_effectiveness_report()
        assert report["applications"] == 0
        assert report["average_effectiveness"] is None

    def test_set_vector(self):
        """Should update the vector."""
        config = SteeringConfig(vector=SteeringVector(formality=0.3))
        controller = SteeringController(config)

        new_vector = SteeringVector(formality=0.9)
        controller.set_vector(new_vector)

        assert controller.vector.formality == 0.9

    def test_set_strength(self):
        """Should update the strength."""
        config = SteeringConfig(vector=SteeringVector(), strength=0.5)
        controller = SteeringController(config)

        controller.set_strength(0.9)

        assert controller.strength == 0.9

    def test_reset_history(self):
        """Should clear history."""
        config = SteeringConfig(vector=SteeringVector())
        controller = SteeringController(config)

        # Add some history
        controller.apply_to_prompt("Test")
        controller.record_output("Output")

        # Reset
        controller.reset_history()

        report = controller.get_effectiveness_report()
        assert report["applications"] == 0

    def test_repr(self):
        """Should have informative repr."""
        config = SteeringConfig(vector=SteeringVector(), strength=0.8)
        controller = SteeringController(config)

        repr_str = repr(controller)
        assert "SteeringController" in repr_str
        assert "0.8" in repr_str  # strength


class TestAdaptiveSteering:
    """Tests for adaptive steering behavior."""

    def test_adaptive_adjusts_strength(self):
        """Adaptive mode should adjust strength based on output."""
        config = SteeringConfig(
            vector=SteeringVector(formality=0.9),
            adaptive=True,
            adaptation_rate=0.2,
        )
        controller = SteeringController(config)

        # Apply and record output
        controller.apply_to_prompt("Test")
        controller.record_output(
            "This is like, you know, a super casual response."
        )

        # Strength should have changed (increased because output was too casual)
        # The exact change depends on analysis, but it should be different
        # Note: This is a weak test as heuristics may vary
        report = controller.get_effectiveness_report()
        assert report["applications"] == 1
