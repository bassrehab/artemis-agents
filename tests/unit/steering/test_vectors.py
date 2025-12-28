"""Tests for steering vectors."""

import pytest

from artemis.steering.vectors import SteeringConfig, SteeringMode, SteeringVector


class TestSteeringVector:
    """Tests for SteeringVector."""

    def test_create_default_vector(self):
        """Should create vector with default values."""
        vector = SteeringVector()
        assert vector.formality == 0.5
        assert vector.aggression == 0.3
        assert vector.evidence_emphasis == 0.7
        assert vector.conciseness == 0.5
        assert vector.emotional_appeal == 0.3
        assert vector.confidence == 0.5
        assert vector.creativity == 0.5

    def test_create_custom_vector(self):
        """Should create vector with custom values."""
        vector = SteeringVector(
            formality=0.9,
            aggression=0.1,
            evidence_emphasis=0.8,
        )
        assert vector.formality == 0.9
        assert vector.aggression == 0.1
        assert vector.evidence_emphasis == 0.8

    def test_validate_range(self):
        """Should reject values outside 0-1 range."""
        with pytest.raises(ValueError):
            SteeringVector(formality=1.5)

        with pytest.raises(ValueError):
            SteeringVector(aggression=-0.1)

    def test_to_dict(self):
        """Should convert to dictionary."""
        vector = SteeringVector(formality=0.8, confidence=0.9)
        d = vector.to_dict()
        assert d["formality"] == 0.8
        assert d["confidence"] == 0.9
        assert "aggression" in d

    def test_from_dict(self):
        """Should create from dictionary."""
        d = {"formality": 0.8, "confidence": 0.9}
        vector = SteeringVector.from_dict(d)
        assert vector.formality == 0.8
        assert vector.confidence == 0.9

    def test_blend(self):
        """Should blend two vectors."""
        v1 = SteeringVector(formality=0.2, confidence=0.8)
        v2 = SteeringVector(formality=0.8, confidence=0.2)

        # 50/50 blend
        blended = v1.blend(v2, 0.5)
        assert blended.formality == pytest.approx(0.5, abs=0.01)
        assert blended.confidence == pytest.approx(0.5, abs=0.01)

        # Full weight to v2
        blended = v1.blend(v2, 1.0)
        assert blended.formality == pytest.approx(0.8, abs=0.01)

        # No weight to v2 (stay at v1)
        blended = v1.blend(v2, 0.0)
        assert blended.formality == pytest.approx(0.2, abs=0.01)

    def test_blend_invalid_weight(self):
        """Should reject invalid blend weights."""
        v1 = SteeringVector()
        v2 = SteeringVector()

        with pytest.raises(ValueError):
            v1.blend(v2, 1.5)

        with pytest.raises(ValueError):
            v1.blend(v2, -0.1)

    def test_distance(self):
        """Should calculate distance between vectors."""
        v1 = SteeringVector(formality=0.0, confidence=0.0)
        v2 = SteeringVector(formality=1.0, confidence=0.0)

        # Distance should be non-zero
        dist = v1.distance(v2)
        assert dist > 0

        # Same vector should have zero distance
        dist = v1.distance(v1)
        assert dist == pytest.approx(0.0, abs=0.001)

    def test_magnitude(self):
        """Should calculate magnitude (distance from neutral)."""
        neutral = SteeringVector()
        extreme = SteeringVector(
            formality=1.0,
            aggression=1.0,
            evidence_emphasis=1.0,
            conciseness=1.0,
            emotional_appeal=1.0,
            confidence=1.0,
            creativity=1.0,
        )

        # Extreme should have higher magnitude
        assert extreme.magnitude() > neutral.magnitude()

    def test_repr(self):
        """Should have informative repr."""
        vector = SteeringVector(formality=0.9)
        repr_str = repr(vector)
        assert "SteeringVector" in repr_str
        assert "formality" in repr_str


class TestSteeringConfig:
    """Tests for SteeringConfig."""

    def test_create_config(self):
        """Should create config with defaults."""
        vector = SteeringVector()
        config = SteeringConfig(vector=vector)
        assert config.vector == vector
        assert config.mode == SteeringMode.PROMPT
        assert config.strength == 1.0
        assert config.adaptive is False

    def test_create_config_with_options(self):
        """Should create config with custom options."""
        vector = SteeringVector()
        config = SteeringConfig(
            vector=vector,
            mode=SteeringMode.BOTH,
            strength=0.8,
            adaptive=True,
            adaptation_rate=0.2,
        )
        assert config.mode == SteeringMode.BOTH
        assert config.strength == 0.8
        assert config.adaptive is True
        assert config.adaptation_rate == 0.2

    def test_validate_strength(self):
        """Should reject invalid strength."""
        vector = SteeringVector()
        with pytest.raises(ValueError):
            SteeringConfig(vector=vector, strength=1.5)

    def test_with_strength(self):
        """Should create copy with different strength."""
        config = SteeringConfig(vector=SteeringVector(), strength=0.5)
        new_config = config.with_strength(0.8)
        assert new_config.strength == 0.8
        assert config.strength == 0.5  # Original unchanged

    def test_with_vector(self):
        """Should create copy with different vector."""
        v1 = SteeringVector(formality=0.3)
        v2 = SteeringVector(formality=0.9)
        config = SteeringConfig(vector=v1)
        new_config = config.with_vector(v2)
        assert new_config.vector.formality == 0.9
        assert config.vector.formality == 0.3  # Original unchanged


class TestSteeringMode:
    """Tests for SteeringMode."""

    def test_modes_exist(self):
        """All modes should exist."""
        assert SteeringMode.PROMPT
        assert SteeringMode.OUTPUT
        assert SteeringMode.BOTH

    def test_modes_are_strings(self):
        """Modes should have string values."""
        assert SteeringMode.PROMPT.value == "prompt"
        assert SteeringMode.OUTPUT.value == "output"
        assert SteeringMode.BOTH.value == "both"
