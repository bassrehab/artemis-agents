"""Pre-configured steering vector presets.

Common steering configurations for typical debate scenarios.
"""

from artemis.steering.vectors import SteeringConfig, SteeringMode, SteeringVector


# Preset vectors for common styles
PRESET_VECTORS = {
    "formal_academic": SteeringVector(
        formality=0.9,
        aggression=0.2,
        evidence_emphasis=0.9,
        conciseness=0.4,
        emotional_appeal=0.1,
        confidence=0.7,
        creativity=0.3,
    ),
    "aggressive_debater": SteeringVector(
        formality=0.5,
        aggression=0.8,
        evidence_emphasis=0.6,
        conciseness=0.7,
        emotional_appeal=0.4,
        confidence=0.9,
        creativity=0.5,
    ),
    "diplomatic": SteeringVector(
        formality=0.7,
        aggression=0.1,
        evidence_emphasis=0.5,
        conciseness=0.5,
        emotional_appeal=0.4,
        confidence=0.5,
        creativity=0.4,
    ),
    "persuasive_speaker": SteeringVector(
        formality=0.5,
        aggression=0.4,
        evidence_emphasis=0.6,
        conciseness=0.5,
        emotional_appeal=0.7,
        confidence=0.8,
        creativity=0.6,
    ),
    "analytical": SteeringVector(
        formality=0.7,
        aggression=0.2,
        evidence_emphasis=0.9,
        conciseness=0.6,
        emotional_appeal=0.1,
        confidence=0.6,
        creativity=0.4,
    ),
    "creative_thinker": SteeringVector(
        formality=0.4,
        aggression=0.3,
        evidence_emphasis=0.4,
        conciseness=0.4,
        emotional_appeal=0.5,
        confidence=0.7,
        creativity=0.9,
    ),
    "concise_communicator": SteeringVector(
        formality=0.5,
        aggression=0.3,
        evidence_emphasis=0.5,
        conciseness=0.9,
        emotional_appeal=0.2,
        confidence=0.7,
        creativity=0.4,
    ),
    "empathetic_advocate": SteeringVector(
        formality=0.4,
        aggression=0.1,
        evidence_emphasis=0.4,
        conciseness=0.4,
        emotional_appeal=0.8,
        confidence=0.6,
        creativity=0.5,
    ),
    "balanced": SteeringVector(
        formality=0.5,
        aggression=0.3,
        evidence_emphasis=0.5,
        conciseness=0.5,
        emotional_appeal=0.3,
        confidence=0.5,
        creativity=0.5,
    ),
    "devil_advocate": SteeringVector(
        formality=0.6,
        aggression=0.6,
        evidence_emphasis=0.7,
        conciseness=0.6,
        emotional_appeal=0.3,
        confidence=0.8,
        creativity=0.7,
    ),
}


def get_preset(name: str) -> SteeringVector:
    """Get a preset steering vector by name.

    Args:
        name: Name of the preset.

    Returns:
        The preset SteeringVector.

    Raises:
        KeyError: If preset name not found.
    """
    if name not in PRESET_VECTORS:
        available = ", ".join(PRESET_VECTORS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")

    return PRESET_VECTORS[name]


def get_preset_config(
    name: str,
    mode: SteeringMode = SteeringMode.PROMPT,
    strength: float = 1.0,
    adaptive: bool = False,
) -> SteeringConfig:
    """Get a preset as a full SteeringConfig.

    Args:
        name: Name of the preset.
        mode: Steering mode to use.
        strength: Application strength.
        adaptive: Whether to enable adaptation.

    Returns:
        SteeringConfig with the preset vector.
    """
    return SteeringConfig(
        vector=get_preset(name),
        mode=mode,
        strength=strength,
        adaptive=adaptive,
    )


def list_presets() -> list[str]:
    """List available preset names.

    Returns:
        List of preset names.
    """
    return list(PRESET_VECTORS.keys())


def describe_preset(name: str) -> str:
    """Get a human-readable description of a preset.

    Args:
        name: Name of the preset.

    Returns:
        Description string.
    """
    vector = get_preset(name)

    descriptions = {
        "formal_academic": "Formal, evidence-heavy academic style",
        "aggressive_debater": "Direct, confrontational debate style",
        "diplomatic": "Cooperative, bridge-building style",
        "persuasive_speaker": "Emotionally engaging, confident style",
        "analytical": "Logic-focused, evidence-based analysis",
        "creative_thinker": "Unconventional, innovative approaches",
        "concise_communicator": "Brief, to-the-point communication",
        "empathetic_advocate": "Emotionally resonant, understanding style",
        "balanced": "Neutral, balanced approach",
        "devil_advocate": "Critical, challenging perspective",
    }

    desc = descriptions.get(name, f"Custom preset: {name}")

    # Add key dimension highlights
    highlights = []
    if vector.formality > 0.7:
        highlights.append("formal")
    elif vector.formality < 0.3:
        highlights.append("casual")

    if vector.aggression > 0.6:
        highlights.append("assertive")
    elif vector.aggression < 0.2:
        highlights.append("cooperative")

    if vector.evidence_emphasis > 0.7:
        highlights.append("data-driven")

    if vector.emotional_appeal > 0.6:
        highlights.append("emotionally engaging")

    if vector.confidence > 0.7:
        highlights.append("confident")

    highlight_str = ", ".join(highlights) if highlights else "balanced"

    return f"{desc} ({highlight_str})"
