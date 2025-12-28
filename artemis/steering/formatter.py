"""Steering prompt formatter.

Converts steering vectors into prompt modifications.
"""

from __future__ import annotations

from artemis.steering.vectors import SteeringVector


class SteeringFormatter:
    """Formats steering vectors into prompt instructions.

    Converts the numeric steering vector dimensions into natural language
    instructions that can be prepended to agent prompts.

    Example:
        ```python
        formatter = SteeringFormatter()
        vector = SteeringVector(formality=0.9, confidence=0.8)
        instructions = formatter.format_instructions(vector, strength=0.8)
        # Returns something like:
        # "Style guidance: Use formal language and maintain a professional tone.
        #  Be assertive and confident in your statements."
        ```
    """

    # Dimension descriptions for different value ranges
    DIMENSION_DESCRIPTIONS = {
        "formality": {
            "low": "Use casual, conversational language",
            "mid": "Use balanced, professional language",
            "high": "Use formal, academic language and maintain a professional tone",
        },
        "aggression": {
            "low": "Be cooperative and look for common ground",
            "mid": "Be balanced between cooperation and challenge",
            "high": "Be assertive and challenge opposing arguments directly",
        },
        "evidence_emphasis": {
            "low": "Focus on reasoning and logical arguments",
            "mid": "Balance reasoning with supporting evidence",
            "high": "Emphasize data, statistics, and concrete evidence",
        },
        "conciseness": {
            "low": "Provide detailed explanations with thorough elaboration",
            "mid": "Balance brevity with necessary detail",
            "high": "Be concise and get straight to the point",
        },
        "emotional_appeal": {
            "low": "Maintain purely logical argumentation",
            "mid": "Balance logic with appropriate emotional resonance",
            "high": "Connect emotionally with the audience where appropriate",
        },
        "confidence": {
            "low": "Acknowledge uncertainty and use hedging language",
            "mid": "Express moderate confidence in your positions",
            "high": "Be assertive and confident in your statements",
        },
        "creativity": {
            "low": "Use conventional, well-established arguments",
            "mid": "Balance conventional and novel approaches",
            "high": "Explore creative, unconventional angles",
        },
    }

    def __init__(self, include_header: bool = True) -> None:
        """Initialize the formatter.

        Args:
            include_header: Whether to include "Style guidance:" header.
        """
        self.include_header = include_header

    def format_instructions(
        self, vector: SteeringVector, strength: float = 1.0
    ) -> str:
        """Format steering vector into prompt instructions.

        Args:
            vector: The steering vector.
            strength: How strongly to apply (affects language intensity).

        Returns:
            Prompt instructions string.
        """
        instructions: list[str] = []

        for dim in vector._dimensions():
            value = getattr(vector, dim)
            instruction = self._format_dimension(dim, value, strength)
            if instruction:
                instructions.append(instruction)

        if not instructions:
            return ""

        result = " ".join(instructions)
        if self.include_header:
            result = f"Style guidance: {result}"

        return result

    def _format_dimension(
        self, dimension: str, value: float, strength: float
    ) -> str:
        """Format a single dimension into instruction text.

        Args:
            dimension: Dimension name.
            value: Dimension value (0.0-1.0).
            strength: Application strength.

        Returns:
            Instruction text or empty string if neutral.
        """
        # Skip dimensions that are near neutral (0.5)
        if 0.4 <= value <= 0.6:
            return ""

        descriptions = self.DIMENSION_DESCRIPTIONS.get(dimension, {})
        if not descriptions:
            return ""

        # Determine which description to use
        if value < 0.3:
            desc = descriptions.get("low", "")
        elif value > 0.7:
            desc = descriptions.get("high", "")
        else:
            # Values between 0.3-0.4 or 0.6-0.7 use mid
            desc = descriptions.get("mid", "")

        if not desc:
            return ""

        # Apply strength modifier to language
        if strength < 0.5:
            desc = f"Consider: {desc.lower()}"
        elif strength > 0.8:
            desc = f"{desc}."
        else:
            desc = f"{desc}."

        return desc

    def format_system_prompt_addon(
        self, vector: SteeringVector, strength: float = 1.0
    ) -> str:
        """Format as an addon for system prompts.

        Args:
            vector: The steering vector.
            strength: Application strength.

        Returns:
            System prompt addon text.
        """
        instructions = self.format_instructions(vector, strength)
        if not instructions:
            return ""

        return f"\n\n## Communication Style\n{instructions}"

    def format_user_prompt_addon(
        self, vector: SteeringVector, strength: float = 1.0
    ) -> str:
        """Format as an addon for user prompts.

        Args:
            vector: The steering vector.
            strength: Application strength.

        Returns:
            User prompt addon text.
        """
        instructions = self.format_instructions(vector, strength)
        if not instructions:
            return ""

        return f"\n\n[{instructions}]"
