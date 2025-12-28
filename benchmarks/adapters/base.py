"""Base adapter interface for framework benchmarks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DebateResult:
    """Standardized debate result across frameworks."""

    framework: str
    topic: str
    transcript: list[dict[str, str]]  # [{"agent": "pro", "content": "..."}, ...]
    verdict: str | None = None  # "pro", "con", or None
    verdict_reasoning: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    time_seconds: float = 0.0
    error: str | None = None

    @property
    def full_transcript(self) -> str:
        """Get transcript as single string for evaluation."""
        lines = []
        for turn in self.transcript:
            agent = turn.get("agent", "unknown")
            content = turn.get("content", "")
            lines.append(f"[{agent.upper()}]: {content}")

        if self.verdict:
            lines.append(f"\n[VERDICT]: {self.verdict}")
            if self.verdict_reasoning:
                lines.append(f"[REASONING]: {self.verdict_reasoning}")

        return "\n\n".join(lines)


class DebateAdapter(ABC):
    """Abstract base class for framework debate adapters."""

    name: str = "base"

    def __init__(self, model: str = "gpt-4o", rounds: int = 3):
        self.model = model
        self.rounds = rounds

    @abstractmethod
    async def run_debate(
        self,
        topic: str,
        pro_position: str,
        con_position: str,
    ) -> DebateResult:
        """
        Run a debate on the given topic.

        Args:
            topic: The debate topic/question.
            pro_position: Position for the pro side.
            con_position: Position for the con side.

        Returns:
            Standardized DebateResult.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the framework is installed and available."""
        pass

    def get_config(self) -> dict[str, Any]:
        """Get adapter configuration."""
        return {
            "framework": self.name,
            "model": self.model,
            "rounds": self.rounds,
        }
