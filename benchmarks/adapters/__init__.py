"""Framework adapters for benchmark comparisons."""

from benchmarks.adapters.base import DebateAdapter, DebateResult
from benchmarks.adapters.artemis_adapter import ArtemisAdapter
from benchmarks.adapters.autogen_adapter import AutoGenAdapter
from benchmarks.adapters.crewai_adapter import CrewAIAdapter
from benchmarks.adapters.camel_adapter import CAMELAdapter

__all__ = [
    "DebateAdapter",
    "DebateResult",
    "ArtemisAdapter",
    "AutoGenAdapter",
    "CrewAIAdapter",
    "CAMELAdapter",
]
