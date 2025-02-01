"""
ARTEMIS Agents - Adaptive Reasoning Through Evaluation of Multi-agent Intelligent Systems

A production-ready framework for structured multi-agent debates with adaptive evaluation,
causal reasoning, and built-in safety monitoring.

Example:
    >>> from artemis import Debate, Agent, JuryPanel
    >>> 
    >>> agents = [
    ...     Agent(name="Proponent", role="Argues in favor", model="gpt-4o"),
    ...     Agent(name="Opponent", role="Argues against", model="gpt-4o"),
    ... ]
    >>> 
    >>> debate = Debate(
    ...     topic="Should AI systems be given legal personhood?",
    ...     agents=agents,
    ...     jury=JuryPanel(evaluators=3),
    ...     rounds=3
    ... )
    >>> 
    >>> result = await debate.run()
    >>> print(f"Verdict: {result.verdict.decision}")

For more information, see:
- Documentation: https://github.com/bassrehab/artemis-agents
- Paper: https://www.tdcommons.org/dpubs_series/7729/
"""

__version__ = "1.0.0"
__author__ = "Subhadip Mitra"
__email__ = "contact@subhadipmitra.com"

# Core exports (will be populated as modules are implemented)
# from artemis.core.debate import Debate
# from artemis.core.agent import Agent
# from artemis.core.jury import JuryPanel
# from artemis.core.types import Argument, ArgumentLevel, Turn, Verdict

__all__ = [
    "__version__",
    # "Debate",
    # "Agent", 
    # "JuryPanel",
    # "Argument",
    # "ArgumentLevel",
    # "Turn",
    # "Verdict",
]
