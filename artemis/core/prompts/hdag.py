"""
ARTEMIS H-L-DAG Prompt Templates

Hierarchical prompt templates for generating arguments at different levels
of the H-L-DAG (Hierarchical Argument Generation) framework.

Uses centralized prompts from artemis.prompts for consistency and versioning.
"""

from artemis.core.types import ArgumentLevel, DebateContext
from artemis.prompts import get_prompt


def _get_level_instructions(level: ArgumentLevel) -> str:
    """Get level-specific instructions from centralized prompts."""
    level_map = {
        ArgumentLevel.STRATEGIC: "hdag.strategic_instructions",
        ArgumentLevel.TACTICAL: "hdag.tactical_instructions",
        ArgumentLevel.OPERATIONAL: "hdag.operational_instructions",
    }
    return get_prompt(level_map[level])


# =============================================================================
# System Prompts
# =============================================================================


def build_system_prompt(
    agent_name: str,
    role: str,
    level: ArgumentLevel,
    persona: str | None = None,
) -> str:
    """
    Build the system prompt for argument generation.

    Args:
        agent_name: Name of the debate agent.
        role: The agent's role in the debate.
        level: The hierarchical level of argument to generate.
        persona: Optional persona description.

    Returns:
        Complete system prompt string.
    """
    persona_section = f"\nPersona: {persona}" if persona else ""
    level_instructions = _get_level_instructions(level)

    template = get_prompt("hdag.system_prompt_template")
    return template.format(
        agent_name=agent_name,
        role=role,
        persona_section=persona_section,
        level_instructions=level_instructions,
    )


def build_context_prompt(context: DebateContext) -> str:
    """
    Build the context section of the prompt.

    Args:
        context: Current debate context.

    Returns:
        Formatted context string.
    """
    lines = [
        "## Debate Context",
        "",
        f"**Topic**: {context.topic}",
        f"**Round**: {context.current_round + 1} of {context.total_rounds}",
        f"**Turn**: {context.turn_in_round + 1}",
        "",
    ]

    # Add positions if available
    if context.agent_positions:
        lines.append("**Positions**:")
        for agent, position in context.agent_positions.items():
            lines.append(f"- {agent}: {position}")
        lines.append("")

    # Add relevant transcript history
    if context.transcript:
        lines.append("## Recent Arguments")
        lines.append("")

        # Show last few turns
        recent_turns = context.transcript[-3:]
        for turn in recent_turns:
            lines.append(f"**{turn.agent}** ({turn.argument.level.value}):")
            # Truncate long arguments
            content = turn.argument.content
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(content)
            lines.append("")

    return "\n".join(lines)


def build_generation_prompt(
    context: DebateContext,
    agent_name: str,
    role: str,
    level: ArgumentLevel,
    persona: str | None = None,
    additional_instructions: str | None = None,
) -> tuple[str, str]:
    """
    Build complete system and user prompts for argument generation.

    Args:
        context: Current debate context.
        agent_name: Name of the debate agent.
        role: The agent's role in the debate.
        level: The hierarchical level of argument to generate.
        persona: Optional persona description.
        additional_instructions: Optional extra instructions.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = build_system_prompt(agent_name, role, level, persona)

    user_parts = [build_context_prompt(context)]

    if additional_instructions:
        user_parts.append(f"\n## Additional Instructions\n\n{additional_instructions}")

    user_parts.append(
        f"\n## Your Task\n\n"
        f"Generate a {level.value}-level argument for your position. "
        f"Be persuasive, well-reasoned, and intellectually rigorous."
    )

    user_prompt = "\n".join(user_parts)

    return system_prompt, user_prompt


# =============================================================================
# Special Prompts - accessed via get_prompt()
# =============================================================================


def _get_opening_prompt() -> str:
    """Get opening statement prompt from centralized prompts."""
    return get_prompt("hdag.opening_statement")


def _get_closing_prompt() -> str:
    """Get closing statement prompt from centralized prompts."""
    return get_prompt("hdag.closing_statement")


def _get_rebuttal_prompt() -> str:
    """Get rebuttal prompt from centralized prompts."""
    return get_prompt("hdag.rebuttal")


def build_opening_prompt(
    context: DebateContext,
    agent_name: str,
    role: str,
    persona: str | None = None,
) -> tuple[str, str]:
    """Build prompts for opening statement."""
    system_prompt = build_system_prompt(agent_name, role, ArgumentLevel.STRATEGIC, persona)

    user_prompt = (
        f"## Debate Topic\n\n{context.topic}\n\n"
        f"{_get_opening_prompt()}\n\n"
        f"Generate your opening statement."
    )

    return system_prompt, user_prompt


def build_closing_prompt(
    context: DebateContext,
    agent_name: str,
    role: str,
    persona: str | None = None,
) -> tuple[str, str]:
    """Build prompts for closing statement."""
    system_prompt = build_system_prompt(agent_name, role, ArgumentLevel.STRATEGIC, persona)

    # Include full transcript summary for closing
    transcript_summary = build_context_prompt(context)

    user_prompt = (
        f"{transcript_summary}\n\n{_get_closing_prompt()}\n\nGenerate your closing statement."
    )

    return system_prompt, user_prompt


def build_rebuttal_prompt(
    context: DebateContext,  # noqa: ARG001 - reserved for future context use
    agent_name: str,
    role: str,
    target_argument: str,
    level: ArgumentLevel = ArgumentLevel.TACTICAL,
    persona: str | None = None,
) -> tuple[str, str]:
    """Build prompts for rebutting a specific argument."""
    system_prompt = build_system_prompt(agent_name, role, level, persona)

    user_prompt = (
        f"## Argument to Rebut\n\n{target_argument}\n\n"
        f"{_get_rebuttal_prompt()}\n\n"
        f"Generate your rebuttal at the {level.value} level."
    )

    return system_prompt, user_prompt
