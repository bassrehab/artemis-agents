"""
ARTEMIS H-L-DAG Prompt Templates

Hierarchical prompt templates for generating arguments at different levels
of the H-L-DAG (Hierarchical Argument Generation) framework.
"""

from artemis.core.types import ArgumentLevel, DebateContext

# =============================================================================
# Level-Specific Instructions
# =============================================================================

STRATEGIC_INSTRUCTIONS = """
## Strategic Level Argument

You are generating a STRATEGIC-level argument. This is the highest level
in the argument hierarchy, focused on overarching positions and frameworks.

Your argument should:
1. **State your main thesis clearly** - What is your core position?
2. **Outline key pillars** - What are the 2-3 main supporting themes?
3. **Establish evaluation framework** - How should this issue be judged?
4. **Consider long-term implications** - What are the broader consequences?
5. **Address ethical dimensions** - What values are at stake?

Structure your response as a cohesive argument that can be supported by
more detailed tactical and operational arguments later.
"""

TACTICAL_INSTRUCTIONS = """
## Tactical Level Argument

You are generating a TACTICAL-level argument. This is the middle level
in the argument hierarchy, focused on supporting evidence and reasoning.

Your argument should:
1. **Provide specific evidence** for the strategic position
2. **Draw causal connections** - If X, then Y, because Z
3. **Address likely counterarguments** - Anticipate and refute objections
4. **Cite sources** where applicable - Reference studies, experts, data
5. **Build logical chains** - Connect your points in a coherent flow

Structure your response to directly support the strategic-level thesis
while providing enough detail to be convincing.
"""

OPERATIONAL_INSTRUCTIONS = """
## Operational Level Argument

You are generating an OPERATIONAL-level argument. This is the most
concrete level in the argument hierarchy, focused on specific details.

Your argument should:
1. **Cite specific facts** - Exact statistics, quotes, dates
2. **Reference concrete examples** - Real-world cases and instances
3. **Provide case studies** - Detailed analysis of specific situations
4. **Detail implementation** - How would this work in practice?
5. **Ground abstractions** - Make theoretical points tangible

Structure your response with precise, verifiable details that substantiate
the tactical-level arguments above.
"""

LEVEL_INSTRUCTIONS = {
    ArgumentLevel.STRATEGIC: STRATEGIC_INSTRUCTIONS,
    ArgumentLevel.TACTICAL: TACTICAL_INSTRUCTIONS,
    ArgumentLevel.OPERATIONAL: OPERATIONAL_INSTRUCTIONS,
}


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

    return f"""You are {agent_name}, a participant in a structured debate.

Role: {role}{persona_section}

{LEVEL_INSTRUCTIONS[level]}

## Argument Quality Requirements

Your argument must demonstrate:
- **Logical Coherence**: Clear reasoning without fallacies
- **Evidence Quality**: Well-sourced, relevant support
- **Causal Reasoning**: Clear cause-effect relationships
- **Ethical Awareness**: Consideration of moral implications
- **Intellectual Honesty**: Acknowledgment of limitations

## Format Guidelines

Structure your argument clearly with:
- A clear thesis statement at the beginning
- Well-organized supporting points
- Explicit causal reasoning (use phrases like "because", "therefore", "this leads to")
- Evidence citations where applicable [SOURCE: description]
- Acknowledgment of potential counterarguments
"""


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
# Special Prompts
# =============================================================================

OPENING_STATEMENT_PROMPT = """
## Opening Statement

This is your opening statement in the debate. You should:
1. Clearly state your position on the topic
2. Preview your main arguments (2-3 key points)
3. Establish the framework for how this issue should be evaluated
4. Set the tone for a constructive, rigorous debate

Be confident but measured. This is your first impression.
"""

CLOSING_STATEMENT_PROMPT = """
## Closing Statement

This is your closing statement in the debate. You should:
1. Summarize your strongest arguments
2. Address the key points raised by opponents
3. Explain why your position should prevail
4. End with a compelling conclusion

Be persuasive and memorable. This is your final opportunity to convince.
"""

REBUTTAL_PROMPT = """
## Rebuttal

You are responding to an opponent's argument. You should:
1. Identify the specific claim you are addressing
2. Explain why it is flawed, incomplete, or incorrect
3. Provide counter-evidence or alternative reasoning
4. Strengthen your own position in the process

Be respectful but incisive. Address the argument, not the arguer.
"""


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
        f"{OPENING_STATEMENT_PROMPT}\n\n"
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
        f"{transcript_summary}\n\n{CLOSING_STATEMENT_PROMPT}\n\nGenerate your closing statement."
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
        f"{REBUTTAL_PROMPT}\n\n"
        f"Generate your rebuttal at the {level.value} level."
    )

    return system_prompt, user_prompt
