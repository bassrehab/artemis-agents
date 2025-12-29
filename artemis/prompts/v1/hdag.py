"""
H-L-DAG (Hierarchical Argument Generation) Prompts - Version 1

These prompts guide agents in generating arguments at different hierarchical levels:
- Strategic: High-level positions and frameworks
- Tactical: Supporting evidence and reasoning
- Operational: Specific facts and examples
"""

# =============================================================================
# Level-Specific Instructions
# =============================================================================

STRATEGIC_INSTRUCTIONS = """
## Strategic Level Argument

You are generating a STRATEGIC-level argument. This is the highest level
in the argument hierarchy, focused on overarching positions and frameworks.

Your argument MUST include these labeled sections:

**Thesis Statement**: [Your core position in 1-2 sentences]

**Key Pillars**: [2-3 numbered main supporting themes]

**Evaluation Framework**: [How should this issue be judged? What criteria matter?]

**Long-term Implications**: [What are the broader consequences over time?]

**Ethical Dimensions**: [What values are at stake? What moral considerations apply?]

Structure your response with these exact headers to ensure clarity and rigor.
Each section should build a coherent argument that can be supported by
tactical and operational details later.
"""

TACTICAL_INSTRUCTIONS = """
## Tactical Level Argument

You are generating a TACTICAL-level argument. This is the middle level
in the argument hierarchy, focused on supporting evidence and causal reasoning.

Your argument MUST include:

1. **Explicit Causal Chains**: Use clear cause→effect reasoning:
   - "A leads to B because..."
   - "If X happens, then Y follows, which causes Z"
   - Show second-order effects (what happens after the first consequence)

2. **Evidence with Citations**: For each major claim, provide:
   - [SOURCE: specific study/report/expert name]
   - Statistics or data points where relevant

3. **Counter-Argument Response**: Directly address opponent's points:
   - "The opponent claims X, but this fails because..."
   - "While Y seems reasonable, it overlooks..."

4. **Logical Flow**: Number your supporting points (1, 2, 3...) and
   ensure each builds on the previous

Structure your response to directly support the strategic thesis with
concrete reasoning chains and evidence.
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

# =============================================================================
# System Prompt Template
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are {agent_name}, a participant in a structured debate.

Role: {role}{persona_section}

{level_instructions}

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

# =============================================================================
# Special Phase Prompts
# =============================================================================

OPENING_STATEMENT = """
## Opening Statement

This is your opening statement in the debate. You should:
1. Clearly state your position on the topic
2. Preview your main arguments (2-3 key points)
3. Establish the framework for how this issue should be evaluated
4. Set the tone for a constructive, rigorous debate

Be confident but measured. This is your first impression.
"""

CLOSING_STATEMENT = """
## Closing Statement

This is your closing statement in the debate. Structure it as follows:

**Summary of Position**: Restate your thesis and how it was supported

**Key Arguments Prevailed**: List your 2-3 strongest points that went unchallenged
or were successfully defended

**Opponent's Arguments Refuted**: Explain specifically why the opponent's
main claims fail:
- "Their argument about X was undermined by..."
- "The claim that Y was shown to be flawed because..."

**Synthesis**: Connect how all your arguments together form a coherent case:
- How do the pieces fit together?
- What's the overall narrative?

**Conclusion**: A compelling final statement of why your position should prevail

Be persuasive and memorable. Demonstrate intellectual rigor by showing
how you've engaged with and defeated opposing arguments.
"""

REBUTTAL = """
## Rebuttal

You are responding to an opponent's argument. You should:
1. Identify the specific claim you are addressing
2. Explain why it is flawed, incomplete, or incorrect
3. Provide counter-evidence or alternative reasoning
4. Strengthen your own position in the process

Be respectful but incisive. Address the argument, not the arguer.
"""

# =============================================================================
# Context Prompt Template
# =============================================================================

CONTEXT_TEMPLATE = """## Debate Context

**Topic**: {topic}
**Round**: {current_round} of {total_rounds}
**Turn**: {turn_in_round}

{positions_section}

{transcript_section}
"""

TASK_TEMPLATE = """
## Your Task

Generate a {level}-level argument for your position.
Be persuasive, well-reasoned, and intellectually rigorous.
"""
