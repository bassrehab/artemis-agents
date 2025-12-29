"""
Extraction Prompts - Version 1

Prompts for LLM-based extraction of evidence and causal relationships from arguments.
"""

# =============================================================================
# Causal Extraction
# =============================================================================

CAUSAL_EXTRACTION = """Analyze the following argument and extract all causal relationships.

For each causal relationship, identify:
1. The cause (what triggers the effect)
2. The effect (what results from the cause)
3. The mechanism (how/why the cause leads to the effect)
4. The strength (strong/moderate/weak based on evidence provided)

Argument:
{content}

Respond with a JSON array of causal links:
[
    {{
        "cause": "<description of cause>",
        "effect": "<description of effect>",
        "mechanism": "<how cause leads to effect>",
        "strength": "<strong|moderate|weak>",
        "quote": "<relevant quote from argument if available>"
    }}
]

If no causal relationships are found, return an empty array: []
Only extract explicit or strongly implied causal claims, not speculative ones."""

# =============================================================================
# Evidence Extraction
# =============================================================================

EVIDENCE_EXTRACTION = """Analyze the following argument and extract all evidence cited.

For each piece of evidence, identify:
1. The type (fact, statistic, quote, example, study, expert_opinion)
2. The content (the actual evidence)
3. The source (where it comes from, if mentioned)
4. The claim it supports (what point it backs up)

Argument:
{content}

Respond with a JSON array of evidence:
[
    {{
        "type": "<fact|statistic|quote|example|study|expert_opinion>",
        "content": "<the evidence itself>",
        "source": "<source if mentioned, otherwise null>",
        "supports_claim": "<the claim this evidence supports>"
    }}
]

If no evidence is found, return an empty array: []
Only extract actual evidence, not assertions or opinions without backing."""

# =============================================================================
# Evidence Types
# =============================================================================

EVIDENCE_TYPES = [
    "fact",  # Verifiable factual statement
    "statistic",  # Numerical data or percentage
    "quote",  # Direct quote from a source
    "example",  # Specific real-world example
    "study",  # Reference to research or study
    "expert_opinion",  # Opinion from a recognized expert
]

# =============================================================================
# Strength Definitions
# =============================================================================

CAUSAL_STRENGTH_DEFINITIONS = {
    "strong": "Direct causal link with clear mechanism and evidence",
    "moderate": "Plausible causal link with some supporting reasoning",
    "weak": "Implied or speculative causal relationship",
}
