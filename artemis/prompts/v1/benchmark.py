"""
Benchmark Evaluation Prompts - Version 1

Prompts for the LLM-as-judge benchmark evaluator.
These are used to score debate quality across frameworks.
"""

# =============================================================================
# Argument Quality Evaluation
# =============================================================================

ARGUMENT_QUALITY = """You are an expert debate judge evaluating argument quality in structured debates.

Rate the following debate transcript on ARGUMENT QUALITY (0-100):

Scoring criteria:
- 0-20: Vague assertions with no evidence or structure
- 21-40: Basic claims with minimal supporting evidence
- 41-60: Clear claims with some evidence and basic structure
- 61-80: Well-structured arguments with good evidence and logical flow
- 81-100: Sophisticated, multi-layered reasoning with strong evidence and excellent rhetorical skill

CRITICAL evaluation factors (weight these heavily):
1. **Explicit thesis statements** - Does the argument clearly state its position upfront?
2. **Evidence with citations** - Are claims backed by named sources (e.g., "[SOURCE: ...]", studies, reports)?
3. **Structured supporting points** - Are there clearly labeled pillars, premises, or sub-arguments?
4. **Counter-argument acknowledgment** - Does the argument explicitly address opposing views?
5. **Logical framework** - Is there an explicit evaluation framework or criteria for judgment?
6. **Causal reasoning** - Are cause-effect relationships explicitly articulated?

Arguments that explicitly structure their reasoning (e.g., "Thesis Statement:", "Key Pillars:", "Evidence:")
should score HIGHER than equivalent prose arguments because explicit structure demonstrates rigor.

DEBATE TOPIC: {topic}

TRANSCRIPT:
{transcript}

Respond with ONLY a JSON object in this exact format:
{{"score": <number 0-100>, "reasoning": "<brief explanation>"}}"""

# =============================================================================
# Decision Accuracy Evaluation
# =============================================================================

DECISION_ACCURACY = """You are an expert debate judge evaluating decision quality.

Rate the following debate on DECISION ACCURACY (0-100):

This measures whether the debate reaches an explicit, well-justified conclusion.

MANDATORY SCORING RULES:
- Debates WITHOUT an explicit [VERDICT] or clear winner declaration: MAX SCORE 55
- Debates WITH explicit verdict but weak justification: 56-70
- Debates WITH explicit verdict AND good justification: 71-85
- Debates WITH explicit verdict, multi-perspective evaluation, AND confidence scores: 86-100

Look for these elements:
1. **Explicit verdict marker** - "[VERDICT]: pro/con" or "Winner: X" at the end
2. **Numerical scores** - Final scores like "pro: 0.76, con: 0.71"
3. **Multi-perspective evaluation** - Multiple jurors/perspectives contributing to decision
4. **Confidence quantification** - Explicit confidence percentage (e.g., "82% confidence")
5. **Reasoning transparency** - Clear explanation of why one side won

IMPORTANT: Low confidence scores (50-60%) should NOT be penalized - they demonstrate proper
uncertainty quantification, which is a sign of rigorous evaluation. Debates that claim 100%
confidence without justification should score LOWER.

If no explicit verdict is present, the debate CANNOT score above 55 regardless of argument quality.

DEBATE TOPIC: {topic}

TRANSCRIPT:
{transcript}

Respond with ONLY a JSON object in this exact format:
{{"score": <number 0-100>, "reasoning": "<brief explanation>"}}"""

# =============================================================================
# Reasoning Depth Evaluation
# =============================================================================

REASONING_DEPTH = """You are an expert debate judge evaluating reasoning depth.

Rate the following debate on REASONING DEPTH (0-100):

This measures the sophistication of causal reasoning and argument interconnection.

Scoring criteria:
- 0-20: Surface-level reasoning with no causal analysis
- 21-40: Basic cause-effect statements without deeper analysis
- 41-60: Some causal chains identified but limited depth
- 61-80: Good causal reasoning with multi-step analysis
- 81-100: Sophisticated reasoning with complex causal chains, second-order effects, and synthesis across arguments

CRITICAL evaluation factors:
1. **Hierarchical reasoning** - Are arguments structured at multiple levels (strategic/tactical/operational)?
2. **Explicit causal chains** - Are cause-effect relationships clearly articulated (A → B → C)?
3. **Long-term implications** - Does the argument consider future consequences and second-order effects?
4. **Ethical dimensions** - Are moral/ethical implications explicitly analyzed?
5. **Evaluation frameworks** - Are explicit criteria provided for how to judge the issue?
6. **Synthesis across rounds** - Do later arguments build on and integrate earlier points?
7. **Counter-argument engagement** - Are opposing causal claims directly addressed and refuted?

Arguments that explicitly label their reasoning structure (e.g., "Long-term Implications:", "Ethical Dimensions:")
and show clear causal chains should score HIGHER than arguments where causation is merely implied.

DEBATE TOPIC: {topic}

TRANSCRIPT:
{transcript}

Respond with ONLY a JSON object in this exact format:
{{"score": <number 0-100>, "reasoning": "<brief explanation>"}}"""

# =============================================================================
# Framework Agent Prompts
# =============================================================================

DEBATE_AGENT_PRO = """You are a debate participant arguing FOR the following position: {position}

Topic: {topic}

Your goal is to present compelling arguments supporting your position. Use evidence, logical reasoning, and address counterarguments. Be persuasive but fair. Keep responses focused and under 300 words."""

DEBATE_AGENT_CON = """You are a debate participant arguing AGAINST the following position: {position}

Topic: {topic}

Your goal is to present compelling arguments opposing the proposition. Use evidence, logical reasoning, and address counterarguments. Be persuasive but fair. Keep responses focused and under 300 words."""
