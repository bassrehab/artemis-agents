# LLM-Based Extraction

ARTEMIS uses LLM-based extraction to identify evidence and causal relationships in arguments. This provides more accurate extraction than regex patterns alone.

## Overview

Traditional regex-based extraction struggles with natural language variations. LLM extraction understands context and can identify:

- **Evidence**: Facts, statistics, quotes, examples, studies, expert opinions
- **Causal Links**: Cause-effect relationships with mechanisms and strength

## Extractors

### LLMCausalExtractor

Extracts cause-effect relationships from argument text.

```python
from artemis.core.llm_extraction import LLMCausalExtractor

extractor = LLMCausalExtractor(
    model_name="gpt-4o-mini",  # Model for extraction
    use_cache=True,            # Cache results by content hash
)

links = await extractor.extract(content)
# Returns: list[CausalLink]
```

Each `CausalLink` contains:

| Field | Type | Description |
|-------|------|-------------|
| `cause` | str | What triggers the effect |
| `effect` | str | What results from the cause |
| `mechanism` | str | How the cause leads to the effect |
| `strength` | float | Confidence (0.0-1.0) |

### LLMEvidenceExtractor

Extracts evidence citations from argument text.

```python
from artemis.core.llm_extraction import LLMEvidenceExtractor

extractor = LLMEvidenceExtractor(
    model_name="gpt-4o-mini",
    use_cache=True,
)

evidence = await extractor.extract(content)
# Returns: list[Evidence]
```

Each `Evidence` contains:

| Field | Type | Description |
|-------|------|-------------|
| `type` | str | One of: fact, statistic, quote, example, study, expert_opinion |
| `content` | str | The evidence text |
| `source` | str | Source attribution if mentioned |
| `verified` | bool | Whether evidence has been verified |

## Hybrid Extractors

For cost-effective extraction, use hybrid extractors that try fast regex patterns first and fall back to LLM when needed.

```python
from artemis.core.llm_extraction import (
    HybridCausalExtractor,
    HybridEvidenceExtractor,
)

# Tries regex first, falls back to LLM
causal_extractor = HybridCausalExtractor(
    model_name="gpt-4o-mini",
    use_cache=True,
)

evidence_extractor = HybridEvidenceExtractor(
    model_name="gpt-4o-mini",
    use_cache=True,
)

# Use same as LLM extractors
links = await causal_extractor.extract(content)
evidence = await evidence_extractor.extract(content)
```

### Extraction Strategy

```
Content → Regex Extraction
              │
              ├─ Results found? → Return results
              │
              └─ No results? → LLM Extraction → Return results
```

## Caching

Extraction results are cached by content hash to avoid redundant LLM calls:

```python
# First call: LLM extraction
links1 = await extractor.extract("Climate change causes...")

# Second call with same content: Cache hit (no LLM call)
links2 = await extractor.extract("Climate change causes...")
```

Clear the cache when needed:

```python
from artemis.core.llm_extraction import clear_extraction_cache

clear_extraction_cache()
```

## Integration with Agents

The extraction system is automatically used during argument generation:

```python
from artemis.core.agent import Agent

agent = Agent(
    name="analyst",
    role="Research analyst",
    model="gpt-4o",
)

# Extraction happens automatically during argument generation
argument = await agent.generate_argument(context)

# Access extracted data
print(f"Evidence found: {len(argument.evidence)}")
print(f"Causal links: {len(argument.causal_links)}")
```

## Example Output

Given an argument about climate policy:

```python
content = """
Rising global temperatures cause increased extreme weather events.
According to the IPCC 2023 report, we've seen a 40% increase in
Category 4+ hurricanes since 1980. This leads to higher economic
costs, estimated at $300 billion annually by 2030.
"""

# Causal extraction
links = await causal_extractor.extract(content)
# [
#   CausalLink(
#     cause="Rising global temperatures",
#     effect="increased extreme weather events",
#     strength=0.8
#   ),
#   CausalLink(
#     cause="extreme weather events",
#     effect="higher economic costs",
#     strength=0.7
#   )
# ]

# Evidence extraction
evidence = await evidence_extractor.extract(content)
# [
#   Evidence(
#     type="study",
#     content="IPCC 2023 report",
#     source="IPCC"
#   ),
#   Evidence(
#     type="statistic",
#     content="40% increase in Category 4+ hurricanes since 1980",
#     source="IPCC 2023 report"
#   ),
#   Evidence(
#     type="statistic",
#     content="$300 billion annually by 2030",
#     source=None
#   )
# ]
```

## Configuration

### Model Selection

Use smaller models for cost-effective extraction:

```python
# Cost-effective (recommended)
extractor = LLMCausalExtractor(model_name="gpt-4o-mini")

# Higher accuracy for critical applications
extractor = LLMCausalExtractor(model_name="gpt-4o")
```

### Disabling Cache

For debugging or when content changes frequently:

```python
extractor = LLMCausalExtractor(
    model_name="gpt-4o-mini",
    use_cache=False,  # Always call LLM
)
```

### Custom Model Instance

Pass a pre-configured model:

```python
from artemis.models.base import ModelRegistry

model = ModelRegistry.create("gpt-4o-mini", temperature=0)

extractor = LLMCausalExtractor(model=model)
```
