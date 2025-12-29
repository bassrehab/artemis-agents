# Centralized Prompts System

ARTEMIS uses a centralized prompt management system for versioning, A/B testing, and maintainability. All prompts are stored in `artemis/prompts/` and accessed through a unified API.

## Why Centralized Prompts?

- **Versioning**: Test new prompt versions without code changes
- **A/B Testing**: Compare prompt performance across versions
- **Maintainability**: Single source of truth for all prompts
- **Rollback**: Quickly revert to previous prompt versions

## Usage

### Basic Usage

```python
from artemis.prompts import get_prompt, list_prompts

# Get a prompt by key
prompt = get_prompt("hdag.strategic_instructions")

# Get prompt with formatting variables
prompt = get_prompt(
    "evaluation.user",
    topic="AI Safety",
    level="strategic",
    content="...",
    round=1,
    total_rounds=3,
    position="pro",
    prev_count=2,
)

# List all available prompts
prompts = list_prompts()
# Returns: {'hdag': ['STRATEGIC_INSTRUCTIONS', ...], 'evaluation': [...], ...}
```

### Prompt Key Format

Prompts are accessed using dot notation: `module.prompt_name`

- `hdag.strategic_instructions` - H-L-DAG strategic level prompt
- `evaluation.system` - Evaluation system prompt
- `extraction.causal_extraction` - Causal link extraction prompt
- `jury.analytical_perspective` - Analytical juror perspective

## Available Modules

| Module | Description | Key Prompts |
|--------|-------------|-------------|
| `hdag` | H-L-DAG argument generation | `STRATEGIC_INSTRUCTIONS`, `TACTICAL_INSTRUCTIONS`, `OPERATIONAL_INSTRUCTIONS`, `OPENING_STATEMENT`, `CLOSING_STATEMENT` |
| `evaluation` | LLM-based evaluation | `SYSTEM`, `USER`, `CRITERIA_DEFINITIONS`, `DEFAULT_WEIGHTS` |
| `extraction` | Evidence/causal extraction | `CAUSAL_EXTRACTION`, `EVIDENCE_EXTRACTION` |
| `jury` | Jury perspectives | `ANALYTICAL_PERSPECTIVE`, `ETHICAL_PERSPECTIVE`, `PRACTICAL_PERSPECTIVE`, `ADVERSARIAL_PERSPECTIVE` |
| `benchmark` | Benchmark evaluation | `ARGUMENT_QUALITY`, `DECISION_ACCURACY`, `REASONING_DEPTH` |

## Versioning

Prompts are organized by version (`v1`, `v2`, etc.) for controlled experimentation.

### Default Version

The default version is `v1`. All `get_prompt()` calls use this unless specified otherwise.

```python
from artemis.prompts import get_prompt_version

current = get_prompt_version()  # Returns "v1"
```

### Switching Versions

#### Global Version Switch

```python
from artemis.prompts.loader import set_prompt_version

# Switch all prompts to v2
set_prompt_version("v2")

# Now all get_prompt() calls use v2
prompt = get_prompt("hdag.strategic_instructions")  # Uses v2
```

#### Per-Call Version

```python
from artemis.prompts import get_prompt

# Use v2 for this specific call
prompt = get_prompt("hdag.strategic_instructions", version="v2")

# Default version unchanged for other calls
other_prompt = get_prompt("evaluation.system")  # Still uses v1
```

## Adding New Prompts

### 1. Add to Existing Module

Edit the appropriate file in `artemis/prompts/v1/`:

```python
# artemis/prompts/v1/hdag.py

# Add new constant (uppercase)
MY_NEW_PROMPT = """Your prompt content here.

Variables use {curly_braces} for formatting:
Topic: {topic}
"""
```

Access with:
```python
prompt = get_prompt("hdag.my_new_prompt", topic="AI Safety")
```

### 2. Create New Version

Copy the `v1/` directory to `v2/` and modify prompts:

```
artemis/prompts/
├── v1/
│   ├── hdag.py
│   └── ...
└── v2/          # New version
    ├── hdag.py  # Modified prompts
    └── ...
```

Register in `artemis/prompts/loader.py`:

```python
PROMPT_MODULES = {
    "v1": {...},
    "v2": {
        "hdag": "artemis.prompts.v2.hdag",
        # ... other modules
    },
}
```

## Prompt Module Structure

Each prompt module follows this structure:

```python
# artemis/prompts/v1/example.py
"""
Example Prompts - Version 1

Description of what these prompts are for.
"""

# Prompts are uppercase constants
SYSTEM_PROMPT = """You are an assistant..."""

USER_PROMPT = """Given the following:
Topic: {topic}
Context: {context}

Please provide your analysis."""

# Supporting constants (also accessible)
DEFAULT_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 1000,
}
```

## Integration Example

Here's how the prompt system integrates with ARTEMIS components:

```python
from artemis.prompts import get_prompt
from artemis.core.types import Message

# In LLM evaluation
system_prompt = get_prompt("evaluation.system")
user_prompt = get_prompt(
    "evaluation.user",
    topic=context.topic,
    level=argument.level.value,
    content=argument.content[:2000],
    round=context.current_round,
    total_rounds=context.total_rounds,
    position=position,
    prev_count=len(context.transcript),
)

messages = [
    Message(role="system", content=system_prompt),
    Message(role="user", content=user_prompt),
]

response = await model.generate(messages=messages)
```

## Best Practices

1. **Use descriptive names**: `STRATEGIC_INSTRUCTIONS` not `PROMPT1`
2. **Document variables**: Comment which `{variables}` the prompt expects
3. **Keep prompts focused**: One purpose per prompt
4. **Test before deploying**: Verify new versions in benchmarks
5. **Version incrementally**: Small changes per version for easier debugging
