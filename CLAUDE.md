# CLAUDE.md

This file provides guidance to Claude Code (or any AI coding assistant) for working on the ARTEMIS Agents project.

## Project Overview

ARTEMIS (Adaptive Reasoning Through Evaluation of Multi-agent Intelligent Systems) is a Python framework for structured multi-agent debates with adaptive evaluation. It implements the research paper published in Google Technical Disclosure Commons.

**Key differentiators:**
1. Hierarchical Argument Generation (H-L-DAG) - not just chat, structured arguments
2. Adaptive Evaluation with Causal Reasoning (L-AE-CR) - dynamic criteria weighting
3. Jury Scoring Mechanism - multi-perspective evaluation
4. Safety monitoring - sandbagging/deception detection (unique to this project)

## Repository Structure

```
artemis-agents/
├── artemis/                    # Main package
│   ├── __init__.py
│   ├── core/                   # Core ARTEMIS implementation
│   │   ├── __init__.py
│   │   ├── debate.py           # Main Debate orchestrator
│   │   ├── agent.py            # Agent class with H-L-DAG
│   │   ├── argument.py         # Argument data structures
│   │   ├── jury.py             # Jury scoring mechanism
│   │   ├── evaluation.py       # L-AE-CR adaptive evaluation
│   │   └── ethics.py           # Ethical alignment module
│   │
│   ├── models/                 # LLM provider integrations
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base model
│   │   ├── openai.py           # OpenAI (GPT-4o, o1)
│   │   ├── anthropic.py        # Anthropic (Claude)
│   │   ├── google.py           # Google (Gemini)
│   │   ├── deepseek.py         # DeepSeek (R1)
│   │   └── reasoning.py        # Reasoning model configs
│   │
│   ├── safety/                 # Safety monitoring (YOUR MOAT)
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract monitor
│   │   ├── sandbagging.py      # Sandbagging detection
│   │   ├── deception.py        # Deception monitoring
│   │   ├── behavior.py         # Behavioral drift tracking
│   │   └── metacognition.py    # Integration with ai-metacognition-toolkit
│   │
│   ├── integrations/           # Framework integrations
│   │   ├── __init__.py
│   │   ├── langchain.py        # LangChain tool
│   │   ├── langgraph.py        # LangGraph node
│   │   └── crewai.py           # CrewAI integration
│   │
│   ├── mcp/                    # MCP server implementation
│   │   ├── __init__.py
│   │   ├── server.py           # MCP server
│   │   └── tools.py            # MCP tool definitions
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── logging.py
│       ├── config.py
│       └── types.py
│
├── tests/                      # Test suite
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
│
├── examples/                   # Usage examples
│   ├── basic_debate.py
│   ├── ethical_dilemma.py
│   ├── enterprise_decision.py
│   ├── with_safety_monitors.py
│   └── langgraph_workflow.py
│
├── benchmarks/                 # Performance benchmarks
│   ├── vs_autogen.py
│   ├── vs_crewai.py
│   └── debate_quality.py
│
├── docs/                       # Documentation
│   ├── DESIGN.md
│   ├── API.md
│   ├── ARCHITECTURE.md
│   └── PLAN.md
│
├── pyproject.toml
├── README.md
├── CLAUDE.md                   # This file
├── CONTRIBUTING.md
└── LICENSE
```

## Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_debate.py -v

# Run with coverage
pytest tests/ --cov=artemis --cov-report=html

# Type checking
mypy artemis/

# Linting
ruff check artemis/
ruff format artemis/

# Run a single example
python examples/basic_debate.py

# Start MCP server
python -m artemis.mcp.server --port 8080
```

## Code Style Guidelines

1. **Type hints everywhere** - Use Python 3.10+ type hints, including `|` union syntax
2. **Pydantic models** - Use Pydantic v2 for all data classes
3. **Async by default** - Core debate logic should be async-first
4. **Docstrings** - Google-style docstrings for all public methods
5. **Error handling** - Custom exceptions in `artemis/exceptions.py`

## Key Design Decisions

### 1. H-L-DAG (Hierarchical Argument Generation)

Arguments are generated at three levels:
- **Strategic**: High-level position and thesis
- **Tactical**: Supporting points and evidence
- **Operational**: Specific facts, quotes, examples

```python
class Argument(BaseModel):
    level: Literal["strategic", "tactical", "operational"]
    content: str
    evidence: list[Evidence]
    causal_links: list[CausalLink]
    ethical_score: float
```

### 2. L-AE-CR (Adaptive Evaluation with Causal Reasoning)

Evaluation criteria weights are dynamically adjusted based on:
- Debate topic domain
- Current round context
- Causal relationship strength
- Ethical implications

### 3. Safety Monitoring Architecture

Monitors are injected as middleware and run after each agent turn:

```python
class SafetyMonitor(ABC):
    @abstractmethod
    async def analyze(self, turn: Turn, context: DebateContext) -> SafetyResult:
        pass
```

### 4. Provider Abstraction

All LLM providers implement a common interface:

```python
class BaseModel(ABC):
    @abstractmethod
    async def generate(self, messages: list[Message], **kwargs) -> Response:
        pass
    
    @abstractmethod
    async def generate_with_reasoning(self, messages: list[Message], thinking_budget: int) -> ReasoningResponse:
        pass
```

## Important Implementation Notes

### Reasoning Model Support

For o1/R1/Gemini 2.5 Pro, we need special handling:
- Extended thinking tokens are separate from output
- Some providers don't support system prompts in reasoning mode
- Token budgets need explicit management

```python
# Pattern for reasoning models
if agent.reasoning_config.enabled:
    response = await model.generate_with_reasoning(
        messages=messages,
        thinking_budget=agent.reasoning_config.thinking_budget
    )
    # response.thinking contains the reasoning trace
    # response.output contains the final argument
```

### MCP Server Implementation

The MCP server exposes these tools:
- `artemis_debate`: Start a new debate
- `artemis_add_round`: Add a debate round
- `artemis_get_verdict`: Get jury verdict
- `artemis_get_transcript`: Get full debate transcript

### Safety Monitor Integration

Monitors can be:
- **Passive**: Only observe and report (default)
- **Active**: Can intervene and halt debate

```python
debate = Debate(
    monitors=[
        SandbagDetector(mode="active", halt_threshold=0.9),
        DeceptionMonitor(mode="passive")
    ]
)
```

## Testing Strategy

1. **Unit tests**: Test individual components in isolation
2. **Integration tests**: Test component interactions
3. **Benchmark tests**: Compare against other frameworks
4. **Safety tests**: Verify monitor detection capabilities

Mock LLM responses for unit tests:

```python
@pytest.fixture
def mock_model():
    model = AsyncMock(spec=BaseModel)
    model.generate.return_value = Response(
        content="Test argument",
        usage=Usage(prompt_tokens=100, completion_tokens=50)
    )
    return model
```

## Git Commit Conventions

Format: `<type>(<scope>): <description>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

Examples:
- `feat(core): implement H-L-DAG argument generation`
- `feat(safety): add sandbagging detection monitor`
- `fix(models): handle o1 reasoning token limits`
- `docs: add API reference for Debate class`

## Dependencies

Core:
- `pydantic>=2.0` - Data validation
- `httpx` - Async HTTP client
- `openai>=1.0` - OpenAI API
- `anthropic>=0.20` - Anthropic API
- `google-generativeai` - Google AI API

Optional:
- `langchain-core` - LangChain integration
- `langgraph` - LangGraph integration
- `crewai` - CrewAI integration
- `mcp` - MCP server support

Dev:
- `pytest`, `pytest-asyncio`, `pytest-cov`
- `mypy`
- `ruff`

## Common Tasks for Claude Code

### Adding a new LLM provider

1. Create `artemis/models/{provider}.py`
2. Implement `BaseModel` interface
3. Add to `artemis/models/__init__.py`
4. Add tests in `tests/unit/models/test_{provider}.py`
5. Update docs

### Adding a new safety monitor

1. Create `artemis/safety/{monitor}.py`
2. Implement `SafetyMonitor` interface
3. Add tests with mock scenarios
4. Document detection methodology

### Adding a new framework integration

1. Create `artemis/integrations/{framework}.py`
2. Follow framework's extension patterns
3. Add example in `examples/`
4. Add integration test

## Questions to Ask When Stuck

1. "Does this align with the H-L-DAG architecture from the paper?"
2. "Is this async-compatible?"
3. "Does this work with all supported model providers?"
4. "Have I considered the safety implications?"
5. "Is this tested?"

## External Resources

- [ARTEMIS Paper (TD Commons)](https://www.tdcommons.org/dpubs_series/7729/)
- [AI Metacognition Toolkit](https://github.com/bassrehab/ai-metacognition-toolkit) - For safety monitor patterns
- [Steering Vectors Research](https://github.com/bassrehab/steering-vectors-agents) - Future integration
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [MCP Specification](https://modelcontextprotocol.io/)
