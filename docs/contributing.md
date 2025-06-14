# Contributing to ARTEMIS

Thank you for your interest in contributing to ARTEMIS! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A GitHub account

### Setting Up Development Environment

1. **Fork and clone the repository**

```bash
git clone https://github.com/your-username/artemis-agents.git
cd artemis-agents
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

3. **Install development dependencies**

```bash
pip install -e ".[dev]"
```

4. **Set up environment variables**

```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Run tests to verify setup**

```bash
pytest tests/ -v
```

## Development Workflow

### Creating a Branch

```bash
git checkout -b feature/your-feature-name
# or: git checkout -b fix/your-bug-fix
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_debate.py -v

# Run with coverage
pytest tests/ --cov=artemis --cov-report=html
```

### Code Quality

```bash
# Type checking
mypy artemis/

# Linting
ruff check artemis/

# Formatting
ruff format artemis/
```

### Pre-commit Checks

Before committing, run:

```bash
# Format code
ruff format artemis/

# Check for issues
ruff check artemis/

# Run type checks
mypy artemis/

# Run tests
pytest tests/
```

## Code Guidelines

### Type Hints

Use Python 3.10+ type hints everywhere:

```python
# Good
def process_argument(
    argument: Argument,
    context: DebateContext | None = None,
) -> ProcessedArgument:
    ...

# Avoid
def process_argument(argument, context=None):
    ...
```

### Pydantic Models

Use Pydantic v2 for data classes:

```python
from pydantic import BaseModel, Field

class Argument(BaseModel):
    content: str = Field(description="The argument content")
    level: ArgumentLevel
    evidence: list[Evidence] = Field(default_factory=list)
```

### Async Code

Core debate logic should be async:

```python
# Good
async def generate_argument(self, context: DebateContext) -> Argument:
    response = await self.model.generate(messages)
    return self.parse_argument(response)

# Avoid sync in core logic
def generate_argument(self, context: DebateContext) -> Argument:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
async def evaluate(
    self,
    argument: Argument,
    context: DebateContext,
) -> Evaluation:
    """Evaluate an argument using adaptive criteria.

    Args:
        argument: The argument to evaluate.
        context: The debate context for evaluation.

    Returns:
        Evaluation result with scores and feedback.

    Raises:
        EvaluationError: If evaluation fails.
    """
    ...
```

### Error Handling

Use custom exceptions:

```python
from artemis.exceptions import DebateError, AgentError

try:
    result = await agent.generate_argument(context)
except ModelError as e:
    raise AgentError(f"Failed to generate argument: {e}") from e
```

## Adding New Features

### New LLM Provider

1. Create `artemis/models/{provider}.py`
2. Implement `BaseModel` interface
3. Add to `artemis/models/__init__.py`
4. Add tests in `tests/unit/models/test_{provider}.py`
5. Update documentation

```python
# artemis/models/newprovider.py
from artemis.models.base import BaseModel

class NewProviderModel(BaseModel):
    async def generate(
        self,
        messages: list[Message],
        **kwargs,
    ) -> Response:
        ...

    async def generate_with_reasoning(
        self,
        messages: list[Message],
        thinking_budget: int = 8000,
    ) -> ReasoningResponse:
        ...
```

### New Safety Monitor

1. Create `artemis/safety/{monitor}.py`
2. Implement `SafetyMonitor` interface
3. Add tests with mock scenarios
4. Document detection methodology

```python
# artemis/safety/newmonitor.py
from artemis.safety.base import SafetyMonitor, SafetyResult

class NewMonitor(SafetyMonitor):
    async def analyze(
        self,
        turn: Turn,
        context: DebateContext,
    ) -> SafetyResult:
        ...
```

### New Framework Integration

1. Create `artemis/integrations/{framework}.py`
2. Follow framework's extension patterns
3. Add example in `examples/`
4. Add integration tests

## Commit Guidelines

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

### Examples

```
feat(core): implement H-L-DAG argument generation

Add hierarchical argument generation with three levels:
- Strategic: high-level thesis
- Tactical: supporting points
- Operational: specific evidence
```

```
fix(models): handle o1 reasoning token limits

The o1 model has specific token limits for reasoning.
This fix ensures we properly manage the thinking budget.

Fixes #123
```

## Pull Request Process

1. **Update tests**: Add tests for new features
2. **Update docs**: Document new functionality
3. **Run checks**: Ensure all tests pass
4. **Create PR**: Use the PR template
5. **Address feedback**: Respond to review comments

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added
- [ ] All tests pass
- [ ] Manual testing done

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

## Testing Guidelines

### Unit Tests

Test individual components in isolation:

```python
@pytest.fixture
def mock_model():
    model = AsyncMock(spec=BaseModel)
    model.generate.return_value = Response(
        content="Test argument",
        usage=Usage(prompt_tokens=100, completion_tokens=50),
    )
    return model

async def test_agent_generates_argument(mock_model):
    agent = Agent(name="test", model=mock_model)
    argument = await agent.generate_argument(context)
    assert argument.content == "Test argument"
```

### Integration Tests

Test component interactions:

```python
async def test_debate_runs_complete():
    debate = Debate(
        topic="Test topic",
        agents=agents,
        rounds=2,
    )
    result = await debate.run()
    assert result.verdict is not None
    assert len(result.transcript) == 4  # 2 rounds * 2 agents
```

### Safety Tests

Verify monitor detection:

```python
async def test_detects_sandbagging():
    detector = SandbagDetector(sensitivity=0.8)

    # Create turns with declining capability
    turns = create_declining_capability_turns()

    for turn in turns:
        result = await detector.analyze(turn, context)

    assert any(r.score > 0.7 for r in results)
```

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Features**: Open a GitHub Issue with `[Feature Request]` prefix

## Code of Conduct

Please be respectful and inclusive in all interactions. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
