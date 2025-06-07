# Contributing to ARTEMIS Agents

Thank you for your interest in contributing to ARTEMIS Agents! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and constructive. We're here to build something great together.

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- An API key for at least one LLM provider (OpenAI, Anthropic, or Google)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/bassrehab/artemis-agents.git
cd artemis-agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify setup
pytest tests/ -v
```

### Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Add your API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## Development Workflow

### 1. Create a Branch

```bash
# For features
git checkout -b feat/your-feature-name

# For bug fixes
git checkout -b fix/issue-description

# For documentation
git checkout -b docs/what-youre-documenting
```

### 2. Make Your Changes

- Write code following our style guide (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=artemis --cov-report=html

# Run specific test file
pytest tests/unit/core/test_debate.py -v

# Run type checking
mypy artemis/

# Run linting
ruff check artemis/
ruff format artemis/
```

### 4. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with conventional commit message
git commit -m "feat(core): add new argument validation"
```

### 5. Push and Create PR

```bash
git push origin your-branch-name
```

Then create a Pull Request on GitHub.

## Code Style Guide

### Python Style

- **Type hints**: Required for all public functions and methods
- **Docstrings**: Google-style for all public APIs
- **Line length**: 100 characters max
- **Imports**: Use `isort` ordering (handled by ruff)

### Example

```python
async def generate_argument(
    self,
    context: DebateContext,
    level: ArgumentLevel = ArgumentLevel.TACTICAL
) -> Argument:
    """Generate an argument at the specified hierarchical level.
    
    Uses the H-L-DAG framework to create structured arguments with
    evidence and causal reasoning.
    
    Args:
        context: Current debate context including topic and transcript.
        level: Hierarchical level for argument generation.
            - STRATEGIC: High-level thesis and position
            - TACTICAL: Supporting arguments and evidence
            - OPERATIONAL: Specific facts and examples
    
    Returns:
        A structured Argument with content, evidence, and causal links.
    
    Raises:
        AgentError: If argument generation fails.
        ModelError: If the LLM provider returns an error.
    
    Example:
        >>> agent = Agent(name="Proponent", role="Argues in favor", model="gpt-4o")
        >>> context = DebateContext(topic="AI rights", current_round=1, total_rounds=3)
        >>> argument = await agent.generate_argument(context)
        >>> print(argument.content)
    """
    # Implementation...
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Examples:
```
feat(safety): add deception detection monitor
fix(models): handle rate limiting in OpenAI provider
docs: add API reference for JuryPanel
test(core): add integration tests for debate orchestrator
```

## Testing Guidelines

### Unit Tests

- Test one thing per test function
- Use descriptive test names
- Mock external dependencies (especially LLM calls)

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_agent_generates_strategic_argument():
    """Agent should generate high-level argument at strategic level."""
    # Arrange
    mock_model = AsyncMock()
    mock_model.generate.return_value = Response(
        content="Strategic argument content...",
        usage=Usage(prompt_tokens=100, completion_tokens=50)
    )
    
    agent = Agent(name="Test", role="Tester", model=mock_model)
    context = DebateContext(topic="Test topic", current_round=1, total_rounds=3)
    
    # Act
    argument = await agent.generate_argument(context, level=ArgumentLevel.STRATEGIC)
    
    # Assert
    assert argument.level == ArgumentLevel.STRATEGIC
    assert "Strategic" in mock_model.generate.call_args[1]["messages"][0].content
```

### Integration Tests

- Test component interactions
- Use real (but mocked) workflows
- Clean up any state after tests

### Benchmark Tests

- Located in `benchmarks/`
- Compare against other frameworks
- Document methodology

## Pull Request Guidelines

### Before Submitting

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Type checking passes (`mypy artemis/`)
- [ ] Linting passes (`ruff check artemis/`)
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated

### PR Description Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactoring

## Testing
Describe how you tested these changes.

## Related Issues
Fixes #123
```

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address all review comments
4. Maintainer will merge when ready

## Architecture Guidelines

### Adding a New Model Provider

1. Create `artemis/models/newprovider.py`
2. Implement `BaseModel` interface
3. Handle provider-specific quirks (rate limiting, token counting)
4. Add to `artemis/models/__init__.py`
5. Add tests in `tests/unit/models/test_newprovider.py`
6. Update documentation

### Adding a New Safety Monitor

1. Create `artemis/safety/newmonitor.py`
2. Implement `SafetyMonitor` interface
3. Define clear detection criteria
4. Document false positive/negative rates
5. Add tests with adversarial scenarios

### Adding a Framework Integration

1. Create `artemis/integrations/framework.py`
2. Follow the framework's extension patterns
3. Create example in `examples/`
4. Add integration test

## Documentation

### Where to Document

- **Code**: Docstrings for all public APIs
- **README.md**: High-level overview and quick start
- **examples/**: Runnable code examples

### Documentation Style

- Use clear, simple language
- Include code examples
- Show expected outputs
- Link to related sections

## Questions?

- Open a [GitHub Discussion](https://github.com/bassrehab/artemis-agents/discussions)
- Check existing issues and PRs
- Read the [README](README.md)

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md
- Release notes
- Project README (for significant contributions)

Thank you for contributing to ARTEMIS Agents! 🎉
