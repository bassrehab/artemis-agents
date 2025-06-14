# Installation

## Requirements

- Python 3.10 or higher
- pip or poetry for package management

## Install from PyPI

```bash
pip install artemis-agents
```

## Install from Source

For the latest development version:

```bash
git clone https://github.com/bassrehab/artemis-agents.git
cd artemis-agents
pip install -e ".[dev]"
```

## Optional Dependencies

ARTEMIS has optional dependencies for different use cases:

### Framework Integrations

```bash
# LangChain integration
pip install artemis-agents[langchain]

# LangGraph integration
pip install artemis-agents[langgraph]

# CrewAI integration
pip install artemis-agents[crewai]

# All integrations
pip install artemis-agents[integrations]
```

### Development

```bash
# Development tools (testing, linting, etc.)
pip install artemis-agents[dev]
```

### All Optional Dependencies

```bash
pip install artemis-agents[all]
```

## Environment Setup

### API Keys

ARTEMIS requires API keys for LLM providers. Set them as environment variables:

```bash
# OpenAI (GPT-4o, o1)
export OPENAI_API_KEY="your-openai-key"

# Anthropic (Claude)
export ANTHROPIC_API_KEY="your-anthropic-key"

# Google (Gemini)
export GOOGLE_API_KEY="your-google-key"

# DeepSeek (R1)
export DEEPSEEK_API_KEY="your-deepseek-key"
```

Or use a `.env` file:

```bash
# .env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

### Verify Installation

```python
import artemis

print(f"ARTEMIS version: {artemis.__version__}")

# Check available providers
from artemis.models import list_providers
print(f"Available providers: {list_providers()}")
```

## Docker

Run ARTEMIS in a Docker container:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install artemis-agents[all]

COPY . .

CMD ["python", "-m", "artemis.mcp.cli"]
```

Build and run:

```bash
docker build -t artemis-agents .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY artemis-agents
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you have the correct Python version:

```bash
python --version  # Should be 3.10+
```

### API Key Issues

If API calls fail, verify your keys are set:

```python
import os
print(os.environ.get("OPENAI_API_KEY", "Not set"))
```

### Dependency Conflicts

If you have dependency conflicts, try installing in a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
pip install artemis-agents
```

## Next Steps

Once installed, proceed to the [Quick Start](quickstart.md) guide to run your first debate.
