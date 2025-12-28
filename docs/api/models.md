# Models API Reference

This page documents the ARTEMIS model provider interfaces.

## create_model

Factory function for creating model instances.

```python
from artemis.models import create_model
```

### Signature

```python
def create_model(
    model: str,
    provider: str | None = None,
    **kwargs,
) -> BaseModel
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | str | Yes | Model identifier (e.g., "gpt-4o", "deepseek-reasoner") |
| `provider` | str | No | Provider name. If not specified, inferred from model name |
| `**kwargs` | - | No | Additional arguments passed to the model constructor |

**Example:**

```python
# OpenAI (provider inferred from model name)
model = create_model("gpt-4o")

# DeepSeek with explicit provider
model = create_model("deepseek-reasoner", provider="deepseek")

# With additional options
model = create_model("gpt-4o", timeout=120.0, max_retries=5)
```

---

## BaseModel

Abstract base class for all model providers.

```python
from artemis.models.base import BaseModel
```

### Abstract Methods

#### generate

```python
async def generate(
    self,
    messages: list[Message],
    **kwargs,
) -> Response
```

Generates a response from the model.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | list[Message] | Conversation messages |
| `**kwargs` | - | Additional options |

**Returns:** `Response` object.

#### generate_with_reasoning

```python
async def generate_with_reasoning(
    self,
    messages: list[Message],
    thinking_budget: int = 8000,
    **kwargs,
) -> ReasoningResponse
```

Generates a response with extended thinking.

**Returns:** `ReasoningResponse` with thinking and output.

---

## OpenAIModel

OpenAI model provider.

```python
from artemis.models.openai import OpenAIModel
```

### Constructor

```python
OpenAIModel(
    model: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    **kwargs,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "gpt-4o" | Model identifier |
| `api_key` | str \| None | None | API key (reads from OPENAI_API_KEY env var if not provided) |
| `base_url` | str \| None | None | Custom API base URL (for Azure or proxies) |
| `organization` | str \| None | None | OpenAI organization ID |
| `timeout` | float | 60.0 | Request timeout in seconds |
| `max_retries` | int | 3 | Maximum retry attempts |

### Supported Models

| Model | Reasoning | Description |
|-------|-----------|-------------|
| `gpt-4o` | No | GPT-4 Optimized |
| `gpt-4-turbo` | No | GPT-4 Turbo |
| `o1` | Yes | OpenAI Reasoning |
| `o1-preview` | Yes | OpenAI Reasoning Preview |
| `o1-mini` | Yes | OpenAI Reasoning Mini |

---

## DeepSeekModel

DeepSeek model provider with R1 reasoning support.

```python
from artemis.models.deepseek import DeepSeekModel
```

### Constructor

```python
DeepSeekModel(
    model: str = "deepseek-reasoner",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 3,
    **kwargs,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "deepseek-reasoner" | Model identifier |
| `api_key` | str \| None | None | API key (reads from DEEPSEEK_API_KEY env var if not provided) |
| `base_url` | str \| None | "https://api.deepseek.com/v1" | Custom API base URL |
| `timeout` | float | 120.0 | Request timeout in seconds |
| `max_retries` | int | 3 | Maximum retry attempts |

### Supported Models

| Model | Reasoning | Description |
|-------|-----------|-------------|
| `deepseek-chat` | No | Standard chat model |
| `deepseek-coder` | No | Code-specialized model |
| `deepseek-reasoner` | Yes | Full R1 with extended thinking |
| `deepseek-r1-distill-llama-70b` | Yes | Distilled R1 variant |

---

## ReasoningConfig

Configuration for reasoning models.

```python
from artemis.models import ReasoningConfig, ReasoningStrategy, create_reasoning_config
```

### Class Definition

```python
class ReasoningConfig(BaseModel):
    model: str = "o1"
    strategy: ReasoningStrategy = ReasoningStrategy.ADAPTIVE
    thinking_budget: int = 8000
    show_thinking: bool = False
    thinking_style: str = "thorough"  # "thorough", "concise", "analytical"
    temperature: float = 1.0
    use_system_prompt: bool = True
```

### ReasoningStrategy

```python
class ReasoningStrategy(str, Enum):
    ALWAYS = "always"     # Always use extended thinking
    ADAPTIVE = "adaptive" # Use reasoning only for complex problems
    NEVER = "never"       # Never use extended thinking
```

### create_reasoning_config

```python
def create_reasoning_config(
    model: str,
    **overrides,
) -> ReasoningConfig
```

Creates appropriate config with model-specific defaults.

**Example:**

```python
# Automatic configuration for o1
config = create_reasoning_config("o1", thinking_budget=16000)

# For DeepSeek R1
config = create_reasoning_config("deepseek-reasoner", show_thinking=True)
```

---

## Message

Message structure for model calls.

```python
from artemis.core.types import Message
```

### Class Definition

```python
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    name: str | None = None  # Optional name for multi-agent scenarios
```

---

## ModelResponse

Model response structure.

```python
from artemis.core.types import ModelResponse
```

### Class Definition

```python
class ModelResponse(BaseModel):
    content: str
    usage: Usage
    model: str | None = None
    finish_reason: str | None = None
```

---

## ReasoningResponse

Response from reasoning models.

```python
from artemis.core.types import ReasoningResponse
```

### Class Definition

```python
class ReasoningResponse(ModelResponse):
    thinking: str | None = None  # The extended thinking/reasoning trace
    thinking_tokens: int = 0
```

---

## Usage

Token usage information.

```python
from artemis.core.types import Usage
```

### Class Definition

```python
class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None  # For reasoning models
```

---

## list_providers

List available model providers.

```python
from artemis.models import list_providers

providers = list_providers()
# ['openai', 'deepseek']
```

---

## is_reasoning_model

Check if a model supports extended reasoning.

```python
from artemis.models import is_reasoning_model

is_reasoning_model("o1")  # True
is_reasoning_model("gpt-4o")  # False
```

---

## Next Steps

- [Core API](core.md)
- [Safety API](safety.md)
- [Integrations API](integrations.md)
