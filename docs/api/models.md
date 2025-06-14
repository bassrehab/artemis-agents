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
    model_name: str,
    api_key: str | None = None,
    **kwargs,
) -> BaseModel
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | str | Yes | Model identifier |
| `api_key` | str | No | API key (uses env var if not provided) |
| `**kwargs` | - | No | Additional model-specific options |

**Example:**

```python
# OpenAI
model = create_model("gpt-4o", temperature=0.7)

# Anthropic
model = create_model("claude-3-opus", max_tokens=4000)

# DeepSeek
model = create_model("deepseek-reasoner", timeout=120.0)
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
    temperature: float = 0.7,
    max_tokens: int = 4000,
    timeout: float = 60.0,
    max_retries: int = 3,
)
```

### Supported Models

| Model | Reasoning | Description |
|-------|-----------|-------------|
| `gpt-4o` | No | GPT-4 Optimized |
| `gpt-4-turbo` | No | GPT-4 Turbo |
| `o1` | Yes | OpenAI Reasoning |
| `o1-preview` | Yes | OpenAI Reasoning Preview |
| `o1-mini` | Yes | OpenAI Reasoning Mini |

---

## AnthropicModel

Anthropic model provider.

```python
from artemis.models.anthropic import AnthropicModel
```

### Constructor

```python
AnthropicModel(
    model: str = "claude-3-opus",
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4000,
    timeout: float = 60.0,
)
```

### Supported Models

| Model | Description |
|-------|-------------|
| `claude-3-opus` | Most capable |
| `claude-3-sonnet` | Balanced |
| `claude-3-haiku` | Fast |

---

## GoogleModel

Google AI model provider.

```python
from artemis.models.google import GoogleModel
```

### Constructor

```python
GoogleModel(
    model: str = "gemini-2.0-flash",
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4000,
)
```

### Supported Models

| Model | Reasoning | Description |
|-------|-----------|-------------|
| `gemini-2.0-flash` | No | Fast Gemini |
| `gemini-2.5-pro` | Yes | Pro with thinking |

---

## DeepSeekModel

DeepSeek model provider.

```python
from artemis.models.deepseek import DeepSeekModel
```

### Constructor

```python
DeepSeekModel(
    model: str = "deepseek-reasoner",
    api_key: str | None = None,
    temperature: float = 1.0,
    max_tokens: int = 8000,
    timeout: float = 120.0,
)
```

### Supported Models

| Model | Reasoning | Description |
|-------|-----------|-------------|
| `deepseek-chat` | No | Standard chat |
| `deepseek-reasoner` | Yes | R1 reasoning model |

---

## ReasoningConfig

Configuration for reasoning models.

```python
from artemis.models.reasoning import ReasoningConfig, create_reasoning_config
```

### Class Definition

```python
class ReasoningConfig(BaseModel):
    model: str
    strategy: str = "adaptive"  # always, adaptive, never
    thinking_budget: int = 8000
    show_thinking: bool = True
    temperature: float = 1.0
```

### create_reasoning_config

```python
def create_reasoning_config(
    model: str,
    thinking_budget: int = 8000,
    show_thinking: bool = True,
) -> ReasoningConfig
```

Creates appropriate config based on model capabilities.

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
from artemis.models.types import Message
```

### Class Definition

```python
class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str
```

---

## Response

Model response structure.

```python
from artemis.models.types import Response
```

### Class Definition

```python
class Response(BaseModel):
    content: str
    usage: Usage
    model: str
    finish_reason: str
```

---

## ReasoningResponse

Response from reasoning models.

```python
from artemis.models.types import ReasoningResponse
```

### Class Definition

```python
class ReasoningResponse(BaseModel):
    thinking: str  # The reasoning trace
    output: str    # The final output
    usage: Usage
    model: str
```

---

## Usage

Token usage information.

```python
from artemis.models.types import Usage
```

### Class Definition

```python
class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    thinking_tokens: int = 0  # For reasoning models
```

---

## list_providers

List available model providers.

```python
from artemis.models import list_providers

providers = list_providers()
# ['openai', 'anthropic', 'google', 'deepseek']
```

---

## list_models

List available models for a provider.

```python
from artemis.models import list_models

models = list_models("openai")
# ['gpt-4o', 'gpt-4-turbo', 'o1', 'o1-preview', 'o1-mini']
```

---

## Next Steps

- [Core API](core.md)
- [Safety API](safety.md)
- [Integrations API](integrations.md)
