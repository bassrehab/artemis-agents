# MCP Server Integration

ARTEMIS provides an MCP (Model Context Protocol) server, allowing any MCP-compatible client to use structured debates.

## Overview

The MCP server exposes ARTEMIS capabilities as tools that can be called by LLM clients that support the Model Context Protocol.

## Starting the Server

### CLI

```bash
# Start with default settings
artemis-mcp

# HTTP mode
artemis-mcp --http --port 8080

# With custom model
artemis-mcp --model gpt-4-turbo

# Verbose logging
artemis-mcp --verbose
```

### Programmatic

```python
from artemis.mcp import ArtemisMCPServer

server = ArtemisMCPServer(
    default_model="gpt-4o",
    max_sessions=100,
)

# Stdio mode (for MCP clients)
await server.run_stdio()

# Or HTTP mode
await server.start(host="127.0.0.1", port=8080)
```

## Available Tools

The MCP server exposes these tools:

### artemis_debate_start

Start a new debate:

```json
{
    "name": "artemis_debate_start",
    "arguments": {
        "topic": "Should we adopt microservices?",
        "rounds": 3,
        "positions": {
            "pro": "supports microservices",
            "con": "opposes microservices"
        }
    }
}
```

**Returns:**

```json
{
    "debate_id": "d-123456",
    "status": "running",
    "topic": "Should we adopt microservices?"
}
```

### artemis_add_round

Add a round to an existing debate:

```json
{
    "name": "artemis_add_round",
    "arguments": {
        "debate_id": "d-123456"
    }
}
```

**Returns:**

```json
{
    "round": 2,
    "turns": [
        {
            "agent": "pro",
            "argument": "...",
            "score": 7.5
        },
        {
            "agent": "con",
            "argument": "...",
            "score": 7.2
        }
    ]
}
```

### artemis_get_verdict

Get the jury's verdict:

```json
{
    "name": "artemis_get_verdict",
    "arguments": {
        "debate_id": "d-123456"
    }
}
```

**Returns:**

```json
{
    "verdict": "pro",
    "confidence": 0.73,
    "reasoning": "The proponent presented stronger evidence...",
    "jury_votes": [
        {"juror": "logical", "vote": "pro"},
        {"juror": "ethical", "vote": "con"},
        {"juror": "practical", "vote": "pro"}
    ]
}
```

### artemis_get_transcript

Get the full debate transcript:

```json
{
    "name": "artemis_get_transcript",
    "arguments": {
        "debate_id": "d-123456"
    }
}
```

**Returns:**

```json
{
    "topic": "Should we adopt microservices?",
    "rounds": 3,
    "transcript": [
        {
            "round": 1,
            "agent": "pro",
            "argument": {
                "level": "strategic",
                "content": "..."
            }
        }
    ]
}
```

### artemis_list_debates

List all active debates:

```json
{
    "name": "artemis_list_debates",
    "arguments": {}
}
```

### artemis_get_safety_report

Get safety monitoring report:

```json
{
    "name": "artemis_get_safety_report",
    "arguments": {
        "debate_id": "d-123456"
    }
}
```

## Server Configuration

### Full Options

```python
from artemis.mcp import ArtemisMCPServer
from artemis.core.types import DebateConfig

config = DebateConfig(
    max_round_time_seconds=300,
    jury_size=3,
    enable_safety_monitoring=True,
)

server = ArtemisMCPServer(
    # Model settings
    default_model="gpt-4o",
    api_keys={
        "openai": "sk-...",
        "anthropic": "sk-ant-...",
    },

    # Session settings
    max_sessions=100,
    session_timeout=3600,

    # Debate settings
    config=config,

    # Server settings
    enable_logging=True,
    log_level="INFO",
)
```

### Environment Variables

The server respects these environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ARTEMIS_DEFAULT_MODEL` | Default model |
| `ARTEMIS_LOG_LEVEL` | Logging level |
| `ARTEMIS_MAX_SESSIONS` | Max concurrent sessions |

## Client Usage

### Claude Desktop

Add to your Claude Desktop config:

```json
{
    "mcpServers": {
        "artemis": {
            "command": "artemis-mcp",
            "args": ["--model", "gpt-4o"]
        }
    }
}
```

### Other MCP Clients

Connect to the HTTP endpoint:

```python
import httpx

async with httpx.AsyncClient() as client:
    # Start debate
    response = await client.post(
        "http://localhost:8080/tools/artemis_debate_start",
        json={
            "topic": "Your topic",
            "rounds": 3,
        }
    )

    debate_id = response.json()["debate_id"]

    # Get verdict
    response = await client.post(
        "http://localhost:8080/tools/artemis_get_verdict",
        json={"debate_id": debate_id}
    )

    print(response.json())
```

## Session Management

### Session Lifecycle

```
┌─────────────────┐
│  Start Debate   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Session Active │◄──────┐
└────────┬────────┘       │
         │                │
         ▼                │
┌─────────────────┐       │
│   Add Rounds    │───────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Get Verdict   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Session Ended  │
└─────────────────┘
```

### Session Persistence

Sessions can be persisted:

```python
from artemis.mcp import ArtemisMCPServer
from artemis.mcp.sessions import FileSessionStore

server = ArtemisMCPServer(
    session_store=FileSessionStore("./sessions"),
)

# Sessions survive server restarts
```

## Safety Monitoring

Enable safety monitoring in the server:

```python
from artemis.mcp import ArtemisMCPServer
from artemis.safety import CompositeMonitor

server = ArtemisMCPServer(
    safety_monitor=CompositeMonitor.default(),
    halt_on_safety_violation=False,
)
```

Safety alerts are included in responses:

```json
{
    "debate_id": "d-123456",
    "verdict": "pro",
    "safety_alerts": [
        {
            "type": "deception",
            "severity": 0.45,
            "agent": "con",
            "round": 2
        }
    ]
}
```

## Error Handling

The server returns structured errors:

```json
{
    "error": {
        "code": "DEBATE_NOT_FOUND",
        "message": "Debate d-999999 not found",
        "details": {}
    }
}
```

Error codes:

| Code | Description |
|------|-------------|
| `DEBATE_NOT_FOUND` | Debate ID doesn't exist |
| `INVALID_ARGUMENTS` | Missing or invalid arguments |
| `DEBATE_COMPLETED` | Cannot modify completed debate |
| `RATE_LIMITED` | Too many requests |
| `INTERNAL_ERROR` | Server error |

## Monitoring

### Health Check

```bash
curl http://localhost:8080/health
```

### Metrics

```bash
curl http://localhost:8080/metrics
```

Returns:

```json
{
    "active_debates": 5,
    "total_debates": 127,
    "uptime_seconds": 3600,
    "model_usage": {
        "gpt-4o": 95,
        "claude-3-opus": 32
    }
}
```

## Best Practices

1. **Use session IDs**: Track debates across requests
2. **Handle timeouts**: Debates can take time
3. **Check safety alerts**: Review alerts in responses
4. **Clean up sessions**: End completed debates
5. **Monitor usage**: Watch metrics for issues

## Next Steps

- Learn about [LangChain Integration](langchain.md)
- Explore [LangGraph Integration](langgraph.md)
- Configure [CrewAI Integration](crewai.md)
