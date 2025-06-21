# LangChain Integration

ARTEMIS provides native LangChain integration, allowing you to use debates as tools in LangChain chains and agents.

## Installation

```bash
pip install artemis-agents[langchain]
```

## ArtemisDebateTool

The primary integration point is the `ArtemisDebateTool`:

```python
from artemis.integrations import ArtemisDebateTool

# Create the tool
debate_tool = ArtemisDebateTool(
    model="gpt-4o",
    default_rounds=3,
)

# Use directly
result = debate_tool.invoke({
    "topic": "Should we adopt microservices architecture?",
    "rounds": 2,
})

print(result.verdict)
```

## Tool Configuration

### Full Options

```python
from artemis.integrations import ArtemisDebateTool
from artemis.core.types import DebateConfig

config = DebateConfig(
    turn_timeout=60,
    round_timeout=300,
    require_evidence=True,
    safety_mode="passive",
)

tool = ArtemisDebateTool(
    model="gpt-4o",
    default_rounds=3,
    config=config,
)
```

### Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | required | LLM model to use |
| `default_rounds` | int | 3 | Default debate rounds |
| `config` | DebateConfig | None | Debate configuration |
| `name` | str | "artemis_debate" | Tool name |
| `description` | str | ... | Tool description |

## With LangChain Agents

### Using with AgentExecutor

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from artemis.integrations import ArtemisDebateTool

# Create the tool
debate_tool = ArtemisDebateTool(model="gpt-4o")

# Create agent
llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can run debates to analyze topics."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_functions_agent(llm, [debate_tool], prompt)
executor = AgentExecutor(agent=agent, tools=[debate_tool])

# Run
result = executor.invoke({
    "input": "Analyze the pros and cons of remote work by running a debate"
})
```

### Using with LCEL

```python
from langchain_core.runnables import RunnableLambda
from artemis.integrations import ArtemisDebateTool

debate_tool = ArtemisDebateTool(model="gpt-4o")

# Create a chain
chain = (
    RunnableLambda(lambda x: {"topic": x["question"], "rounds": 2})
    | debate_tool
    | RunnableLambda(lambda x: f"Verdict: {x.verdict.decision}")
)

result = chain.invoke({"question": "Is Python better than JavaScript?"})
```

## Tool Output

The tool returns a structured result:

```python
result = debate_tool.invoke({"topic": "Your topic"})

# Access verdict
print(result.verdict.decision)    # "pro", "con", or "tie"
print(result.verdict.confidence)  # 0.0 to 1.0
print(result.verdict.reasoning)   # Explanation

# Access transcript
for turn in result.transcript:
    print(f"{turn.agent}: {turn.argument.content}")

# Access safety alerts
for alert in result.safety_alerts:
    print(f"Alert: {alert.type}")
```

## Multiple Debate Agents

You can specify custom agents with positions:

```python
result = debate_tool.invoke({
    "topic": "Which database should we use?",
    "agents": [
        {"name": "sql_advocate", "role": "Database expert", "position": "advocates for SQL databases"},
        {"name": "nosql_advocate", "role": "Database expert", "position": "advocates for NoSQL databases"},
    ],
    "rounds": 3,
})
```

## With Safety Monitoring

Enable safety monitoring in debates:

```python
from artemis.integrations import ArtemisDebateTool
from artemis.safety import SandbagDetector, DeceptionMonitor, MonitorMode

tool = ArtemisDebateTool(
    model="gpt-4o",
    safety_monitors=[
        SandbagDetector(mode=MonitorMode.PASSIVE, sensitivity=0.7),
        DeceptionMonitor(mode=MonitorMode.PASSIVE, sensitivity=0.6),
    ],
)
```

## Async Support

The tool supports async execution:

```python
from artemis.integrations import ArtemisDebateTool

tool = ArtemisDebateTool(model="gpt-4o")

# Async invocation
result = await tool.ainvoke({
    "topic": "Should we use async programming?",
})
```

## Custom Tool Schema

Customize the tool's input schema:

```python
from artemis.integrations import ArtemisDebateTool
from pydantic import BaseModel, Field

class DebateInput(BaseModel):
    topic: str = Field(description="The debate topic")
    rounds: int = Field(default=3, description="Number of rounds")
    require_evidence: bool = Field(default=True, description="Require evidence")

tool = ArtemisDebateTool(
    model="gpt-4o",
    args_schema=DebateInput,
)
```

## Integration Patterns

### Decision Support

```python
from langchain.agents import AgentExecutor
from artemis.integrations import ArtemisDebateTool

debate_tool = ArtemisDebateTool(model="gpt-4o")

# Use debate for decision support
chain = create_decision_chain(
    tools=[debate_tool],
    system_prompt="""
    When faced with a decision, use the debate tool to analyze
    pros and cons before making a recommendation.
    """,
)
```

### Research Assistant

```python
# Combine with search tools
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
debate = ArtemisDebateTool(model="gpt-4o")

tools = [search, debate]

# Agent can search for info, then debate the findings
```

### Analysis Pipeline

```python
from langchain_core.runnables import RunnableParallel

# Run multiple debates in parallel
parallel = RunnableParallel(
    tech_debate=ArtemisDebateTool(model="gpt-4o"),
    business_debate=ArtemisDebateTool(model="gpt-4o"),
)

results = parallel.invoke({
    "tech_debate": {"topic": "Technical feasibility of X"},
    "business_debate": {"topic": "Business viability of X"},
})
```

## Error Handling

```python
from artemis.integrations import ArtemisDebateTool
from artemis.exceptions import DebateError

tool = ArtemisDebateTool(model="gpt-4o")

try:
    result = tool.invoke({"topic": "Your topic"})
except DebateError as e:
    print(f"Debate failed: {e}")
```

## Callbacks

Use LangChain callbacks with the tool:

```python
from langchain.callbacks import StdOutCallbackHandler

tool = ArtemisDebateTool(model="gpt-4o")

result = tool.invoke(
    {"topic": "Your topic"},
    config={"callbacks": [StdOutCallbackHandler()]},
)
```

## Next Steps

- Learn about [LangGraph Integration](langgraph.md)
- Explore [CrewAI Integration](crewai.md)
- Configure [MCP Server](mcp.md)
