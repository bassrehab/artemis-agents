# CrewAI Integration

ARTEMIS integrates with CrewAI, allowing you to use structured debates as tools within CrewAI crews.

## Installation

```bash
pip install artemis-agents[crewai]
```

## ArtemisCrewTool

The primary integration is the `ArtemisCrewTool`:

```python
from artemis.integrations import ArtemisCrewTool

# Create the tool
debate_tool = ArtemisCrewTool(
    model="gpt-4o",
    default_rounds=3,
)
```

## Basic Usage

### With a Crew

```python
from crewai import Agent, Task, Crew
from artemis.integrations import ArtemisCrewTool

# Create the debate tool
debate_tool = ArtemisCrewTool(model="gpt-4o")

# Create an agent with the tool
analyst = Agent(
    role="Decision Analyst",
    goal="Analyze decisions using structured debate",
    backstory="Expert at evaluating options through rigorous debate",
    tools=[debate_tool],
)

# Create a task
task = Task(
    description="Analyze whether we should migrate to microservices",
    agent=analyst,
    expected_output="A detailed analysis with a recommendation",
)

# Create and run crew
crew = Crew(
    agents=[analyst],
    tasks=[task],
)

result = crew.kickoff()
```

## Tool Configuration

```python
from artemis.integrations import ArtemisCrewTool
from artemis.core.types import DebateConfig

config = DebateConfig(
    max_round_time_seconds=300,
    jury_size=3,
    enable_safety_monitoring=True,
)

tool = ArtemisCrewTool(
    model="gpt-4o",
    default_rounds=3,
    config=config,
    name="Structured Debate",
    description="Run a structured debate to analyze a topic from multiple perspectives",
)
```

## Tool Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | required | LLM model to use |
| `default_rounds` | int | 3 | Default number of rounds |
| `config` | DebateConfig | None | Debate configuration |
| `name` | str | "Debate Tool" | Tool display name |
| `description` | str | ... | Tool description |

## Using the Tool

### Direct Invocation

```python
tool = ArtemisCrewTool(model="gpt-4o")

result = tool.run(
    topic="Should we adopt Kubernetes?",
    rounds=2,
)

print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']}")
print(f"Reasoning: {result['reasoning']}")
```

### With Custom Positions

```python
result = tool.run(
    topic="Which cloud provider should we use?",
    positions={
        "aws_advocate": "argues for AWS",
        "azure_advocate": "argues for Azure",
        "gcp_advocate": "argues for GCP",
    },
)
```

## Multi-Agent Crews

### Decision Crew

```python
from crewai import Agent, Task, Crew, Process
from artemis.integrations import ArtemisCrewTool

debate_tool = ArtemisCrewTool(model="gpt-4o")

# Research agent gathers information
researcher = Agent(
    role="Research Analyst",
    goal="Gather relevant information for decisions",
    backstory="Expert researcher with access to multiple sources",
    tools=[search_tool, web_scraper_tool],
)

# Debate agent analyzes pros and cons
debater = Agent(
    role="Debate Analyst",
    goal="Analyze options through structured debate",
    backstory="Expert at evaluating arguments from multiple perspectives",
    tools=[debate_tool],
)

# Decision agent makes final call
decider = Agent(
    role="Decision Maker",
    goal="Make informed decisions based on analysis",
    backstory="Experienced executive who makes data-driven decisions",
)

# Tasks
research_task = Task(
    description="Research the implications of {topic}",
    agent=researcher,
    expected_output="Comprehensive research findings",
)

debate_task = Task(
    description="Run a structured debate on {topic} using the research findings",
    agent=debater,
    expected_output="Debate verdict with reasoning",
    context=[research_task],
)

decision_task = Task(
    description="Make a final decision based on the debate",
    agent=decider,
    expected_output="Clear decision with justification",
    context=[debate_task],
)

# Crew
crew = Crew(
    agents=[researcher, debater, decider],
    tasks=[research_task, debate_task, decision_task],
    process=Process.sequential,
)

result = crew.kickoff(inputs={"topic": "Adopting AI in our workflow"})
```

### Parallel Debates

```python
# Multiple debate agents for different aspects
technical_debater = Agent(
    role="Technical Analyst",
    goal="Debate technical aspects",
    tools=[ArtemisCrewTool(model="gpt-4o")],
)

business_debater = Agent(
    role="Business Analyst",
    goal="Debate business aspects",
    tools=[ArtemisCrewTool(model="gpt-4o")],
)

risk_debater = Agent(
    role="Risk Analyst",
    goal="Debate risk aspects",
    tools=[ArtemisCrewTool(model="gpt-4o")],
)

# Hierarchical crew for synthesis
crew = Crew(
    agents=[technical_debater, business_debater, risk_debater, synthesizer],
    tasks=[tech_task, business_task, risk_task, synthesis_task],
    process=Process.hierarchical,
    manager_llm=ChatOpenAI(model="gpt-4o"),
)
```

## Safety Integration

```python
from artemis.integrations import ArtemisCrewTool
from artemis.safety import CompositeMonitor

tool = ArtemisCrewTool(
    model="gpt-4o",
    safety_monitor=CompositeMonitor.default(),
)

# Safety alerts are included in results
result = tool.run(topic="Sensitive topic")

if result.get("safety_alerts"):
    print("Safety concerns detected:")
    for alert in result["safety_alerts"]:
        print(f"  - {alert['type']}: {alert['details']}")
```

## Output Formatting

The tool returns structured output:

```python
result = tool.run(topic="Your topic")

# Result structure
{
    "verdict": "pro" | "con" | "tie",
    "confidence": 0.0 - 1.0,
    "reasoning": "Explanation of the verdict",
    "transcript": [
        {
            "round": 1,
            "agent": "proponent",
            "argument": "...",
        },
        ...
    ],
    "safety_alerts": [...],
    "metadata": {
        "model": "gpt-4o",
        "rounds": 3,
        "duration_seconds": 45.2,
    },
}
```

## Integration Patterns

### Research → Debate → Action

```python
# 1. Research phase
research_task = Task(
    description="Research {topic}",
    agent=researcher,
)

# 2. Debate phase
debate_task = Task(
    description="Debate the research findings",
    agent=debater,
    context=[research_task],
)

# 3. Action phase
action_task = Task(
    description="Create action plan based on debate",
    agent=planner,
    context=[debate_task],
)
```

### Iterative Refinement

```python
# First debate on broad topic
initial_debate = Task(
    description="Debate: Should we enter market X?",
    agent=debater,
)

# Refine based on initial result
refined_debate = Task(
    description="Debate the specific concerns raised",
    agent=debater,
    context=[initial_debate],
)
```

## Error Handling

```python
from artemis.exceptions import DebateError

try:
    result = tool.run(topic="Your topic")
except DebateError as e:
    print(f"Debate failed: {e}")
    # Fallback logic
```

## Async Support

For async crews:

```python
from artemis.integrations import ArtemisCrewTool

tool = ArtemisCrewTool(model="gpt-4o")

# Async run
result = await tool.arun(topic="Your topic")
```

## Best Practices

1. **Use context**: Pass research findings to debate tasks
2. **Combine with search**: Research before debating
3. **Check safety alerts**: Review alerts for sensitive topics
4. **Handle errors**: Implement fallback logic
5. **Use appropriate rounds**: More rounds for complex topics

## Next Steps

- Learn about [LangChain Integration](langchain.md)
- Explore [LangGraph Integration](langgraph.md)
- Configure [MCP Server](mcp.md)
