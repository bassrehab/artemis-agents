# Google ADK Integration via MCP

ARTEMIS integrates with Google Agent Development Kit (ADK) through its MCP server. This allows ADK agents to use structured debates as tools without requiring a native integration.

## Prerequisites

```bash
# Install Google ADK
pip install google-adk

# Install ARTEMIS with MCP support
pip install artemis-agents
```

## Basic Integration

ADK can consume ARTEMIS debates via the MCP toolset:

```python
import asyncio
from google.adk.agents import Agent
from google.adk.tools.mcp_tool import MCPToolset, StdioServerParameters

async def create_decision_agent():
    # Connect to ARTEMIS MCP server
    artemis_tools, exit_stack = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command="artemis-mcp",
            args=["--model", "gpt-4o"],
        )
    )

    # Create an ADK agent with ARTEMIS debate capabilities
    agent = Agent(
        name="decision_analyst",
        model="gemini-2.0-flash",
        instruction="""
        You are a decision analyst. When faced with complex decisions or
        controversial topics, use the ARTEMIS debate tools to analyze
        multiple perspectives before making recommendations.

        Available debate tools:
        - artemis_debate_start: Start a structured debate on a topic
        - artemis_get_verdict: Get the jury's verdict after debate
        - artemis_get_transcript: Review the full debate transcript
        """,
        tools=[artemis_tools],
    )

    return agent, exit_stack

async def main():
    agent, exit_stack = await create_decision_agent()

    async with exit_stack:
        # The agent can now use ARTEMIS debates
        response = await agent.run(
            "Should our company adopt a four-day work week? "
            "Run a debate to analyze this decision."
        )
        print(response)

asyncio.run(main())
```

## Multi-Agent Decision System

Use ARTEMIS debates within an ADK multi-agent hierarchy:

```python
import asyncio
from google.adk.agents import Agent
from google.adk.tools.mcp_tool import MCPToolset, StdioServerParameters

async def create_decision_system():
    # Connect to ARTEMIS
    artemis_tools, exit_stack = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command="artemis-mcp",
            args=["--model", "gpt-4o"],
        )
    )

    # Specialist agent for running debates
    debate_agent = Agent(
        name="debate_specialist",
        model="gemini-2.0-flash",
        instruction="""
        You are a debate specialist. When asked to analyze a topic:
        1. Use artemis_debate_start to begin a structured debate
        2. Wait for the debate to complete
        3. Use artemis_get_verdict to get the final verdict
        4. Summarize the key arguments from both sides
        """,
        tools=[artemis_tools],
    )

    # Research agent for gathering context
    research_agent = Agent(
        name="researcher",
        model="gemini-2.0-flash",
        instruction="""
        You are a research specialist. Gather relevant information
        and context about topics before they are debated.
        """,
        tools=[google_search],  # ADK's built-in search
    )

    # Coordinator agent
    coordinator = Agent(
        name="decision_coordinator",
        model="gemini-2.0-flash",
        instruction="""
        You coordinate decision-making by:
        1. Having the researcher gather context
        2. Having the debate specialist run a structured debate
        3. Synthesizing findings into a recommendation
        """,
        sub_agents=[research_agent, debate_agent],
    )

    return coordinator, exit_stack

async def main():
    coordinator, exit_stack = await create_decision_system()

    async with exit_stack:
        response = await coordinator.run(
            "We need to decide whether to migrate our monolith to microservices. "
            "Research the topic and run a structured debate to help us decide."
        )
        print(response)

asyncio.run(main())
```

## Using Specific Debate Tools

ADK agents can call individual ARTEMIS tools:

```python
import asyncio
from google.adk.agents import Agent
from google.adk.tools.mcp_tool import MCPToolset, StdioServerParameters

async def run_controlled_debate():
    artemis_tools, exit_stack = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command="artemis-mcp",
        )
    )

    agent = Agent(
        name="analyst",
        model="gemini-2.0-flash",
        instruction="""
        You help users run structured debates. Guide them through:

        1. Starting a debate with artemis_debate_start
           - Specify topic, rounds, and optionally custom positions

        2. Monitoring progress with artemis_get_transcript
           - Review arguments as they develop

        3. Getting results with artemis_get_verdict
           - Retrieve the final jury decision

        4. Checking safety with artemis_get_safety_report
           - Review any safety alerts from the debate

        Always explain what each tool does before using it.
        """,
        tools=[artemis_tools],
    )

    async with exit_stack:
        # Interactive session
        response = await agent.run(
            "Start a 2-round debate on whether AI should be used in hiring decisions. "
            "Use custom positions: one focusing on efficiency, one on fairness."
        )
        print(response)

asyncio.run(run_controlled_debate())
```

## With Human-in-the-Loop

ADK's tool confirmation flow works with ARTEMIS:

```python
import asyncio
from google.adk.agents import Agent
from google.adk.tools.mcp_tool import MCPToolset, StdioServerParameters
from google.adk.tools import ToolConfirmation

async def run_with_confirmation():
    artemis_tools, exit_stack = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command="artemis-mcp",
        )
    )

    # Wrap tools with confirmation requirement
    confirmed_tools = ToolConfirmation(
        tools=artemis_tools,
        require_confirmation=True,
        confirmation_message="About to run a debate. Proceed?",
    )

    agent = Agent(
        name="careful_analyst",
        model="gemini-2.0-flash",
        instruction="You run debates but always confirm with the user first.",
        tools=[confirmed_tools],
    )

    async with exit_stack:
        response = await agent.run(
            "Debate whether we should open-source our internal tools."
        )
        print(response)

asyncio.run(run_with_confirmation())
```

## Deployment on Vertex AI

Deploy an ADK agent with ARTEMIS to Vertex AI:

```python
from google.adk.agents import Agent
from google.adk.tools.mcp_tool import MCPToolset, StdioServerParameters

# Define agent configuration
agent_config = {
    "name": "enterprise_decision_maker",
    "model": "gemini-2.0-flash",
    "instruction": """
    Enterprise decision support agent with structured debate capabilities.
    Use ARTEMIS debates for complex decisions requiring multiple perspectives.
    """,
    "tools": [
        {
            "type": "mcp",
            "command": "artemis-mcp",
            "args": ["--model", "gpt-4o"],
        }
    ],
}

# Deploy to Vertex AI Agent Engine
# See ADK docs for deployment details
```

## Available MCP Tools

When connected to ARTEMIS via MCP, these tools are available:

| Tool | Description |
|------|-------------|
| `artemis_debate_start` | Start a new structured debate |
| `artemis_add_round` | Add a round to existing debate |
| `artemis_get_verdict` | Get the jury's final verdict |
| `artemis_get_transcript` | Get full debate transcript |
| `artemis_list_debates` | List all active debates |
| `artemis_get_safety_report` | Get safety monitoring report |

## Why MCP Integration?

Using MCP rather than a native ADK tool provides:

1. **Zero maintenance**: No custom ADK code to update
2. **Always current**: MCP server reflects latest ARTEMIS features
3. **Flexibility**: Same MCP server works with any MCP-compatible client
4. **Separation**: ARTEMIS runs as separate process, isolating dependencies

## Next Steps

- See [MCP Server](../integrations/mcp.md) for server configuration
- Learn about [Basic Debates](basic-debate.md) to understand ARTEMIS
- Explore [Safety Monitors](safety-monitors.md) for safety features
