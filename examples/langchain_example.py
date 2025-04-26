#!/usr/bin/env python3
"""
ARTEMIS LangChain Integration Example

This example demonstrates using ARTEMIS debates within LangChain.
Run with: python examples/langchain_example.py

Requirements:
    pip install langchain-core langchain-openai
"""

import asyncio
import os

from artemis.integrations.langchain import ArtemisDebateTool, QuickDebate


async def basic_tool_usage():
    """Basic usage of ArtemisDebateTool."""
    print("\n" + "=" * 60)
    print("Basic ArtemisDebateTool Usage")
    print("=" * 60)

    # Create the debate tool
    tool = ArtemisDebateTool(
        model="gpt-4o",
        default_rounds=3,
    )

    # Run a debate
    result = await tool.ainvoke({
        "topic": "Should artificial intelligence be regulated by governments?",
        "pro_position": "supports government regulation of AI",
        "con_position": "opposes government regulation of AI",
        "rounds": 3,
    })

    print(f"\nTopic: {result.topic}")
    print(f"Verdict: {result.verdict}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"\nReasoning: {result.reasoning}")
    print("\nScores:")
    print(f"  Pro Agent: {result.pro_score:.2f}")
    print(f"  Con Agent: {result.con_score:.2f}")
    print(f"\nTotal Turns: {result.total_turns}")
    print(f"\nSummary: {result.summary}")


async def quick_debate_usage():
    """Quick debate convenience method."""
    print("\n" + "=" * 60)
    print("Quick Debate Usage")
    print("=" * 60)

    # One-liner debate
    result = await QuickDebate.arun(
        topic="Is remote work better than office work?",
        rounds=2,
        model="gpt-4o",
    )

    print(f"\nTopic: {result.topic}")
    print(f"Verdict: {result.verdict} ({result.confidence:.0%} confidence)")


async def langchain_integration():
    """Integration with LangChain agents."""
    print("\n" + "=" * 60)
    print("LangChain Tool Integration")
    print("=" * 60)

    try:
        # Check if langchain_core is available
        import langchain_core.tools  # noqa: F401

        # Create ARTEMIS tool
        artemis_tool = ArtemisDebateTool(model="gpt-4o")

        # Convert to LangChain tool
        langchain_tool = artemis_tool.as_langchain_tool()

        print(f"\nTool Name: {langchain_tool.name}")
        print(f"Description: {langchain_tool.description[:100]}...")
        print("\nThis tool can now be used with LangChain agents!")

        # Example with LangChain agent (commented - requires full setup)
        """
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_tool_calling_agent, AgentExecutor

        llm = ChatOpenAI(model="gpt-4o")
        tools = [langchain_tool]
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools)

        result = await executor.ainvoke({
            "input": "Analyze whether AI should be regulated using a debate format"
        })
        """

    except ImportError:
        print("\nNote: Install langchain-core for full LangChain integration")
        print("  pip install langchain-core")


async def openai_function_format():
    """Export as OpenAI function calling format."""
    print("\n" + "=" * 60)
    print("OpenAI Function Format")
    print("=" * 60)

    tool = ArtemisDebateTool()
    function_def = tool.as_openai_function()

    print("\nOpenAI Function Definition:")
    print(f"  Name: {function_def['name']}")
    print(f"  Description: {function_def['description'][:80]}...")
    print(f"  Parameters: {list(function_def['parameters']['properties'].keys())}")


async def main():
    """Run all LangChain examples."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        print("\nExample: export OPENAI_API_KEY='your-api-key'")
        return

    print("=" * 60)
    print("ARTEMIS Agents - LangChain Integration Examples")
    print("=" * 60)

    # Show function format (doesn't require API calls)
    await openai_function_format()

    # Note: The following examples require API calls
    print("\n" + "-" * 60)
    print("The following examples require API calls.")
    print("Uncomment them to run actual debates.")
    print("-" * 60)

    # Uncomment to run actual debates:
    # await basic_tool_usage()
    # await quick_debate_usage()
    # await langchain_integration()

    print("\n[Examples configured - uncomment sections to run full debates]")


if __name__ == "__main__":
    asyncio.run(main())
