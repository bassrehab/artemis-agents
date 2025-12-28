"""AutoGen framework adapter for benchmarks."""

import os
import time

from benchmarks.adapters.base import DebateAdapter, DebateResult


class AutoGenAdapter(DebateAdapter):
    """Adapter for Microsoft AutoGen framework (v0.7+)."""

    name = "autogen"

    def is_available(self) -> bool:
        """Check if AutoGen is available."""
        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.teams import RoundRobinGroupChat
            from autogen_ext.models.openai import OpenAIChatCompletionClient

            return True
        except ImportError:
            return False

    async def run_debate(
        self,
        topic: str,
        pro_position: str,
        con_position: str,
    ) -> DebateResult:
        """Run a debate using AutoGen."""
        start_time = time.time()
        transcript = []

        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.base import TaskResult
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.teams import RoundRobinGroupChat
            from autogen_ext.models.openai import OpenAIChatCompletionClient

            # Create model client
            model_client = OpenAIChatCompletionClient(
                model=self.model,
                api_key=os.environ.get("OPENAI_API_KEY"),
            )

            # Create pro agent
            pro_agent = AssistantAgent(
                name="pro",
                model_client=model_client,
                system_message=f"""You are a debate participant arguing FOR the following position: {pro_position}

Topic: {topic}

Your goal is to present compelling arguments supporting your position. Use evidence, logical reasoning, and address counterarguments. Be persuasive but fair. Keep responses focused and under 300 words.""",
            )

            # Create con agent
            con_agent = AssistantAgent(
                name="con",
                model_client=model_client,
                system_message=f"""You are a debate participant arguing AGAINST the following position: {con_position}

Topic: {topic}

Your goal is to present compelling arguments opposing the proposition. Use evidence, logical reasoning, and address counterarguments. Be persuasive but fair. Keep responses focused and under 300 words.""",
            )

            # Create team with round-robin turns
            max_messages = self.rounds * 2 + 1  # +1 for initial message
            termination = MaxMessageTermination(max_messages=max_messages)

            team = RoundRobinGroupChat(
                participants=[pro_agent, con_agent],
                termination_condition=termination,
            )

            # Run debate
            initial_message = f"Let's debate: {topic}\n\nPro will argue FOR: {pro_position}\nCon will argue AGAINST: {con_position}\n\nPro, please begin with your opening argument."

            result: TaskResult = await team.run(task=initial_message)

            # Extract transcript from messages
            for msg in result.messages:
                if hasattr(msg, "content") and hasattr(msg, "source"):
                    content = msg.content
                    if isinstance(content, str) and content.strip():
                        transcript.append({
                            "agent": msg.source,
                            "content": content,
                        })

            elapsed = time.time() - start_time

            return DebateResult(
                framework=self.name,
                topic=topic,
                transcript=transcript,
                verdict=None,
                verdict_reasoning=None,
                confidence=None,
                metadata={
                    "model": self.model,
                    "rounds": self.rounds,
                    "max_messages": max_messages,
                    "features": [
                        "assistant_agents",
                        "round_robin_chat",
                        "autogen_v0.7",
                    ],
                },
                tokens_used=0,
                time_seconds=elapsed,
            )

        except Exception as e:
            return DebateResult(
                framework=self.name,
                topic=topic,
                transcript=transcript,
                error=str(e),
                time_seconds=time.time() - start_time,
            )
