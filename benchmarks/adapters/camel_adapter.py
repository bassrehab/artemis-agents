"""CAMEL framework adapter for benchmarks."""

import time

from benchmarks.adapters.base import DebateAdapter, DebateResult


class CAMELAdapter(DebateAdapter):
    """Adapter for CAMEL (Communicative Agents for Mind Exploration) framework."""

    name = "camel"

    def is_available(self) -> bool:
        """Check if CAMEL is available."""
        try:
            from camel.societies import RolePlaying
            from camel.types import ModelType
            return True
        except ImportError:
            return False

    async def run_debate(
        self,
        topic: str,
        pro_position: str,
        con_position: str,
    ) -> DebateResult:
        """Run a debate using CAMEL."""
        import asyncio

        start_time = time.time()
        transcript = []

        try:
            from camel.societies import RolePlaying
            from camel.types import ModelType, TaskType
            from camel.models import ModelFactory
            from camel.configs import ChatGPTConfig

            # Map model name to CAMEL ModelType
            model_type = ModelType.GPT_4O if "gpt-4o" in self.model else ModelType.GPT_4_TURBO

            # Create role-playing session
            task_prompt = f"""Debate Topic: {topic}

The proponent argues FOR: {pro_position}
The opponent argues AGAINST: {con_position}

Conduct a structured debate with {self.rounds} rounds of arguments from each side."""

            role_play_session = RolePlaying(
                assistant_role_name="Proponent",
                assistant_agent_kwargs=dict(
                    model=ModelFactory.create(
                        model_platform="openai",
                        model_type=model_type,
                        model_config_dict=ChatGPTConfig(temperature=0.7).as_dict(),
                    ),
                ),
                user_role_name="Opponent",
                user_agent_kwargs=dict(
                    model=ModelFactory.create(
                        model_platform="openai",
                        model_type=model_type,
                        model_config_dict=ChatGPTConfig(temperature=0.7).as_dict(),
                    ),
                ),
                task_prompt=task_prompt,
                with_task_specify=False,
            )

            # Run conversation rounds
            loop = asyncio.get_event_loop()

            # Get initial messages
            input_msg = role_play_session.init_chat()

            # Run for specified number of exchanges
            max_turns = self.rounds * 2
            turn_count = 0

            while turn_count < max_turns:
                # Step the conversation
                assistant_response, user_response = await loop.run_in_executor(
                    None,
                    lambda: role_play_session.step(input_msg)
                )

                # Extract assistant (pro) response
                if assistant_response.msgs:
                    for msg in assistant_response.msgs:
                        transcript.append({
                            "agent": "pro",
                            "content": msg.content,
                        })
                        turn_count += 1

                # Extract user (con) response
                if user_response.msgs:
                    for msg in user_response.msgs:
                        transcript.append({
                            "agent": "con",
                            "content": msg.content,
                        })
                        turn_count += 1

                # Check for termination
                if assistant_response.terminated or user_response.terminated:
                    break

                # Prepare next input
                if user_response.msgs:
                    input_msg = user_response.msgs[-1]
                else:
                    break

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
                    "features": [
                        "role_playing",
                        "communicative_agents",
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
