"""CrewAI framework adapter for benchmarks."""

import time

from benchmarks.adapters.base import DebateAdapter, DebateResult


class CrewAIAdapter(DebateAdapter):
    """Adapter for CrewAI framework."""

    name = "crewai"

    def is_available(self) -> bool:
        """Check if CrewAI is available."""
        try:
            from crewai import Agent, Task, Crew
            return True
        except ImportError:
            return False

    async def run_debate(
        self,
        topic: str,
        pro_position: str,
        con_position: str,
    ) -> DebateResult:
        """Run a debate using CrewAI."""
        import asyncio

        start_time = time.time()
        transcript = []

        try:
            from crewai import Agent, Task, Crew, Process

            # Create pro agent
            pro_agent = Agent(
                role="Proponent",
                goal=f"Argue convincingly FOR: {pro_position}",
                backstory=f"""You are an expert debater tasked with arguing FOR the position: {pro_position}

You must present compelling arguments, use evidence, and address counterarguments.""",
                verbose=False,
                allow_delegation=False,
            )

            # Create con agent
            con_agent = Agent(
                role="Opponent",
                goal=f"Argue convincingly AGAINST: {con_position}",
                backstory=f"""You are an expert debater tasked with arguing AGAINST the position: {con_position}

You must present compelling arguments, use evidence, and address counterarguments.""",
                verbose=False,
                allow_delegation=False,
            )

            # Create debate tasks for each round
            tasks = []
            for round_num in range(1, self.rounds + 1):
                # Pro argument
                pro_task = Task(
                    description=f"""Round {round_num} - Present your argument FOR: {topic}

{'Present your opening argument.' if round_num == 1 else 'Respond to the opponent and strengthen your position.'}

Be specific, use evidence, and address counterarguments.""",
                    agent=pro_agent,
                    expected_output="A well-structured argument supporting the position",
                )
                tasks.append(pro_task)

                # Con argument
                con_task = Task(
                    description=f"""Round {round_num} - Present your argument AGAINST: {topic}

{'Present your opening argument.' if round_num == 1 else 'Respond to the proponent and strengthen your position.'}

Be specific, use evidence, and address counterarguments.""",
                    agent=con_agent,
                    expected_output="A well-structured argument opposing the position",
                )
                tasks.append(con_task)

            # Create crew
            crew = Crew(
                agents=[pro_agent, con_agent],
                tasks=tasks,
                process=Process.sequential,
                verbose=False,
            )

            # Run crew (blocking call, run in executor)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, crew.kickoff)

            # Extract transcript from task outputs
            for i, task in enumerate(tasks):
                agent_name = "pro" if i % 2 == 0 else "con"
                output = task.output.raw if hasattr(task.output, 'raw') else str(task.output)
                transcript.append({
                    "agent": agent_name,
                    "content": output,
                    "round": (i // 2) + 1,
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
                    "features": [
                        "agent_roles",
                        "sequential_tasks",
                        "crew_orchestration",
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
