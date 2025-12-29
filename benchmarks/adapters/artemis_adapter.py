"""ARTEMIS framework adapter for benchmarks."""

import time

from benchmarks.adapters.base import DebateAdapter, DebateResult


class ArtemisAdapter(DebateAdapter):
    """Adapter for ARTEMIS debate framework."""

    name = "artemis"

    def is_available(self) -> bool:
        """Check if ARTEMIS is available."""
        try:
            from artemis.core.agent import Agent  # noqa: F401
            from artemis.core.debate import Debate  # noqa: F401

            return True
        except ImportError:
            return False

    async def run_debate(
        self,
        topic: str,
        pro_position: str,
        con_position: str,
    ) -> DebateResult:
        """Run a debate using ARTEMIS."""
        from artemis.core.agent import Agent
        from artemis.core.debate import Debate
        from artemis.models import OpenAIModel

        start_time = time.time()
        tokens_used = 0
        transcript = []

        try:
            # Create model
            model = OpenAIModel(model_name=self.model)

            # Create agents with LLM extraction for QUALITY mode
            extraction_model = "gpt-4o-mini"  # Fast model for extraction

            pro_agent = Agent(
                name="pro",
                role=pro_position,
                model=model,
                extraction_model=extraction_model,
            )
            con_agent = Agent(
                name="con",
                role=con_position,
                model=model,
                extraction_model=extraction_model,
            )

            # Create config with QUALITY mode for benchmarking
            from artemis.core.types import DebateConfig, EvaluationMode

            config = DebateConfig(
                evaluation_mode=EvaluationMode.QUALITY,
            )

            # Create and run debate
            debate = Debate(
                topic=topic,
                agents=[pro_agent, con_agent],
                rounds=self.rounds,
                config=config,
            )

            debate.assign_positions({
                "pro": pro_position,
                "con": con_position,
            })

            result = await debate.run()

            # Extract transcript
            for turn in result.transcript:
                transcript.append({
                    "agent": turn.agent,
                    "content": turn.argument.content,
                    "level": turn.argument.level.value if turn.argument.level else "unknown",
                    "evidence_count": len(turn.argument.evidence) if turn.argument.evidence else 0,
                    "causal_links_count": len(turn.argument.causal_links) if turn.argument.causal_links else 0,
                })

            # Extract verdict
            verdict = None
            verdict_reasoning = None
            confidence = None

            if result.verdict:
                verdict = result.verdict.decision
                verdict_reasoning = result.verdict.reasoning
                confidence = result.verdict.confidence

            elapsed = time.time() - start_time

            return DebateResult(
                framework=self.name,
                topic=topic,
                transcript=transcript,
                verdict=verdict,
                verdict_reasoning=verdict_reasoning,
                confidence=confidence,
                metadata={
                    "model": self.model,
                    "extraction_model": extraction_model,
                    "rounds": self.rounds,
                    "evaluation_mode": "quality",
                    "features": [
                        "H-L-DAG",
                        "L-AE-CR",
                        "jury_verdict",
                        "llm_evidence_extraction",
                        "llm_causal_extraction",
                        "causal_graph",
                        "llm_evaluation",
                        "closed_loop_feedback",
                        "adaptive_level_selection",
                        "perspective_weighting",
                    ],
                },
                tokens_used=tokens_used,
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
