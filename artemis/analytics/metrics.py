"""Metrics calculation for ARTEMIS debate analytics."""

from __future__ import annotations

import math
from collections import defaultdict
from functools import cached_property
from typing import TYPE_CHECKING, Any

from artemis.analytics.types import RoundMetrics

if TYPE_CHECKING:
    from artemis.core.types import Turn


class DebateMetricsCalculator:
    """Computes aggregate metrics from debate transcript."""

    def __init__(
        self,
        transcript: list[Turn],
        agents: list[str],
    ) -> None:
        self._transcript = transcript
        self._agents = agents
        self._cache: dict[str, Any] = {}

    def compute_all(self) -> dict[str, Any]:
        """Compute all metrics and return as dict."""
        return {
            "rebuttal_effectiveness": self.rebuttal_effectiveness,
            "evidence_utilization": self.evidence_utilization,
            "argument_diversity_index": self.argument_diversity_index,
            "topic_coverage": self.topic_coverage,
            "round_metrics": self.get_all_round_metrics(),
        }

    @cached_property
    def rebuttal_effectiveness(self) -> dict[str, float]:
        """Compute rebuttal effectiveness per agent."""
        # XXX: formula is avg(own_score_when_rebutting) / avg(all_own_scores)
        agent_rebuttal_scores: dict[str, list[float]] = {a: [] for a in self._agents}
        agent_all_scores: dict[str, list[float]] = {a: [] for a in self._agents}

        for turn in self._transcript:
            if turn.evaluation is None:
                continue

            score = turn.evaluation.total_score
            agent_all_scores[turn.agent].append(score)

            # Check if this turn rebuts something
            if hasattr(turn.argument, "rebuts") and turn.argument.rebuts:
                agent_rebuttal_scores[turn.agent].append(score)

        effectiveness = {}
        for agent in self._agents:
            all_scores = agent_all_scores.get(agent, [])
            rebuttal_scores = agent_rebuttal_scores.get(agent, [])

            if not all_scores:
                effectiveness[agent] = 0.0
                continue

            avg_all = sum(all_scores) / len(all_scores)
            avg_rebuttal = sum(rebuttal_scores) / len(rebuttal_scores) if rebuttal_scores else avg_all

            # Effectiveness is ratio of rebuttal performance to baseline
            if avg_all > 0:
                effectiveness[agent] = min(1.0, avg_rebuttal / avg_all)
            else:
                effectiveness[agent] = 0.0

        return effectiveness

    @cached_property
    def evidence_utilization(self) -> dict[str, float]:
        """Compute evidence utilization rate per agent."""
        agent_evidence: dict[str, float] = dict.fromkeys(self._agents, 0.0)
        agent_turns: dict[str, int] = dict.fromkeys(self._agents, 0)

        for turn in self._transcript:
            if not turn.argument:
                continue

            agent_turns[turn.agent] = agent_turns.get(turn.agent, 0) + 1

            if not turn.argument.evidence:
                continue

            # Weight evidence by confidence and verification
            for evidence in turn.argument.evidence:
                weight = evidence.confidence if hasattr(evidence, "confidence") else 0.5

                # Bonus for verified evidence
                if hasattr(evidence, "verified") and evidence.verified:
                    weight *= 1.5

                agent_evidence[turn.agent] = agent_evidence.get(turn.agent, 0.0) + weight

        utilization = {}
        for agent in self._agents:
            turns = agent_turns.get(agent, 0)
            evidence = agent_evidence.get(agent, 0.0)

            if turns > 0:
                # ugh, normalizing this properly is tricky
                utilization[agent] = min(1.0, evidence / (turns * 3))
            else:
                utilization[agent] = 0.0

        return utilization

    @cached_property
    def argument_diversity_index(self) -> dict[str, float]:
        """Compute argument level diversity per agent using entropy."""
        agent_levels: dict[str, dict[str, int]] = {a: {} for a in self._agents}

        for turn in self._transcript:
            if not turn.argument:
                continue

            level = turn.argument.level.value if hasattr(turn.argument.level, "value") else str(turn.argument.level)
            agent = turn.agent

            if agent not in agent_levels:
                agent_levels[agent] = {}

            agent_levels[agent][level] = agent_levels[agent].get(level, 0) + 1

        diversity = {}
        for agent in self._agents:
            levels = agent_levels.get(agent, {})
            if not levels:
                diversity[agent] = 0.0
                continue

            total = sum(levels.values())
            if total == 0:
                diversity[agent] = 0.0
                continue

            # Calculate entropy
            entropy = 0.0
            for count in levels.values():
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)

            # Normalize by max entropy (log2 of number of levels, usually 3)
            max_entropy = math.log2(3)  # strategic, tactical, operational
            diversity[agent] = entropy / max_entropy if max_entropy > 0 else 0.0

        return diversity

    @cached_property
    def topic_coverage(self) -> dict[str, list[str]]:
        """Extract key topics/concepts covered by each agent."""
        agent_topics: dict[str, set[str]] = {a: set() for a in self._agents}

        for turn in self._transcript:
            if not turn.argument:
                continue

            agent = turn.agent

            # Extract from causal links
            if hasattr(turn.argument, "causal_links") and turn.argument.causal_links:
                for link in turn.argument.causal_links:
                    if hasattr(link, "cause"):
                        # Extract key terms (simplified)
                        cause_terms = self._extract_key_terms(link.cause)
                        agent_topics[agent].update(cause_terms)
                    if hasattr(link, "effect"):
                        effect_terms = self._extract_key_terms(link.effect)
                        agent_topics[agent].update(effect_terms)

            # Extract from evidence sources
            if hasattr(turn.argument, "evidence") and turn.argument.evidence:
                for evidence in turn.argument.evidence:
                    if hasattr(evidence, "source") and evidence.source:
                        agent_topics[agent].add(evidence.source[:50])  # Truncate long sources

        return {agent: list(topics) for agent, topics in agent_topics.items()}

    def _extract_key_terms(self, text: str) -> list[str]:
        if not text:
            return []

        # Simple extraction: split on whitespace, filter short words
        words = text.lower().split()
        return [w for w in words if len(w) > 5][:5]  # Top 5 longer words

    def get_round_metrics(self, round_num: int) -> RoundMetrics:
        """Get metrics for a specific round."""
        round_turns = [t for t in self._transcript if t.round == round_num]

        if not round_turns:
            return RoundMetrics(
                round=round_num,
                agent_scores={},
                score_delta={},
                rebuttal_effectiveness={},
                evidence_utilization={},
                argument_level_distribution={},
            )

        # Compute scores
        agent_scores: dict[str, list[float]] = {a: [] for a in self._agents}
        for turn in round_turns:
            if turn.evaluation:
                agent_scores[turn.agent].append(turn.evaluation.total_score)

        avg_scores = {
            agent: sum(scores) / len(scores) if scores else 0.0
            for agent, scores in agent_scores.items()
        }

        # Compute delta from previous round
        prev_metrics = self._get_previous_round_metrics(round_num)
        score_delta = {}
        if prev_metrics:
            for agent in self._agents:
                prev_score = prev_metrics.agent_scores.get(agent, 0.0)
                curr_score = avg_scores.get(agent, 0.0)
                score_delta[agent] = curr_score - prev_score

        # Compute round-specific rebuttal effectiveness
        round_rebuttal = self._compute_round_rebuttal(round_turns)

        # Compute round-specific evidence utilization
        round_evidence = self._compute_round_evidence(round_turns)

        # Compute level distribution
        level_dist = self._compute_level_distribution(round_turns)

        return RoundMetrics(
            round=round_num,
            agent_scores=avg_scores,
            score_delta=score_delta,
            rebuttal_effectiveness=round_rebuttal,
            evidence_utilization=round_evidence,
            argument_level_distribution=level_dist,
        )

    def get_all_round_metrics(self) -> list[RoundMetrics]:
        if not self._transcript:
            return []

        rounds = sorted({t.round for t in self._transcript})
        return [self.get_round_metrics(r) for r in rounds]

    def _get_previous_round_metrics(self, round_num: int) -> RoundMetrics | None:
        cache_key = f"round_{round_num - 1}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prev_turns = [t for t in self._transcript if t.round == round_num - 1]
        if not prev_turns:
            return None

        metrics = self.get_round_metrics(round_num - 1)
        self._cache[cache_key] = metrics
        return metrics

    def _compute_round_rebuttal(self, round_turns: list[Turn]) -> dict[str, float]:
        result = {}
        for agent in self._agents:
            agent_turns = [t for t in round_turns if t.agent == agent]
            if not agent_turns:
                result[agent] = 0.0
                continue

            rebuttal_count = 0
            rebuttal_score_sum = 0.0

            for turn in agent_turns:
                if hasattr(turn.argument, "rebuts") and turn.argument.rebuts:
                    rebuttal_count += 1
                    if turn.evaluation:
                        rebuttal_score_sum += turn.evaluation.total_score

            if rebuttal_count > 0:
                result[agent] = rebuttal_score_sum / rebuttal_count
            else:
                result[agent] = 0.0

        return result

    def _compute_round_evidence(self, round_turns: list[Turn]) -> dict[str, float]:
        result = {}
        for agent in self._agents:
            agent_turns = [t for t in round_turns if t.agent == agent]
            if not agent_turns:
                result[agent] = 0.0
                continue

            total_evidence = 0
            for turn in agent_turns:
                if turn.argument and turn.argument.evidence:
                    total_evidence += len(turn.argument.evidence)

            # Normalize (assuming 2-3 evidence per turn is good)
            result[agent] = min(1.0, total_evidence / (len(agent_turns) * 2.5))

        return result

    def _compute_level_distribution(
        self,
        round_turns: list[Turn],
    ) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}

        for agent in self._agents:
            result[agent] = {}
            agent_turns = [t for t in round_turns if t.agent == agent]

            for turn in agent_turns:
                if turn.argument:
                    level = turn.argument.level.value if hasattr(turn.argument.level, "value") else str(turn.argument.level)
                    result[agent][level] = result[agent].get(level, 0) + 1

        return result


class RebuttalAnalyzer:
    """Analyze rebuttal patterns and effectiveness."""

    def __init__(self, transcript: list[Turn]) -> None:
        self._transcript = transcript

    def analyze_rebuttal_chain(self) -> list[dict]:
        """Trace argument-rebuttal chains through the debate."""
        chains = []

        # Build index of arguments by ID
        arg_index: dict[str, Turn] = {}
        for turn in self._transcript:
            arg_index[turn.id] = turn

        # Find rebuttals
        for turn in self._transcript:
            if not turn.argument:
                continue

            if hasattr(turn.argument, "rebuts") and turn.argument.rebuts:
                for rebutted_id in turn.argument.rebuts:
                    original = arg_index.get(rebutted_id)

                    chains.append({
                        "argument_id": rebutted_id,
                        "rebutted_by": turn.id,
                        "rebutter_agent": turn.agent,
                        "rebutter_score": turn.evaluation.total_score if turn.evaluation else 0.0,
                        "original_agent": original.agent if original else "unknown",
                        "original_score": original.evaluation.total_score if original and original.evaluation else 0.0,
                    })

        return chains

    def compute_rebuttal_success_rate(self, agent: str) -> float:
        """Compute how often agent's rebuttals scored higher than originals."""
        chains = self.analyze_rebuttal_chain()
        agent_rebuttals = [c for c in chains if c["rebutter_agent"] == agent]

        if not agent_rebuttals:
            return 0.0

        successful = sum(
            1 for c in agent_rebuttals
            if c["rebutter_score"] > c["original_score"]
        )

        return successful / len(agent_rebuttals)

    def get_most_rebutted_arguments(self, top_n: int = 5) -> list[dict]:
        """Find arguments that were rebutted most often."""
        rebuttal_counts: dict[str, int] = defaultdict(int)
        arg_agents: dict[str, str] = {}

        for turn in self._transcript:
            arg_agents[turn.id] = turn.agent

            if hasattr(turn.argument, "rebuts") and turn.argument.rebuts:
                for rebutted_id in turn.argument.rebuts:
                    rebuttal_counts[rebutted_id] += 1

        sorted_args = sorted(rebuttal_counts.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "argument_id": arg_id,
                "rebuttal_count": count,
                "agent": arg_agents.get(arg_id, "unknown"),
            }
            for arg_id, count in sorted_args[:top_n]
        ]
