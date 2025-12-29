"""L-AE-CR Adaptive Evaluation with Causal Reasoning."""

from dataclasses import dataclass, field
from enum import Enum

from artemis.core.causal import CausalExtractor, CausalGraph, LinkType
from artemis.core.evidence import EvidenceExtractor
from artemis.core.types import (
    Argument,
    ArgumentEvaluation,
    ArgumentLevel,
    CausalGraphUpdate,
    CriterionScore,
    DebateContext,
    EvaluationCriteria,
)
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationDimension(str, Enum):
    """Dimensions used for argument evaluation."""

    LOGICAL_COHERENCE = "logical_coherence"
    EVIDENCE_QUALITY = "evidence_quality"
    CAUSAL_REASONING = "causal_reasoning"
    ETHICAL_ALIGNMENT = "ethical_alignment"
    PERSUASIVENESS = "persuasiveness"


@dataclass
class AdaptationConfig:
    adaptation_rate: float = 0.1
    sensitivity_threshold: float = 0.7
    complexity_threshold: float = 0.7
    late_round_factor: float = 0.3
    ethical_boost: float = 1.5
    causal_boost: float = 1.3


@dataclass
class TopicAnalysis:
    """Analysis of debate topic characteristics."""

    sensitivity: float = 0.5  # 0-1
    complexity: float = 0.5
    controversy: float = 0.5

    # Class-level constants for keyword analysis
    _SENSITIVITY_KEYWORDS: list[str] = field(default_factory=list, init=False, repr=False)
    _COMPLEXITY_KEYWORDS: list[str] = field(default_factory=list, init=False, repr=False)
    _CONTROVERSY_KEYWORDS: list[str] = field(default_factory=list, init=False, repr=False)

    @staticmethod
    def _get_sensitivity_keywords():
        return [
            "ethics",
            "moral",
            "life",
            "death",
            "rights",
            "freedom",
            "privacy",
            "discrimination",
            "bias",
            "harm",
            "safety",
            "vulnerable",
            "children",
            "health",
            "religion",
        ]

    @staticmethod
    def _get_complexity_keywords():
        return [
            "technical",
            "scientific",
            "algorithm",
            "system",
            "infrastructure",
            "economic",
            "policy",
            "regulation",
            "international",
            "mechanism",
            "framework",
            "protocol",
        ]

    @staticmethod
    def _get_controversy_keywords():
        return [
            "controversial",
            "debate",
            "disagree",
            "conflict",
            "disputed",
            "polarizing",
            "divisive",
            "contentious",
        ]

    @classmethod
    def analyze(cls, topic: str) -> "TopicAnalysis":
        """Analyze a topic string for characteristics."""
        topic_lower = topic.lower()
        words = topic_lower.split()

        # Calculate sensitivity
        sensitivity_count = sum(1 for kw in cls._get_sensitivity_keywords() if kw in topic_lower)
        sensitivity = min(1.0, sensitivity_count * 0.15)

        # Calculate complexity
        complexity_count = sum(1 for kw in cls._get_complexity_keywords() if kw in topic_lower)
        complexity = min(1.0, complexity_count * 0.15)

        # Also consider topic length as complexity indicator
        if len(words) > 15:
            complexity = min(1.0, complexity + 0.2)

        # Calculate controversy
        controversy_count = sum(1 for kw in cls._get_controversy_keywords() if kw in topic_lower)
        controversy = min(1.0, controversy_count * 0.2)

        return cls(
            sensitivity=sensitivity,
            complexity=complexity,
            controversy=controversy,
        )


class CriterionEvaluator:
    """Evaluates individual criteria for an argument."""

    def __init__(self):
        self._evidence_extractor = EvidenceExtractor()
        self._causal_extractor = CausalExtractor()

    def evaluate_logical_coherence(self, argument):
        """Check thesis, logical flow, contradictions, conclusion."""
        content = argument.content.lower()
        score = 0.5  # Base score

        # Check for thesis indicators
        thesis_indicators = ["therefore", "thus", "conclude", "argue that", "position is"]
        for indicator in thesis_indicators:
            if indicator in content:
                score += 0.1
                break

        # Check for logical connectors
        logical_connectors = [
            "because",
            "therefore",
            "however",
            "furthermore",
            "moreover",
            "consequently",
            "hence",
            "thus",
        ]
        connector_count = sum(1 for conn in logical_connectors if conn in content)
        score += min(0.2, connector_count * 0.05)

        # Check for structured argument (numbered points or clear sections)
        if any(f"{i}." in content or f"{i})" in content for i in range(1, 6)):
            score += 0.1

        # Penalize potential contradictions (simplified check)
        contradiction_pairs = [
            ("always", "never"),
            ("all", "none"),
            ("must", "cannot"),
        ]
        for pos, neg in contradiction_pairs:
            if pos in content and neg in content:
                score -= 0.15

        # Strategic level arguments should score higher for having overview structure
        if argument.level == ArgumentLevel.STRATEGIC and any(
            word in content for word in ["first", "second", "third", "finally"]
        ):
            score += 0.1

        return max(0.0, min(1.0, score))

    def evaluate_evidence_quality(self, argument):
        """Check citations, stats, expert refs, credibility."""
        score = 0.3  # Base score

        # Check existing evidence in argument
        if argument.evidence:
            score += min(0.3, len(argument.evidence) * 0.1)

            # Bonus for verified evidence
            verified_count = sum(1 for ev in argument.evidence if ev.verified)
            score += verified_count * 0.1

        # Extract additional evidence from content
        extracted = self._evidence_extractor.extract(argument.content)
        score += min(0.2, len(extracted) * 0.05)

        # Operational level should have more specific evidence
        if argument.level == ArgumentLevel.OPERATIONAL and (
            len(argument.evidence) >= 2 or len(extracted) >= 2
        ):
            score += 0.1

        return max(0.0, min(1.0, score))

    def evaluate_causal_reasoning(self, argument, causal_graph=None):
        """Check cause-effect relationships, chain validity, claim strength."""
        score = 0.3  # Base score

        # Check existing causal links
        if argument.causal_links:
            score += min(0.3, len(argument.causal_links) * 0.1)

            # Check link strength
            avg_strength = sum(link.strength for link in argument.causal_links) / len(
                argument.causal_links
            )
            score += avg_strength * 0.2

        # Extract causal relationships from content
        extracted = self._causal_extractor.extract(argument.content)
        score += min(0.2, len(extracted) * 0.05)

        # Check if causal claims connect to existing graph
        if causal_graph and len(causal_graph) > 0:
            for link in argument.causal_links:
                # Bonus for connecting to established causes/effects
                if causal_graph.get_node(link.cause):
                    score += 0.05
                if causal_graph.get_node(link.effect):
                    score += 0.05

        return max(0.0, min(1.0, score))

    def evaluate_ethical_alignment(self, argument):
        content = argument.content.lower()
        score = 0.5  # Base neutral score

        # Check for ethical considerations
        ethical_positives = [
            "ethical",
            "moral",
            "fair",
            "just",
            "rights",
            "welfare",
            "benefit",
            "stakeholder",
            "responsible",
            "sustainable",
            "equitable",
        ]
        for term in ethical_positives:
            if term in content:
                score += 0.05

        # Check for harm awareness
        harm_awareness = [
            "harm",
            "risk",
            "danger",
            "concern",
            "careful",
            "protect",
            "safeguard",
        ]
        for term in harm_awareness:
            if term in content:
                score += 0.03

        # Use existing ethical score if available
        if argument.ethical_score is not None:
            # Blend with computed score
            score = (score + argument.ethical_score) / 2

        return max(0.0, min(1.0, score))

    def evaluate_persuasiveness(self, argument):
        # rhetoric, emotional appeal, clarity, engagement
        content = argument.content.lower()
        score = 0.4  # Base score

        # Check for rhetorical questions
        if "?" in argument.content:
            score += 0.05

        # Check for call to action or emphasis
        emphasis_words = [
            "must",
            "should",
            "essential",
            "critical",
            "crucial",
            "important",
            "clear",
            "obvious",
        ]
        for word in emphasis_words:
            if word in content:
                score += 0.03

        # Check for audience engagement
        engagement_terms = ["we", "our", "consider", "imagine", "ask yourself"]
        for term in engagement_terms:
            if term in content:
                score += 0.03

        # Check content length (too short = less persuasive)
        word_count = len(content.split())
        if word_count > 100:
            score += 0.1
        elif word_count > 50:
            score += 0.05

        # Strategic arguments should be more persuasive
        if argument.level == ArgumentLevel.STRATEGIC:
            score += 0.05

        return max(0.0, min(1.0, score))


class AdaptiveEvaluator:
    """L-AE-CR: Adaptive Evaluation with Causal Reasoning."""

    DEFAULT_CRITERIA = EvaluationCriteria()

    def __init__(
        self,
        criteria: EvaluationCriteria | None = None,
        adaptation_config: AdaptationConfig | None = None,
    ):
        self.criteria = criteria or self.DEFAULT_CRITERIA
        self.config = adaptation_config or AdaptationConfig()
        self.causal_graph = CausalGraph()
        self._criterion_evaluator = CriterionEvaluator()
        self._topic_analysis: TopicAnalysis | None = None

        logger.debug(
            "AdaptiveEvaluator initialized",
            criteria=self.criteria.to_dict(),
            adaptation_rate=self.config.adaptation_rate,
        )

    def analyze_topic(self, topic: str):
        """Analyze topic for context-aware evaluation."""
        self._topic_analysis = TopicAnalysis.analyze(topic)
        logger.debug(
            "Topic analyzed",
            sensitivity=self._topic_analysis.sensitivity,
            complexity=self._topic_analysis.complexity,
            controversy=self._topic_analysis.controversy,
        )
        return self._topic_analysis

    def adapt_criteria(self, context: DebateContext):
        """Dynamically adjust criteria weights based on context."""
        # Get base weights
        adapted = self.criteria.to_dict()

        # Analyze topic if not already done
        if self._topic_analysis is None:
            self._topic_analysis = TopicAnalysis.analyze(context.topic)

        # Increase ethical weight for sensitive topics
        if self._topic_analysis.sensitivity > self.config.sensitivity_threshold:
            adapted["ethical_alignment"] *= self.config.ethical_boost

        # Increase evidence weight in later rounds
        if context.total_rounds > 0:
            round_progress = context.current_round / context.total_rounds
            round_factor = 1 + round_progress * self.config.late_round_factor
            adapted["evidence_quality"] *= round_factor

        # Increase causal reasoning for complex topics
        if self._topic_analysis.complexity > self.config.complexity_threshold:
            adapted["causal_reasoning"] *= self.config.causal_boost

        # Increase persuasiveness for controversial topics
        if self._topic_analysis.controversy > 0.6:
            adapted["persuasiveness"] *= 1.2

        # Renormalize weights to sum to 1.0
        total = sum(adapted.values())
        normalized = {k: v / total for k, v in adapted.items()}

        logger.debug(
            "Criteria adapted",
            original=self.criteria.to_dict(),
            adapted=normalized,
            round=context.current_round,
        )

        return normalized

    async def evaluate_argument(
        self,
        argument: Argument,
        context: DebateContext,
    ) -> ArgumentEvaluation:
        """Evaluate an argument using adaptive criteria."""
        logger.info(
            "Evaluating argument",
            argument_id=argument.id,
            agent=argument.agent,
            level=argument.level.value,
        )

        # Adapt criteria based on context
        adapted_weights = self.adapt_criteria(context)

        # Score each criterion
        scores: dict[str, float] = {}
        criterion_details: list[CriterionScore] = []

        # Logical coherence
        scores["logical_coherence"] = self._criterion_evaluator.evaluate_logical_coherence(argument)
        criterion_details.append(
            CriterionScore(
                criterion="logical_coherence",
                score=scores["logical_coherence"],
                weight=adapted_weights["logical_coherence"],
            )
        )

        # Evidence quality
        scores["evidence_quality"] = self._criterion_evaluator.evaluate_evidence_quality(argument)
        criterion_details.append(
            CriterionScore(
                criterion="evidence_quality",
                score=scores["evidence_quality"],
                weight=adapted_weights["evidence_quality"],
            )
        )

        # Causal reasoning
        scores["causal_reasoning"] = self._criterion_evaluator.evaluate_causal_reasoning(
            argument, self.causal_graph
        )
        criterion_details.append(
            CriterionScore(
                criterion="causal_reasoning",
                score=scores["causal_reasoning"],
                weight=adapted_weights["causal_reasoning"],
            )
        )

        # Ethical alignment
        scores["ethical_alignment"] = self._criterion_evaluator.evaluate_ethical_alignment(argument)
        criterion_details.append(
            CriterionScore(
                criterion="ethical_alignment",
                score=scores["ethical_alignment"],
                weight=adapted_weights["ethical_alignment"],
            )
        )

        # Persuasiveness
        scores["persuasiveness"] = self._criterion_evaluator.evaluate_persuasiveness(argument)
        criterion_details.append(
            CriterionScore(
                criterion="persuasiveness",
                score=scores["persuasiveness"],
                weight=adapted_weights["persuasiveness"],
            )
        )

        # Compute weighted total
        total_score = sum(scores[c] * adapted_weights[c] for c in scores)

        # Update causal graph and get updates
        causal_update = self._update_causal_graph(argument)

        evaluation = ArgumentEvaluation(
            argument_id=argument.id,
            scores=scores,
            weights=adapted_weights,
            criterion_details=criterion_details,
            causal_score=scores["causal_reasoning"],
            total_score=total_score,
            causal_graph_update=causal_update,
        )

        logger.info(
            "Argument evaluated",
            argument_id=argument.id,
            total_score=total_score,
            scores=scores,
        )

        return evaluation

    def _update_causal_graph(self, argument):
        added_links = []
        strengthened_links: list[str] = []

        for link in argument.causal_links:
            # Check if link already exists
            existing_edge = self.causal_graph.get_edge(link.cause, link.effect)

            if existing_edge:
                # Strengthen existing link
                strengthened_links.append(f"{link.cause}->{link.effect}")
            else:
                # Add new link
                added_links.append(link)

            # Add to graph
            self.causal_graph.add_link(link, LinkType.CAUSES, argument_id=argument.id)

        return CausalGraphUpdate(
            added_links=added_links,
            strengthened_links=strengthened_links,
            weakened_links=[],  # Would be populated by counter-arguments
        )

    def get_criteria_weights(self) -> dict[str, float]:
        """Get current criteria weights."""
        return self.criteria.to_dict()

    def set_criteria_weights(self, weights):
        self.criteria = EvaluationCriteria(**weights)

    def reset_causal_graph(self) -> None:
        """Reset the causal graph to empty state."""
        self.causal_graph = CausalGraph()
        logger.debug("Causal graph reset")


class RoundEvaluator:
    """Evaluates all arguments in a debate round."""

    def __init__(self, evaluator=None):
        self.evaluator = evaluator or AdaptiveEvaluator()

    async def evaluate_round(self, arguments, context):
        """Evaluate all arguments in a round."""
        evaluations = []

        for argument in arguments:
            evaluation = await self.evaluator.evaluate_argument(argument, context)
            evaluations.append(evaluation)

        return evaluations

    def get_round_summary(self, evaluations):
        if not evaluations:
            return {"average_score": 0.0, "max_score": 0.0, "min_score": 0.0}

        scores = [e.total_score for e in evaluations]
        return {
            "average_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "score_spread": max(scores) - min(scores),
        }

    def compare_agents(self, evaluations, arguments):
        """Compare evaluation scores by agent."""
        agent_scores = {}

        for eval_result, argument in zip(evaluations, arguments, strict=True):
            agent = argument.agent
            if agent not in agent_scores:
                agent_scores[agent] = []
            agent_scores[agent].append(eval_result.total_score)

        return {
            agent: {
                "average": sum(scores) / len(scores),
                "total_arguments": len(scores),
                "max": max(scores),
                "min": min(scores),
            }
            for agent, scores in agent_scores.items()
        }
