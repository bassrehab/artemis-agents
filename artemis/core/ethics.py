"""
ARTEMIS Ethics Module

Provides ethical evaluation and alignment scoring for debate arguments.
Implements multi-framework ethical analysis including stakeholder impact assessment.
"""

import re
from dataclasses import dataclass, field
from enum import Enum

from artemis.core.types import Argument, DebateContext
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


class EthicalFramework(str, Enum):
    """Ethical frameworks for evaluation."""

    UTILITARIAN = "utilitarian"
    """Greatest good for the greatest number."""

    DEONTOLOGICAL = "deontological"
    """Rule-based ethics, duties and rights."""

    VIRTUE = "virtue"
    """Character-based ethics, virtuous behavior."""

    CARE = "care"
    """Relationship and responsibility focused."""

    JUSTICE = "justice"
    """Fairness and equitable treatment."""


class StakeholderType(str, Enum):
    """Types of stakeholders to consider."""

    INDIVIDUALS = "individuals"
    COMMUNITIES = "communities"
    ORGANIZATIONS = "organizations"
    SOCIETY = "society"
    ENVIRONMENT = "environment"
    FUTURE_GENERATIONS = "future_generations"


@dataclass
class StakeholderImpact:
    """Impact assessment for a stakeholder group."""

    stakeholder: StakeholderType
    impact_type: str
    """Type of impact: 'positive', 'negative', 'neutral', 'mixed'."""
    severity: float
    """Severity of impact (0-1)."""
    description: str
    """Description of the impact."""
    reversible: bool = True
    """Whether the impact is reversible."""


@dataclass
class EthicalConcern:
    """An identified ethical concern in an argument."""

    framework: EthicalFramework
    concern_type: str
    description: str
    severity: float
    """Severity of the concern (0-1)."""
    evidence: str
    """Text evidence from the argument."""
    mitigated: bool = False
    """Whether the concern was addressed."""


@dataclass
class FrameworkScore:
    """Score from a specific ethical framework."""

    framework: EthicalFramework
    score: float
    """Score from 0-1."""
    reasoning: str
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)


@dataclass
class EthicsResult:
    """Complete ethics evaluation result."""

    overall_score: float
    """Overall ethical alignment score (0-1)."""
    framework_scores: list[FrameworkScore]
    """Scores from each ethical framework."""
    stakeholder_impacts: list[StakeholderImpact]
    """Impact on various stakeholders."""
    concerns: list[EthicalConcern]
    """Identified ethical concerns."""
    strengths: list[str]
    """Ethical strengths of the argument."""
    recommendations: list[str]
    """Recommendations for improvement."""


class EthicsEvaluator:
    """
    Evaluates ethical alignment of debate arguments.

    Uses multiple ethical frameworks to provide comprehensive
    ethical analysis including stakeholder impact assessment.

    Example:
        >>> evaluator = EthicsEvaluator()
        >>> result = evaluator.evaluate(argument, context)
        >>> print(f"Ethical score: {result.overall_score}")
    """

    # Keywords indicating ethical consideration by framework
    FRAMEWORK_KEYWORDS = {
        EthicalFramework.UTILITARIAN: [
            "benefit", "welfare", "outcome", "consequences", "greater good",
            "maximize", "optimize", "efficiency", "cost-benefit", "utility",
            "happiness", "well-being", "prosperity", "impact",
        ],
        EthicalFramework.DEONTOLOGICAL: [
            "right", "duty", "obligation", "principle", "rule", "law",
            "rights", "justice", "fairness", "respect", "dignity",
            "autonomy", "consent", "universal", "categorical",
        ],
        EthicalFramework.VIRTUE: [
            "character", "virtue", "integrity", "honesty", "courage",
            "wisdom", "temperance", "excellence", "moral", "ethical",
            "trustworthy", "responsible", "compassion", "humility",
        ],
        EthicalFramework.CARE: [
            "care", "relationship", "responsibility", "nurture", "support",
            "empathy", "compassion", "connection", "community", "family",
            "vulnerable", "protect", "trust", "interdependence",
        ],
        EthicalFramework.JUSTICE: [
            "justice", "fair", "equal", "equity", "rights", "distribution",
            "access", "opportunity", "discrimination", "bias", "inclusive",
            "marginalized", "representation", "democratic",
        ],
    }

    # Keywords indicating ethical concerns
    CONCERN_KEYWORDS = {
        "harm": ["harm", "damage", "hurt", "injury", "suffer", "pain", "risk"],
        "deception": ["deceive", "mislead", "manipulate", "lie", "false", "trick"],
        "exploitation": ["exploit", "abuse", "take advantage", "coerce", "force"],
        "privacy": ["privacy", "surveillance", "data", "personal information"],
        "autonomy": ["control", "force", "mandate", "restrict", "limit freedom"],
        "discrimination": ["discriminate", "bias", "prejudice", "exclude", "unfair"],
    }

    # Stakeholder keywords
    STAKEHOLDER_KEYWORDS = {
        StakeholderType.INDIVIDUALS: [
            "person", "individual", "user", "consumer", "patient", "worker",
            "citizen", "student", "employee",
        ],
        StakeholderType.COMMUNITIES: [
            "community", "neighborhood", "local", "group", "culture", "tradition",
        ],
        StakeholderType.ORGANIZATIONS: [
            "company", "business", "organization", "institution", "corporation",
            "enterprise", "firm",
        ],
        StakeholderType.SOCIETY: [
            "society", "public", "nation", "country", "population", "people",
            "humanity", "civilization",
        ],
        StakeholderType.ENVIRONMENT: [
            "environment", "nature", "climate", "ecosystem", "planet", "earth",
            "species", "biodiversity", "sustainable",
        ],
        StakeholderType.FUTURE_GENERATIONS: [
            "future", "children", "generations", "legacy", "long-term",
            "sustainable", "inherit",
        ],
    }

    def __init__(
        self,
        frameworks: list[EthicalFramework] | None = None,
        weights: dict[EthicalFramework, float] | None = None,
    ) -> None:
        """
        Initialize the ethics evaluator.

        Args:
            frameworks: Ethical frameworks to use (all if None).
            weights: Custom weights for each framework.
        """
        self.frameworks = frameworks or list(EthicalFramework)
        self.weights = weights or {f: 1.0 / len(self.frameworks) for f in self.frameworks}

        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.debug(
            "EthicsEvaluator initialized",
            frameworks=[f.value for f in self.frameworks],
        )

    def evaluate(
        self,
        argument: Argument,
        context: DebateContext,
    ) -> EthicsResult:
        """
        Evaluate the ethical alignment of an argument.

        Args:
            argument: The argument to evaluate.
            context: Debate context.

        Returns:
            EthicsResult with comprehensive ethical analysis.
        """
        logger.debug(
            "Evaluating ethics",
            agent=argument.agent,
            level=argument.level.value,
        )

        # Evaluate each framework
        framework_scores = [
            self._evaluate_framework(argument, context, framework)
            for framework in self.frameworks
        ]

        # Analyze stakeholder impacts
        stakeholder_impacts = self._analyze_stakeholder_impacts(argument, context)

        # Identify concerns
        concerns = self._identify_concerns(argument, context)

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            framework_scores, concerns, context
        )

        # Identify strengths
        strengths = self._identify_strengths(argument, framework_scores)

        # Generate recommendations
        recommendations = self._generate_recommendations(concerns, framework_scores)

        result = EthicsResult(
            overall_score=overall_score,
            framework_scores=framework_scores,
            stakeholder_impacts=stakeholder_impacts,
            concerns=concerns,
            strengths=strengths,
            recommendations=recommendations,
        )

        logger.info(
            "Ethics evaluation complete",
            agent=argument.agent,
            score=overall_score,
            concerns=len(concerns),
        )

        return result

    def _evaluate_framework(
        self,
        argument: Argument,
        context: DebateContext,
        framework: EthicalFramework,
    ) -> FrameworkScore:
        """Evaluate argument from a specific ethical framework."""
        content = argument.content.lower()
        keywords = self.FRAMEWORK_KEYWORDS.get(framework, [])

        # Count keyword matches
        matches = sum(1 for kw in keywords if kw in content)
        keyword_score = min(1.0, matches / max(len(keywords) * 0.3, 1))

        # Analyze reasoning quality for this framework
        reasoning_score = self._analyze_framework_reasoning(
            argument, framework
        )

        # Combine scores
        score = 0.4 * keyword_score + 0.6 * reasoning_score

        # Adjust for context sensitivity
        if context.topic_sensitivity > 0.7:
            # Higher bar for sensitive topics
            score *= 0.9

        strengths = []
        weaknesses = []

        if keyword_score > 0.5:
            strengths.append(f"Shows awareness of {framework.value} considerations")
        if reasoning_score > 0.6:
            strengths.append(f"Demonstrates {framework.value} reasoning")

        if keyword_score < 0.3:
            weaknesses.append(f"Limited {framework.value} perspective")
        if reasoning_score < 0.4:
            weaknesses.append(f"Could strengthen {framework.value} arguments")

        reasoning = self._generate_framework_reasoning(
            framework, score, strengths, weaknesses
        )

        return FrameworkScore(
            framework=framework,
            score=score,
            reasoning=reasoning,
            strengths=strengths,
            weaknesses=weaknesses,
        )

    def _analyze_framework_reasoning(
        self,
        argument: Argument,
        framework: EthicalFramework,
    ) -> float:
        """Analyze reasoning quality for a framework."""
        content = argument.content.lower()

        # Framework-specific patterns
        patterns = {
            EthicalFramework.UTILITARIAN: [
                r"benefit.*outweigh",
                r"maximize.*welfare",
                r"greater.*good",
                r"cost.*benefit",
                r"consequences.*positive",
            ],
            EthicalFramework.DEONTOLOGICAL: [
                r"right.*to",
                r"duty.*to",
                r"principle.*requires",
                r"respect.*dignity",
                r"universal.*rule",
            ],
            EthicalFramework.VIRTUE: [
                r"virtue.*of",
                r"character.*requires",
                r"integrity.*demands",
                r"moral.*excellence",
                r"wisdom.*suggests",
            ],
            EthicalFramework.CARE: [
                r"care.*for",
                r"responsibility.*to",
                r"relationship.*between",
                r"protect.*vulnerable",
                r"support.*those",
            ],
            EthicalFramework.JUSTICE: [
                r"fair.*to",
                r"equal.*access",
                r"justice.*requires",
                r"equitable.*distribution",
                r"rights.*of",
            ],
        }

        framework_patterns = patterns.get(framework, [])
        matches = sum(
            1 for pattern in framework_patterns
            if re.search(pattern, content)
        )

        return min(1.0, matches / max(len(framework_patterns) * 0.5, 1))

    def _generate_framework_reasoning(
        self,
        framework: EthicalFramework,
        score: float,
        strengths: list[str],
        weaknesses: list[str],
    ) -> str:
        """Generate reasoning explanation for framework score."""
        if score >= 0.7:
            level = "strong"
        elif score >= 0.4:
            level = "moderate"
        else:
            level = "limited"

        parts = [f"Shows {level} alignment with {framework.value} ethics."]

        if strengths:
            parts.append(f"Strengths: {strengths[0]}.")
        if weaknesses:
            parts.append(f"Area for improvement: {weaknesses[0]}.")

        return " ".join(parts)

    def _analyze_stakeholder_impacts(
        self,
        argument: Argument,
        context: DebateContext,
    ) -> list[StakeholderImpact]:
        """Analyze impacts on various stakeholders."""
        content = argument.content.lower()
        impacts: list[StakeholderImpact] = []

        for stakeholder, keywords in self.STAKEHOLDER_KEYWORDS.items():
            # Check if stakeholder is mentioned
            mentions = sum(1 for kw in keywords if kw in content)
            if mentions == 0:
                continue

            # Determine impact type
            impact_type = self._determine_impact_type(content, stakeholder)

            # Determine severity
            severity = self._calculate_impact_severity(
                content, stakeholder, context
            )

            # Generate description
            description = self._generate_impact_description(
                stakeholder, impact_type, content
            )

            impacts.append(
                StakeholderImpact(
                    stakeholder=stakeholder,
                    impact_type=impact_type,
                    severity=severity,
                    description=description,
                    reversible=impact_type != "negative" or severity < 0.7,
                )
            )

        return impacts

    def _determine_impact_type(
        self,
        content: str,
        _stakeholder: StakeholderType,
    ) -> str:
        """Determine the type of impact on a stakeholder."""
        positive_words = [
            "benefit", "help", "improve", "protect", "support",
            "enhance", "empower", "enable", "positive",
        ]
        negative_words = [
            "harm", "hurt", "damage", "risk", "threaten",
            "undermine", "negative", "suffer", "lose",
        ]

        pos_count = sum(1 for w in positive_words if w in content)
        neg_count = sum(1 for w in negative_words if w in content)

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        elif pos_count > 0 and neg_count > 0:
            return "mixed"
        else:
            return "neutral"

    def _calculate_impact_severity(
        self,
        content: str,
        stakeholder: StakeholderType,
        context: DebateContext,
    ) -> float:
        """Calculate severity of impact on stakeholder."""
        base_severity = 0.5

        # Intensity modifiers
        high_intensity = ["significant", "major", "substantial", "critical", "severe"]
        low_intensity = ["minor", "slight", "small", "limited", "marginal"]

        if any(word in content for word in high_intensity):
            base_severity += 0.2
        if any(word in content for word in low_intensity):
            base_severity -= 0.2

        # Vulnerable groups get higher weight
        if stakeholder in [
            StakeholderType.FUTURE_GENERATIONS,
            StakeholderType.ENVIRONMENT,
        ]:
            base_severity *= 1.2

        # Adjust for topic sensitivity
        base_severity *= (1 + context.topic_sensitivity * 0.2)

        return min(1.0, max(0.0, base_severity))

    def _generate_impact_description(
        self,
        stakeholder: StakeholderType,
        impact_type: str,
        _content: str,
    ) -> str:
        """Generate description of stakeholder impact."""
        stakeholder_names = {
            StakeholderType.INDIVIDUALS: "individual people",
            StakeholderType.COMMUNITIES: "communities",
            StakeholderType.ORGANIZATIONS: "organizations",
            StakeholderType.SOCIETY: "society at large",
            StakeholderType.ENVIRONMENT: "the environment",
            StakeholderType.FUTURE_GENERATIONS: "future generations",
        }

        name = stakeholder_names.get(stakeholder, str(stakeholder))

        impact_descriptions = {
            "positive": f"Argument suggests positive outcomes for {name}.",
            "negative": f"Argument may have negative implications for {name}.",
            "mixed": f"Argument presents both benefits and risks for {name}.",
        }

        return impact_descriptions.get(
            impact_type,
            f"Argument mentions {name} without clear impact direction.",
        )

    def _identify_concerns(
        self,
        argument: Argument,
        context: DebateContext,
    ) -> list[EthicalConcern]:
        """Identify ethical concerns in the argument."""
        content = argument.content.lower()
        concerns: list[EthicalConcern] = []

        for concern_type, keywords in self.CONCERN_KEYWORDS.items():
            matches = [kw for kw in keywords if kw in content]
            if not matches:
                continue

            # Determine severity
            severity = len(matches) / len(keywords) * 0.7
            severity *= (1 + context.topic_sensitivity * 0.3)
            severity = min(1.0, severity)

            # Determine which framework is most relevant
            framework = self._map_concern_to_framework(concern_type)

            # Check if concern is mitigated
            mitigated = self._check_mitigation(content, concern_type)

            if mitigated:
                severity *= 0.5

            concerns.append(
                EthicalConcern(
                    framework=framework,
                    concern_type=concern_type,
                    description=f"Potential {concern_type} concern identified.",
                    severity=severity,
                    evidence=", ".join(matches),
                    mitigated=mitigated,
                )
            )

        return concerns

    def _map_concern_to_framework(
        self,
        concern_type: str,
    ) -> EthicalFramework:
        """Map concern type to most relevant ethical framework."""
        mapping = {
            "harm": EthicalFramework.UTILITARIAN,
            "deception": EthicalFramework.VIRTUE,
            "exploitation": EthicalFramework.JUSTICE,
            "privacy": EthicalFramework.DEONTOLOGICAL,
            "autonomy": EthicalFramework.DEONTOLOGICAL,
            "discrimination": EthicalFramework.JUSTICE,
        }
        return mapping.get(concern_type, EthicalFramework.UTILITARIAN)

    def _check_mitigation(
        self,
        content: str,
        _concern_type: str,
    ) -> bool:
        """Check if the concern is addressed/mitigated in the argument."""
        mitigation_words = [
            "however", "but", "although", "while", "despite",
            "mitigate", "address", "prevent", "protect", "ensure",
            "safeguard", "careful", "responsible",
        ]

        # Check for mitigation language near concern
        return any(word in content for word in mitigation_words)

    def _calculate_overall_score(
        self,
        framework_scores: list[FrameworkScore],
        concerns: list[EthicalConcern],
        context: DebateContext,
    ) -> float:
        """Calculate overall ethical alignment score."""
        if not framework_scores:
            return 0.5

        # Weighted average of framework scores
        weighted_sum = sum(
            fs.score * self.weights.get(fs.framework, 0.2)
            for fs in framework_scores
        )

        # Penalty for concerns
        concern_penalty = sum(c.severity * 0.1 for c in concerns)
        concern_penalty = min(0.3, concern_penalty)

        # Adjust for topic sensitivity
        sensitivity_factor = 1 - (context.topic_sensitivity * 0.1)

        score = (weighted_sum - concern_penalty) * sensitivity_factor

        return max(0.0, min(1.0, score))

    def _identify_strengths(
        self,
        argument: Argument,
        framework_scores: list[FrameworkScore],
    ) -> list[str]:
        """Identify ethical strengths of the argument."""
        strengths: list[str] = []

        # Collect strengths from high-scoring frameworks
        for fs in framework_scores:
            if fs.score >= 0.6:
                strengths.extend(fs.strengths)

        # Check for balanced consideration
        content = argument.content.lower()
        balance_indicators = ["both sides", "consider", "balance", "weigh"]
        if any(ind in content for ind in balance_indicators):
            strengths.append("Shows balanced ethical consideration")

        # Check for stakeholder awareness
        stakeholder_count = sum(
            1 for keywords in self.STAKEHOLDER_KEYWORDS.values()
            if any(kw in content for kw in keywords)
        )
        if stakeholder_count >= 3:
            strengths.append("Demonstrates broad stakeholder awareness")

        return list(set(strengths))[:5]  # Dedupe and limit

    def _generate_recommendations(
        self,
        concerns: list[EthicalConcern],
        framework_scores: list[FrameworkScore],
    ) -> list[str]:
        """Generate recommendations for ethical improvement."""
        recommendations: list[str] = []

        # Address concerns
        for concern in concerns:
            if concern.severity >= 0.5 and not concern.mitigated:
                recommendations.append(
                    f"Address {concern.concern_type} concerns explicitly."
                )

        # Strengthen weak frameworks
        for fs in framework_scores:
            if fs.score < 0.4:
                recommendations.append(
                    f"Strengthen {fs.framework.value} perspective."
                )

        # General recommendations
        if not recommendations:
            recommendations.append("Continue current ethical framing.")

        return list(set(recommendations))[:5]

    def get_framework_summary(
        self,
        result: EthicsResult,
    ) -> dict[str, float]:
        """Get summary of framework scores."""
        return {
            fs.framework.value: fs.score
            for fs in result.framework_scores
        }

    def __repr__(self) -> str:
        return (
            f"EthicsEvaluator(frameworks={[f.value for f in self.frameworks]})"
        )
