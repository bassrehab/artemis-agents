"""Built-in verification rules for argument validation.

Provides rule implementations for verifying different aspects
of argument validity.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from artemis.core.types import (
    VerificationResult,
    VerificationRuleType,
    VerificationViolation,
)
from artemis.utils.logging import get_logger

if TYPE_CHECKING:
    from artemis.core.types import Argument, DebateContext

logger = get_logger(__name__)


class VerificationRuleBase(ABC):
    """Abstract base class for verification rules.

    Example:
        ```python
        class CustomRule(VerificationRuleBase):
            @property
            def rule_type(self) -> VerificationRuleType:
                return VerificationRuleType.LOGICAL_CONSISTENCY

            async def verify(self, argument, context, config):
                # Custom verification logic
                return VerificationResult(...)
        ```
    """

    @property
    @abstractmethod
    def rule_type(self) -> VerificationRuleType:
        """The type of rule this implements."""
        pass

    @abstractmethod
    async def verify(
        self,
        argument: Argument,
        context: DebateContext | None = None,
        config: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Verify the argument against this rule.

        Args:
            argument: The argument to verify.
            context: Optional debate context.
            config: Optional rule configuration.

        Returns:
            VerificationResult with pass/fail and details.
        """
        pass


class CausalChainRule(VerificationRuleBase):
    """Verifies causal reasoning chains are valid.

    Checks that:
    - Causal links are properly connected
    - No circular reasoning
    - Chains have sufficient length

    Example:
        ```python
        rule = CausalChainRule()
        result = await rule.verify(argument, context)
        if not result.passed:
            for v in result.violations:
                print(v.description)
        ```
    """

    @property
    def rule_type(self) -> VerificationRuleType:
        return VerificationRuleType.CAUSAL_CHAIN

    async def verify(
        self,
        argument: Argument,
        context: DebateContext | None = None,
        config: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Verify causal chains in the argument."""
        config = config or {}
        min_chain_length = config.get("min_chain_length", 1)

        violations = []
        causal_links = argument.causal_links or []

        # Check if there are any causal links
        if not causal_links:
            # No causal links - may be intentional for some arguments
            return VerificationResult(
                rule_type=self.rule_type,
                passed=True,
                score=0.7,  # Slight penalty for no causal reasoning
                violations=[],
                details={"causal_links_count": 0},
            )

        # Build graph to check for cycles
        graph: dict[str, list[str]] = {}
        for link in causal_links:
            if link.cause not in graph:
                graph[link.cause] = []
            graph[link.cause].append(link.effect)

        # Check for circular reasoning
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    violations.append(
                        VerificationViolation(
                            description="Circular reasoning detected in causal chain",
                            severity=1.0,
                            suggestion="Break the circular dependency between causes and effects",
                        )
                    )
                    break

        # Check chain length
        max_chain_length = 0
        for start in graph:
            length = self._find_max_chain_length(graph, start, set())
            max_chain_length = max(max_chain_length, length)

        if max_chain_length < min_chain_length:
            violations.append(
                VerificationViolation(
                    description=f"Causal chain too short (length {max_chain_length}, minimum {min_chain_length})",
                    severity=0.5,
                    suggestion="Add more causal links to strengthen the reasoning",
                )
            )

        # Check for unsupported claims (effects without causes)
        all_effects = {link.effect for link in causal_links}
        all_causes = {link.cause for link in causal_links}
        unsupported = all_effects - all_causes

        # It's okay to have some terminal effects, but flag if too many
        unsupported_ratio = len(unsupported) / len(all_effects) if all_effects else 0
        if unsupported_ratio > 0.7:
            violations.append(
                VerificationViolation(
                    description=f"Many effects lack causal support ({len(unsupported)} unsupported)",
                    severity=0.3,
                    suggestion="Add causal links to support the claimed effects",
                )
            )

        # Calculate score
        score = 1.0
        for v in violations:
            score -= v.severity * 0.2
        score = max(0.0, score)

        passed = score >= 0.6 and not any(v.severity >= 1.0 for v in violations)

        return VerificationResult(
            rule_type=self.rule_type,
            passed=passed,
            score=score,
            violations=violations,
            details={
                "causal_links_count": len(causal_links),
                "max_chain_length": max_chain_length,
                "unsupported_effects": len(unsupported),
            },
        )

    def _find_max_chain_length(
        self,
        graph: dict[str, list[str]],
        node: str,
        visited: set[str],
    ) -> int:
        """Find maximum chain length from a node."""
        if node in visited:
            return 0
        visited.add(node)

        max_length = 0
        for neighbor in graph.get(node, []):
            length = self._find_max_chain_length(graph, neighbor, visited.copy())
            max_length = max(max_length, length + 1)

        return max_length


class CitationRule(VerificationRuleBase):
    """Verifies citations and references in arguments.

    Checks for:
    - Proper citation format
    - Citation presence for claims

    Example:
        ```python
        rule = CitationRule()
        result = await rule.verify(argument)
        ```
    """

    # Common citation patterns
    CITATION_PATTERNS = [
        r"\([\w\s]+,?\s*\d{4}\)",  # (Author, 2024)
        r"\[[\w\s]+,?\s*\d{4}\]",  # [Author, 2024]
        r"according to [\w\s]+",  # according to Author
        r"[\w\s]+ et al\.",  # Author et al.
        r"study by [\w\s]+",  # study by Author
        r"research (?:by|from) [\w\s]+",  # research by/from
        r"\d{4} [\w\s]+ study",  # 2024 Harvard study
    ]

    @property
    def rule_type(self) -> VerificationRuleType:
        return VerificationRuleType.CITATION

    async def verify(
        self,
        argument: Argument,
        context: DebateContext | None = None,
        config: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Verify citations in the argument."""
        config = config or {}
        require_citations = config.get("require_citations", False)

        violations = []
        content = argument.content

        # Find citations in text
        citations_found = []
        for pattern in self.CITATION_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            citations_found.extend(matches)

        # Check evidence for citations
        evidence_with_sources = 0
        for ev in argument.evidence or []:
            if ev.source and len(ev.source) > 3:
                evidence_with_sources += 1

        total_evidence = len(argument.evidence or [])
        has_evidence = total_evidence > 0

        # Calculate score
        if not has_evidence and not citations_found:
            if require_citations:
                violations.append(
                    VerificationViolation(
                        description="No citations or evidence sources found",
                        severity=0.8,
                        suggestion="Add citations or references to support claims",
                    )
                )
                score = 0.3
            else:
                # Not required, just lower score
                score = 0.6
        else:
            # Has some citations/evidence
            citation_ratio = (len(citations_found) + evidence_with_sources) / max(
                1, total_evidence + 1
            )
            score = min(1.0, 0.5 + citation_ratio * 0.5)

            if has_evidence and evidence_with_sources < total_evidence * 0.5:
                violations.append(
                    VerificationViolation(
                        description="Some evidence lacks proper source attribution",
                        severity=0.3,
                        suggestion="Add source information to all evidence",
                    )
                )

        passed = score >= 0.6 and not any(v.severity >= 0.8 for v in violations)

        return VerificationResult(
            rule_type=self.rule_type,
            passed=passed,
            score=score,
            violations=violations,
            details={
                "citations_found": len(citations_found),
                "evidence_count": total_evidence,
                "evidence_with_sources": evidence_with_sources,
            },
        )


class LogicalConsistencyRule(VerificationRuleBase):
    """Checks for logical contradictions in arguments.

    Detects:
    - Self-contradicting statements
    - Inconsistent claims

    Example:
        ```python
        rule = LogicalConsistencyRule()
        result = await rule.verify(argument)
        ```
    """

    # Contradiction indicators
    CONTRADICTION_PATTERNS = [
        (r"always", r"never"),
        (r"all", r"none"),
        (r"every", r"no"),
        (r"must", r"cannot"),
        (r"will", r"won't"),
        (r"is", r"isn't"),
        (r"are", r"aren't"),
        (r"should", r"shouldn't"),
    ]

    @property
    def rule_type(self) -> VerificationRuleType:
        return VerificationRuleType.LOGICAL_CONSISTENCY

    async def verify(
        self,
        argument: Argument,
        context: DebateContext | None = None,
        config: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Check for logical contradictions."""
        violations = []
        content = argument.content.lower()

        # Split into sentences
        sentences = re.split(r"[.!?]", content)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Check for contradicting statements
        for i, sent1 in enumerate(sentences):
            for sent2 in sentences[i + 1 :]:
                for pos, neg in self.CONTRADICTION_PATTERNS:
                    # Check if one sentence uses positive and another uses negative
                    if re.search(pos, sent1) and re.search(neg, sent2):
                        # Check if they're about the same topic
                        common_words = set(sent1.split()) & set(sent2.split())
                        if len(common_words) > 2:
                            violations.append(
                                VerificationViolation(
                                    description="Potential contradiction detected between statements",
                                    location=f"'{sent1[:30]}...' vs '{sent2[:30]}...'",
                                    severity=0.6,
                                    suggestion="Review and resolve the apparent contradiction",
                                )
                            )

        # Check for "but" and "however" overuse (weak logic)
        hedging_count = len(re.findall(r"\bbut\b|\bhowever\b|\balthough\b", content))
        if hedging_count > 3:
            violations.append(
                VerificationViolation(
                    description="Excessive hedging may indicate weak logical structure",
                    severity=0.3,
                    suggestion="Strengthen logical connections between claims",
                )
            )

        # Calculate score
        score = 1.0 - (sum(v.severity for v in violations) * 0.15)
        score = max(0.0, min(1.0, score))

        passed = score >= 0.6

        return VerificationResult(
            rule_type=self.rule_type,
            passed=passed,
            score=score,
            violations=violations,
            details={
                "sentence_count": len(sentences),
                "contradiction_count": sum(
                    1 for v in violations if "contradiction" in v.description.lower()
                ),
            },
        )


class EvidenceSupportRule(VerificationRuleBase):
    """Verifies claims are supported by evidence.

    Checks that:
    - Claims have backing evidence
    - Evidence is relevant to claims

    Example:
        ```python
        rule = EvidenceSupportRule()
        result = await rule.verify(argument)
        ```
    """

    # Claim indicators
    CLAIM_INDICATORS = [
        r"therefore",
        r"thus",
        r"hence",
        r"proves that",
        r"shows that",
        r"demonstrates",
        r"it is clear",
        r"obviously",
        r"undoubtedly",
        r"certainly",
    ]

    @property
    def rule_type(self) -> VerificationRuleType:
        return VerificationRuleType.EVIDENCE_SUPPORT

    async def verify(
        self,
        argument: Argument,
        context: DebateContext | None = None,
        config: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Verify evidence support for claims."""
        config = config or {}
        min_evidence_ratio = config.get("min_evidence_ratio", 0.3)

        violations = []
        content = argument.content.lower()

        # Count claim indicators
        claim_count = sum(len(re.findall(pattern, content)) for pattern in self.CLAIM_INDICATORS)

        evidence_count = len(argument.evidence or [])
        causal_count = len(argument.causal_links or [])

        # Calculate support ratio
        total_claims = max(1, claim_count)
        total_support = evidence_count + causal_count
        support_ratio = total_support / total_claims

        if support_ratio < min_evidence_ratio:
            violations.append(
                VerificationViolation(
                    description=f"Insufficient evidence support (ratio: {support_ratio:.2f})",
                    severity=0.5,
                    suggestion="Add more evidence to support claims",
                )
            )

        # Check for unsupported strong claims
        strong_claims = re.findall(
            r"(certainly|obviously|undoubtedly|proves that)[^.!?]*[.!?]",
            content,
        )
        if strong_claims and evidence_count == 0:
            violations.append(
                VerificationViolation(
                    description="Strong claims made without evidence",
                    severity=0.7,
                    suggestion="Provide evidence for strong assertions",
                )
            )

        # Calculate score
        score = min(1.0, 0.4 + support_ratio * 0.6)
        score -= sum(v.severity * 0.1 for v in violations)
        score = max(0.0, score)

        passed = score >= 0.6

        return VerificationResult(
            rule_type=self.rule_type,
            passed=passed,
            score=score,
            violations=violations,
            details={
                "claim_indicators": claim_count,
                "evidence_count": evidence_count,
                "causal_count": causal_count,
                "support_ratio": support_ratio,
            },
        )


class FallacyFreeRule(VerificationRuleBase):
    """Checks for logical fallacies in arguments.

    Detects common fallacies like:
    - Ad hominem
    - Straw man
    - Appeal to authority
    - Slippery slope

    Example:
        ```python
        rule = FallacyFreeRule()
        result = await rule.verify(argument)
        ```
    """

    # Fallacy patterns (simplified)
    FALLACY_PATTERNS = {
        "ad_hominem": [
            r"you('re| are) (wrong|stupid|ignorant)",
            r"(idiot|fool|moron)",
            r"you don't (understand|know)",
        ],
        "appeal_to_authority": [
            r"experts say",
            r"scientists agree",
            r"everyone knows",
        ],
        "straw_man": [
            r"you (think|believe|want) that",
            r"your position is that",
            r"so you're saying",
        ],
        "slippery_slope": [
            r"will (inevitably|certainly) lead to",
            r"if we allow this.*then",
            r"this is just the first step",
        ],
        "false_dichotomy": [
            r"either.*or",
            r"you're either with us or against us",
            r"the only (option|choice|way)",
        ],
    }

    @property
    def rule_type(self) -> VerificationRuleType:
        return VerificationRuleType.FALLACY_FREE

    async def verify(
        self,
        argument: Argument,
        context: DebateContext | None = None,
        config: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Check for logical fallacies."""
        violations = []
        content = argument.content.lower()

        for fallacy_name, patterns in self.FALLACY_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    violations.append(
                        VerificationViolation(
                            description=f"Possible {fallacy_name.replace('_', ' ')} fallacy detected",
                            location=matches[0] if isinstance(matches[0], str) else str(matches[0]),
                            severity=0.5,
                            suggestion=f"Avoid {fallacy_name.replace('_', ' ')} reasoning",
                        )
                    )

        # Calculate score
        score = 1.0 - (len(violations) * 0.15)
        score = max(0.0, min(1.0, score))

        passed = score >= 0.5 and len(violations) < 3

        return VerificationResult(
            rule_type=self.rule_type,
            passed=passed,
            score=score,
            violations=violations,
            details={
                "fallacies_detected": len(violations),
                "fallacy_types": list({v.description.split()[1] for v in violations}),
            },
        )
