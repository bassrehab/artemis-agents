"""Argument verification orchestrator.

Coordinates verification rules to validate arguments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from artemis.core.types import (
    VerificationReport,
    VerificationResult,
    VerificationRule,
    VerificationRuleType,
    VerificationSpec,
)
from artemis.core.verification.rules import (
    CausalChainRule,
    CitationRule,
    EvidenceSupportRule,
    FallacyFreeRule,
    LogicalConsistencyRule,
    VerificationRuleBase,
)
from artemis.utils.logging import get_logger

if TYPE_CHECKING:
    from artemis.core.types import Argument, DebateContext

logger = get_logger(__name__)


class VerificationError(Exception):
    """Raised when verification fails in strict mode."""

    def __init__(self, report: VerificationReport):
        self.report = report
        super().__init__(f"Verification failed: {report.summary}")


class ArgumentVerifier:
    """Verifies arguments against configured rules."""

    # Registry of rule implementations
    RULE_REGISTRY: dict[VerificationRuleType, type[VerificationRuleBase]] = {
        VerificationRuleType.CAUSAL_CHAIN: CausalChainRule,
        VerificationRuleType.CITATION: CitationRule,
        VerificationRuleType.LOGICAL_CONSISTENCY: LogicalConsistencyRule,
        VerificationRuleType.EVIDENCE_SUPPORT: EvidenceSupportRule,
        VerificationRuleType.FALLACY_FREE: FallacyFreeRule,
    }

    def __init__(self, spec: VerificationSpec) -> None:
        self.spec = spec
        self._rules = self._load_rules()

    def _load_rules(self) -> list[tuple[VerificationRule, VerificationRuleBase]]:
        rules = []

        for rule_config in self.spec.rules:
            if not rule_config.enabled:
                continue

            rule_class = self.RULE_REGISTRY.get(rule_config.rule_type)
            if rule_class:
                rules.append((rule_config, rule_class()))
            else:
                logger.warning(
                    "Unknown verification rule type",
                    rule_type=rule_config.rule_type,
                )

        return rules

    async def verify(
        self,
        argument: Argument,
        context: DebateContext | None = None,
    ) -> VerificationReport:
        """Verify an argument against all configured rules."""
        logger.info(
            "Verifying argument",
            argument_id=argument.id,
            rules_count=len(self._rules),
        )

        results: list[VerificationResult] = []
        total_score = 0.0
        total_weight = 0.0

        # NOTE: rules are run sequentially to avoid race conditions on shared state
        for rule_config, rule in self._rules:
            try:
                result = await rule.verify(
                    argument=argument,
                    context=context,
                    config=rule_config.config,
                )
                results.append(result)

                # Weighted score calculation
                total_score += result.score * rule_config.severity
                total_weight += rule_config.severity

            except Exception as e:
                logger.error(
                    "Rule verification failed",
                    rule_type=rule_config.rule_type,
                    error=str(e),
                )
                # Add a failed result
                results.append(
                    VerificationResult(
                        rule_type=rule_config.rule_type,
                        passed=False,
                        score=0.0,
                        violations=[],
                        details={"error": str(e)},
                    )
                )
                total_weight += rule_config.severity

        # Calculate overall score
        overall_score = total_score / total_weight if total_weight > 0 else 0.0

        # Determine if passed
        all_passed = all(r.passed for r in results)
        score_passed = overall_score >= self.spec.min_score

        overall_passed = (all_passed and score_passed) if self.spec.strict_mode else score_passed

        # Build summary
        failed_rules = [r for r in results if not r.passed]
        total_violations = sum(len(r.violations) for r in results)

        if overall_passed:
            summary = f"Verification passed with score {overall_score:.2f}"
        else:
            summary = (
                f"Verification failed with score {overall_score:.2f}. "
                f"Failed rules: {len(failed_rules)}, "
                f"Total violations: {total_violations}"
            )

        report = VerificationReport(
            overall_passed=overall_passed,
            overall_score=overall_score,
            results=results,
            argument_id=argument.id,
            summary=summary,
        )

        logger.info(
            "Verification complete",
            argument_id=argument.id,
            passed=overall_passed,
            score=overall_score,
        )

        # Raise error in strict mode if failed
        if self.spec.strict_mode and not overall_passed:
            raise VerificationError(report)

        return report

    async def verify_batch(
        self,
        arguments: list[Argument],
        context: DebateContext | None = None,
    ) -> list[VerificationReport]:
        """Verify multiple arguments concurrently."""
        import asyncio

        tasks = [self.verify(arg, context) for arg in arguments]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def get_rule_descriptions(self) -> dict[str, str]:
        """Get descriptions of enabled rules."""
        descriptions = {
            VerificationRuleType.CAUSAL_CHAIN: (
                "Verifies causal reasoning chains are valid and connected"
            ),
            VerificationRuleType.CITATION: ("Checks for proper citations and evidence attribution"),
            VerificationRuleType.LOGICAL_CONSISTENCY: (
                "Detects logical contradictions and inconsistencies"
            ),
            VerificationRuleType.EVIDENCE_SUPPORT: ("Ensures claims are supported by evidence"),
            VerificationRuleType.FALLACY_FREE: ("Identifies logical fallacies in arguments"),
        }

        return {
            rule_config.rule_type.value: descriptions.get(rule_config.rule_type, "")
            for rule_config, _ in self._rules
        }


def create_default_verifier(
    strict: bool = False,
    min_score: float = 0.6,
) -> ArgumentVerifier:
    """Create a verifier with default rules enabled."""
    spec = VerificationSpec(
        rules=[
            VerificationRule(
                rule_type=VerificationRuleType.CAUSAL_CHAIN,
                severity=1.0,
            ),
            VerificationRule(
                rule_type=VerificationRuleType.CITATION,
                severity=0.8,
            ),
            VerificationRule(
                rule_type=VerificationRuleType.LOGICAL_CONSISTENCY,
                severity=1.0,
            ),
            VerificationRule(
                rule_type=VerificationRuleType.EVIDENCE_SUPPORT,
                severity=0.9,
            ),
            VerificationRule(
                rule_type=VerificationRuleType.FALLACY_FREE,
                severity=0.7,
            ),
        ],
        strict_mode=strict,
        min_score=min_score,
    )

    return ArgumentVerifier(spec)
