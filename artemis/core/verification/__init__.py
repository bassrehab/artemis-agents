"""Formal verification for debate arguments.

Provides tools for verifying argument validity, including
causal chain verification, citation validation, and logical
consistency checking.
"""

from artemis.core.verification.citation_validator import CitationValidator
from artemis.core.verification.rules import (
    CausalChainRule,
    CitationRule,
    EvidenceSupportRule,
    FallacyFreeRule,
    LogicalConsistencyRule,
    VerificationRuleBase,
)
from artemis.core.verification.verifier import (
    ArgumentVerifier,
    VerificationError,
)

__all__ = [
    # Verifier
    "ArgumentVerifier",
    "VerificationError",
    # Rules
    "VerificationRuleBase",
    "CausalChainRule",
    "CitationRule",
    "LogicalConsistencyRule",
    "EvidenceSupportRule",
    "FallacyFreeRule",
    # Validators
    "CitationValidator",
]
