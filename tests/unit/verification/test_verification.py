"""Tests for argument verification."""

import pytest

from artemis.core.types import (
    Argument,
    ArgumentLevel,
    CausalLink,
    Evidence,
    VerificationRule,
    VerificationRuleType,
    VerificationSpec,
)
from artemis.core.verification import (
    ArgumentVerifier,
    CausalChainRule,
    CitationRule,
    CitationValidator,
    EvidenceSupportRule,
    FallacyFreeRule,
    LogicalConsistencyRule,
    VerificationError,
)
from artemis.core.verification.citation_validator import (
    Citation,
    CitationParser,
    CitationStatus,
)


@pytest.fixture
def simple_argument():
    """Create a simple argument for testing."""
    return Argument(
        agent="TestAgent",
        level=ArgumentLevel.TACTICAL,
        content="This is a test argument with clear reasoning.",
    )


@pytest.fixture
def argument_with_evidence():
    """Create an argument with evidence."""
    return Argument(
        agent="TestAgent",
        level=ArgumentLevel.TACTICAL,
        content="According to Smith (2023), AI adoption is increasing. This proves that technology is advancing.",
        evidence=[
            Evidence(
                type="study",
                content="AI adoption is increasing",
                source="Smith (2023)",
            ),
        ],
    )


@pytest.fixture
def argument_with_causal_links():
    """Create an argument with causal links."""
    return Argument(
        agent="TestAgent",
        level=ArgumentLevel.TACTICAL,
        content="Technology leads to productivity gains, which leads to economic growth.",
        causal_links=[
            CausalLink(cause="Technology", effect="Productivity gains", strength=0.8),
            CausalLink(cause="Productivity gains", effect="Economic growth", strength=0.7),
        ],
    )


class TestVerificationTypes:
    """Tests for verification type models."""

    def test_verification_rule_type_values(self):
        """Should have expected rule types."""
        assert VerificationRuleType.CAUSAL_CHAIN == "causal_chain"
        assert VerificationRuleType.CITATION == "citation"
        assert VerificationRuleType.LOGICAL_CONSISTENCY == "logical_consistency"

    def test_verification_rule_creation(self):
        """Should create verification rule."""
        rule = VerificationRule(
            rule_type=VerificationRuleType.CAUSAL_CHAIN,
            enabled=True,
            severity=0.8,
        )
        assert rule.rule_type == VerificationRuleType.CAUSAL_CHAIN
        assert rule.enabled
        assert rule.severity == 0.8

    def test_verification_spec_creation(self):
        """Should create verification spec."""
        spec = VerificationSpec(
            rules=[
                VerificationRule(rule_type=VerificationRuleType.CAUSAL_CHAIN),
            ],
            strict_mode=True,
            min_score=0.7,
        )
        assert len(spec.rules) == 1
        assert spec.strict_mode
        assert spec.min_score == 0.7


class TestCausalChainRule:
    """Tests for CausalChainRule."""

    @pytest.fixture
    def rule(self):
        return CausalChainRule()

    def test_rule_type(self, rule):
        """Should return correct rule type."""
        assert rule.rule_type == VerificationRuleType.CAUSAL_CHAIN

    @pytest.mark.asyncio
    async def test_verify_no_causal_links(self, rule, simple_argument):
        """Should pass with slight penalty for no causal links."""
        result = await rule.verify(simple_argument)
        assert result.passed
        assert result.score < 1.0  # Slight penalty

    @pytest.mark.asyncio
    async def test_verify_valid_chain(self, rule, argument_with_causal_links):
        """Should pass for valid causal chain."""
        result = await rule.verify(argument_with_causal_links)
        assert result.passed
        assert result.details["causal_links_count"] == 2

    @pytest.mark.asyncio
    async def test_detect_circular_reasoning(self, rule):
        """Should detect circular reasoning."""
        arg = Argument(
            agent="Test",
            level=ArgumentLevel.TACTICAL,
            content="Circular argument",
            causal_links=[
                CausalLink(cause="A", effect="B", strength=0.8),
                CausalLink(cause="B", effect="C", strength=0.8),
                CausalLink(cause="C", effect="A", strength=0.8),  # Circular
            ],
        )
        result = await rule.verify(arg)
        assert not result.passed
        assert any("circular" in v.description.lower() for v in result.violations)


class TestCitationRule:
    """Tests for CitationRule."""

    @pytest.fixture
    def rule(self):
        return CitationRule()

    def test_rule_type(self, rule):
        """Should return correct rule type."""
        assert rule.rule_type == VerificationRuleType.CITATION

    @pytest.mark.asyncio
    async def test_verify_with_citations(self, rule, argument_with_evidence):
        """Should pass for argument with citations."""
        result = await rule.verify(argument_with_evidence)
        assert result.passed
        assert result.details["citations_found"] > 0

    @pytest.mark.asyncio
    async def test_verify_no_citations(self, rule, simple_argument):
        """Should have lower score without citations."""
        result = await rule.verify(simple_argument)
        # Default is not to require citations
        assert result.score < 1.0


class TestLogicalConsistencyRule:
    """Tests for LogicalConsistencyRule."""

    @pytest.fixture
    def rule(self):
        return LogicalConsistencyRule()

    def test_rule_type(self, rule):
        """Should return correct rule type."""
        assert rule.rule_type == VerificationRuleType.LOGICAL_CONSISTENCY

    @pytest.mark.asyncio
    async def test_verify_consistent(self, rule, simple_argument):
        """Should pass for consistent argument."""
        result = await rule.verify(simple_argument)
        assert result.passed

    @pytest.mark.asyncio
    async def test_detect_hedging(self, rule):
        """Should detect excessive hedging."""
        arg = Argument(
            agent="Test",
            level=ArgumentLevel.TACTICAL,
            content="This is true, but also this. However, consider this. Although that may be true, but still.",
        )
        result = await rule.verify(arg)
        assert any("hedging" in v.description.lower() for v in result.violations)


class TestEvidenceSupportRule:
    """Tests for EvidenceSupportRule."""

    @pytest.fixture
    def rule(self):
        return EvidenceSupportRule()

    def test_rule_type(self, rule):
        """Should return correct rule type."""
        assert rule.rule_type == VerificationRuleType.EVIDENCE_SUPPORT

    @pytest.mark.asyncio
    async def test_verify_with_evidence(self, rule, argument_with_evidence):
        """Should pass for well-supported argument."""
        result = await rule.verify(argument_with_evidence)
        assert result.passed

    @pytest.mark.asyncio
    async def test_detect_unsupported_claims(self, rule):
        """Should detect unsupported strong claims."""
        arg = Argument(
            agent="Test",
            level=ArgumentLevel.TACTICAL,
            content="This certainly proves that everything is true. Obviously, everyone agrees.",
        )
        result = await rule.verify(arg)
        assert any("strong claims" in v.description.lower() for v in result.violations)


class TestFallacyFreeRule:
    """Tests for FallacyFreeRule."""

    @pytest.fixture
    def rule(self):
        return FallacyFreeRule()

    def test_rule_type(self, rule):
        """Should return correct rule type."""
        assert rule.rule_type == VerificationRuleType.FALLACY_FREE

    @pytest.mark.asyncio
    async def test_verify_no_fallacies(self, rule, simple_argument):
        """Should pass for argument without fallacies."""
        result = await rule.verify(simple_argument)
        assert result.passed

    @pytest.mark.asyncio
    async def test_detect_ad_hominem(self, rule):
        """Should detect ad hominem fallacy."""
        arg = Argument(
            agent="Test",
            level=ArgumentLevel.TACTICAL,
            content="You're wrong because you don't understand the issue.",
        )
        result = await rule.verify(arg)
        assert any("ad hominem" in v.description.lower() for v in result.violations)


class TestArgumentVerifier:
    """Tests for ArgumentVerifier."""

    @pytest.fixture
    def basic_spec(self):
        return VerificationSpec(
            rules=[
                VerificationRule(rule_type=VerificationRuleType.CAUSAL_CHAIN),
                VerificationRule(rule_type=VerificationRuleType.CITATION, severity=0.5),
            ],
            strict_mode=False,
            min_score=0.5,
        )

    @pytest.fixture
    def verifier(self, basic_spec):
        return ArgumentVerifier(basic_spec)

    @pytest.mark.asyncio
    async def test_verify_simple_argument(self, verifier, simple_argument):
        """Should verify simple argument."""
        report = await verifier.verify(simple_argument)
        assert report.argument_id == simple_argument.id
        assert len(report.results) == 2

    @pytest.mark.asyncio
    async def test_verify_strict_mode(self, simple_argument):
        """Should raise error in strict mode on failure."""
        spec = VerificationSpec(
            rules=[
                VerificationRule(
                    rule_type=VerificationRuleType.CITATION,
                    config={"require_citations": True},
                ),
            ],
            strict_mode=True,
            min_score=0.9,
        )
        verifier = ArgumentVerifier(spec)

        with pytest.raises(VerificationError) as exc_info:
            await verifier.verify(simple_argument)

        assert exc_info.value.report is not None

    @pytest.mark.asyncio
    async def test_verify_batch(self, verifier):
        """Should verify multiple arguments."""
        args = [
            Argument(agent="A1", level=ArgumentLevel.TACTICAL, content="Content 1"),
            Argument(agent="A2", level=ArgumentLevel.TACTICAL, content="Content 2"),
        ]
        reports = await verifier.verify_batch(args)
        assert len(reports) == 2

    def test_get_rule_descriptions(self, verifier):
        """Should return rule descriptions."""
        descriptions = verifier.get_rule_descriptions()
        assert "causal_chain" in descriptions
        assert "citation" in descriptions


class TestCitationParser:
    """Tests for CitationParser."""

    @pytest.fixture
    def parser(self):
        return CitationParser()

    def test_parse_author_year(self, parser):
        """Should parse author (year) citations."""
        text = "According to Smith (2023), this is true."
        citations = parser.parse(text)
        assert len(citations) >= 1
        assert any(c.author and "Smith" in c.author for c in citations)

    def test_parse_doi(self, parser):
        """Should parse DOI."""
        text = "See 10.1234/example.doi for details."
        citations = parser.parse(text)
        assert any(c.doi for c in citations)

    def test_parse_url(self, parser):
        """Should parse URLs."""
        text = "More info at https://example.com/article."
        citations = parser.parse(text)
        assert any(c.url for c in citations)


class TestCitationValidator:
    """Tests for CitationValidator."""

    @pytest.fixture
    def validator(self):
        return CitationValidator()

    @pytest.mark.asyncio
    async def test_validate_doi_format(self, validator):
        """Should validate DOI format."""
        citation = Citation(
            raw_text="10.1234/test",
            doi="10.1234/test",
            citation_type="doi",
        )
        result = await validator.validate(citation)
        assert result.status in (CitationStatus.VALID, CitationStatus.UNVERIFIABLE)

    @pytest.mark.asyncio
    async def test_validate_invalid_year(self, validator):
        """Should reject implausible year."""
        citation = Citation(
            raw_text="Smith (3000)",
            author="Smith",
            year=3000,
            citation_type="author_year",
        )
        result = await validator.validate(citation)
        assert result.status == CitationStatus.INVALID

    @pytest.mark.asyncio
    async def test_validate_text(self, validator):
        """Should parse and validate text."""
        text = "According to Jones (2020), this is true."
        results = await validator.validate_text(text)
        assert len(results) > 0

    def test_get_statistics(self, validator):
        """Should return validation statistics."""
        stats = validator.get_statistics()
        assert CitationStatus.VALID.value in stats
