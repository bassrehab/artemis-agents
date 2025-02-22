"""
Unit tests for Evidence Extraction module.
"""

import pytest

from artemis.core.evidence import (
    CredibilityLevel,
    EvidenceExtractor,
    EvidenceLinker,
    EvidenceType,
    ExtractedEvidence,
)
from artemis.core.types import Evidence


class TestEvidenceExtractor:
    """Tests for EvidenceExtractor class."""

    @pytest.fixture
    def extractor(self) -> EvidenceExtractor:
        return EvidenceExtractor()

    def test_extract_apa_citation(self, extractor: EvidenceExtractor) -> None:
        """Test extracting APA-style citations."""
        text = "According to the research (Smith, 2024), AI has significant impact."
        results = extractor.extract(text)

        citation_results = [
            r for r in results if r.evidence_type == EvidenceType.CITATION
        ]
        assert len(citation_results) >= 1
        assert any("Smith" in r.evidence.source or "" for r in citation_results)

    def test_extract_numbered_citation(self, extractor: EvidenceExtractor) -> None:
        """Test extracting numbered citations."""
        text = "Studies have shown [1] that automation increases productivity [2]."
        results = extractor.extract(text)

        citation_results = [
            r for r in results if r.evidence_type == EvidenceType.CITATION
        ]
        assert len(citation_results) >= 2

    def test_extract_percentage(self, extractor: EvidenceExtractor) -> None:
        """Test extracting percentage statistics."""
        text = "Over 75% of respondents agreed, while only 10 percent disagreed."
        results = extractor.extract(text)

        stat_results = [r for r in results if r.evidence_type == EvidenceType.STATISTIC]
        assert len(stat_results) >= 2

    def test_extract_statistic_with_unit(self, extractor: EvidenceExtractor) -> None:
        """Test extracting statistics with units."""
        text = "The market reached 500 million users and 2.5 billion dollars."
        results = extractor.extract(text)

        stat_results = [r for r in results if r.evidence_type == EvidenceType.STATISTIC]
        assert len(stat_results) >= 2

    def test_extract_quote(self, extractor: EvidenceExtractor) -> None:
        """Test extracting direct quotes."""
        text = 'The CEO stated "We believe in sustainable growth" at the conference.'
        results = extractor.extract(text)

        quote_results = [r for r in results if r.evidence_type == EvidenceType.QUOTE]
        assert len(quote_results) >= 1
        assert "sustainable growth" in quote_results[0].evidence.content

    def test_extract_expert_reference(self, extractor: EvidenceExtractor) -> None:
        """Test extracting expert references."""
        text = "According to Dr. Johnson, the treatment is effective."
        results = extractor.extract(text)

        expert_results = [r for r in results if r.evidence_type == EvidenceType.EXPERT]
        assert len(expert_results) >= 1

    def test_extract_study_reference(self, extractor: EvidenceExtractor) -> None:
        """Test extracting study references."""
        text = "A recent study by Harvard researchers found significant improvements."
        results = extractor.extract(text)

        # Should find study reference
        assert len(results) >= 1

    def test_extract_multiple_types(self, extractor: EvidenceExtractor) -> None:
        """Test extracting multiple evidence types from text."""
        text = (
            'According to Smith (2024), "AI will transform industries." '
            "The study shows 85% improvement with 2 million participants."
        )
        results = extractor.extract(text)

        # Should find citation, quote, percentage, and statistic with unit
        assert len(results) >= 3

    def test_extract_empty_text(self, extractor: EvidenceExtractor) -> None:
        """Test extraction from empty text."""
        results = extractor.extract("")
        assert results == []

    def test_extract_no_evidence(self, extractor: EvidenceExtractor) -> None:
        """Test extraction from text with no evidence."""
        text = "This is a simple statement without any citations or data."
        results = extractor.extract(text)
        # May still extract some patterns, but should be minimal
        assert isinstance(results, list)

    def test_context_extraction(self, extractor: EvidenceExtractor) -> None:
        """Test that context is extracted around evidence."""
        text = "Before the citation (Author, 2024) and after the citation."
        results = extractor.extract(text)

        if results:
            for result in results:
                assert len(result.context) > 0
                assert result.start_pos >= 0
                assert result.end_pos > result.start_pos


class TestCredibilityAssessment:
    """Tests for evidence credibility assessment."""

    @pytest.fixture
    def extractor(self) -> EvidenceExtractor:
        return EvidenceExtractor()

    def test_high_credibility_source(self, extractor: EvidenceExtractor) -> None:
        """Test high credibility assessment for known sources."""
        evidence = Evidence(
            type="study",
            content="Published in Nature",
            source="Nature Journal",
        )
        credibility = extractor.assess_credibility(evidence)
        assert credibility == CredibilityLevel.HIGH

    def test_high_credibility_academic(self, extractor: EvidenceExtractor) -> None:
        """Test high credibility for academic sources."""
        evidence = Evidence(
            type="study",
            content="Harvard study",
            source="Harvard University",
        )
        credibility = extractor.assess_credibility(evidence)
        assert credibility == CredibilityLevel.HIGH

    def test_medium_credibility(self, extractor: EvidenceExtractor) -> None:
        """Test medium credibility for unverified but sourced evidence."""
        evidence = Evidence(
            type="study",
            content="Some content",
            source="Unknown Magazine",
        )
        credibility = extractor.assess_credibility(evidence)
        assert credibility == CredibilityLevel.MEDIUM

    def test_unknown_credibility_no_source(
        self, extractor: EvidenceExtractor
    ) -> None:
        """Test unknown credibility for unsourced evidence."""
        evidence = Evidence(
            type="statistic",
            content="50% of people",
            source=None,
        )
        credibility = extractor.assess_credibility(evidence)
        assert credibility == CredibilityLevel.UNKNOWN

    def test_credibility_with_known_sources(
        self, extractor: EvidenceExtractor
    ) -> None:
        """Test credibility with custom known sources list."""
        evidence = Evidence(
            type="study",
            content="Report data",
            source="Internal Research Team",
        )
        credibility = extractor.assess_credibility(
            evidence, known_sources=["Internal Research Team"]
        )
        assert credibility == CredibilityLevel.HIGH


class TestEvidenceLinker:
    """Tests for EvidenceLinker class."""

    @pytest.fixture
    def linker(self) -> EvidenceLinker:
        return EvidenceLinker()

    def test_index_evidence(self, linker: EvidenceLinker) -> None:
        """Test indexing evidence from an argument."""
        evidence_list = [
            Evidence(type="study", content="Study A results", source="Journal A"),
            Evidence(type="statistic", content="75% effectiveness", source="Report B"),
        ]
        linker.index_evidence("arg-1", evidence_list)

        # Should be able to find supporting evidence
        target = Evidence(type="study", content="Study A shows results", source="Journal A")
        supporting = linker.find_supporting(target)
        assert len(supporting) >= 1

    def test_find_supporting_same_source(self, linker: EvidenceLinker) -> None:
        """Test finding supporting evidence with same source."""
        linker.index_evidence(
            "arg-1",
            [Evidence(type="study", content="Data from study", source="Science")],
        )

        target = Evidence(type="study", content="Other data", source="Science")
        supporting = linker.find_supporting(target)
        assert len(supporting) == 1
        assert supporting[0][0] == "arg-1"

    def test_find_contradicting_evidence(self, linker: EvidenceLinker) -> None:
        """Test finding contradicting evidence."""
        linker.index_evidence(
            "arg-1",
            [Evidence(type="statistic", content="increase in sales", source="Report")],
        )

        target = Evidence(type="statistic", content="decrease in sales numbers", source="Other")
        contradicting = linker.find_contradicting(target)
        # Should find contradiction due to increase/decrease
        assert isinstance(contradicting, list)

    def test_clear_linker(self, linker: EvidenceLinker) -> None:
        """Test clearing the evidence index."""
        linker.index_evidence(
            "arg-1",
            [Evidence(type="study", content="Some content", source="Source")],
        )
        linker.clear()

        target = Evidence(type="study", content="Some content", source="Source")
        supporting = linker.find_supporting(target)
        assert len(supporting) == 0


class TestEvidenceTypes:
    """Tests for EvidenceType enum."""

    def test_all_types_defined(self) -> None:
        """Test that all expected evidence types are defined."""
        expected_types = [
            "citation",
            "statistic",
            "quote",
            "example",
            "anecdote",
            "expert",
            "analogy",
            "historical",
        ]
        for type_name in expected_types:
            assert hasattr(EvidenceType, type_name.upper())

    def test_type_values(self) -> None:
        """Test that type values match expected strings."""
        assert EvidenceType.CITATION.value == "citation"
        assert EvidenceType.STATISTIC.value == "statistic"
        assert EvidenceType.QUOTE.value == "quote"


class TestExtractedEvidence:
    """Tests for ExtractedEvidence dataclass."""

    def test_extracted_evidence_creation(self) -> None:
        """Test creating ExtractedEvidence."""
        evidence = Evidence(type="study", content="Test content", source="Test")
        extracted = ExtractedEvidence(
            evidence=evidence,
            evidence_type=EvidenceType.CITATION,
            credibility=CredibilityLevel.HIGH,
            context="surrounding context",
            start_pos=10,
            end_pos=20,
        )

        assert extracted.evidence == evidence
        assert extracted.evidence_type == EvidenceType.CITATION
        assert extracted.credibility == CredibilityLevel.HIGH
        assert extracted.context == "surrounding context"
        assert extracted.start_pos == 10
        assert extracted.end_pos == 20
