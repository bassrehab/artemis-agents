"""Evidence extraction from arguments."""

import re
from dataclasses import dataclass, field
from enum import Enum

from artemis.core.types import Evidence


class EvidenceType(str, Enum):
    """Types of evidence that can be extracted from arguments."""

    CITATION = "citation"  # Academic or formal citation
    STATISTIC = "statistic"  # Numerical data or statistics
    QUOTE = "quote"  # Direct quotation
    EXAMPLE = "example"  # Illustrative example
    ANECDOTE = "anecdote"  # Personal story or observation
    EXPERT = "expert"  # Expert testimony or opinion
    ANALOGY = "analogy"  # Comparison to similar situation
    HISTORICAL = "historical"  # Historical precedent


class CredibilityLevel(str, Enum):
    """Credibility levels for evidence sources."""

    HIGH = "high"  # Peer-reviewed, authoritative sources
    MEDIUM = "medium"  # Reputable but not verified
    LOW = "low"  # Unverified or questionable
    UNKNOWN = "unknown"  # Cannot determine credibility


@dataclass
class ExtractedEvidence:
    """Evidence extracted from argument text with metadata."""

    evidence: Evidence
    evidence_type: EvidenceType
    credibility: CredibilityLevel
    context: str  # Surrounding text for context
    start_pos: int  # Position in original text
    end_pos: int


@dataclass
class EvidenceExtractor:
    """Extract and classify evidence from argument text."""

    # Configuration for extraction behavior
    min_confidence: float = 0.5
    extract_statistics: bool = True
    extract_quotes: bool = True
    extract_citations: bool = True

    # Pattern definitions
    _patterns: dict[str, re.Pattern[str]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        # XXX: these regexes are pretty fragile
        self._patterns = {
            # Academic citations: (Author, Year) or (Author et al., Year)
            "apa_citation": re.compile(
                r"\(([A-Z][a-z]+(?:\s+(?:et\s+al\.|&\s+[A-Z][a-z]+))?),\s*(\d{4}[a-z]?)\)",
                re.UNICODE,
            ),
            # Numbered citations: [1], [12], etc.
            "numbered_citation": re.compile(r"\[(\d+)\]"),
            # Statistics: percentages and numbers with units
            "percentage": re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent|percentage)", re.IGNORECASE),
            "statistic_with_unit": re.compile(
                r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|trillion|thousand|"
                r"people|users|cases|studies|participants|dollars|euros|pounds)",
                re.IGNORECASE,
            ),
            # Direct quotes
            "double_quotes": re.compile(r'"([^"]{10,})"'),
            "single_quotes": re.compile(r"'([^']{10,})'"),
            # Expert references
            "expert_reference": re.compile(
                r"(?:according\s+to|as\s+stated\s+by|"
                r"(?:Dr\.|Professor|Prof\.|Expert)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))",
                re.IGNORECASE,
            ),
            # Historical references
            "historical": re.compile(
                r"(?:in\s+)?(\d{4}|(?:19|20)\d{2}s?),?\s+([A-Z][a-z]+(?:\s+[a-z]+)*)",
                re.IGNORECASE,
            ),
            # Study/research references
            "study_reference": re.compile(
                r"(?:a\s+)?(?:recent\s+)?(?:study|research|survey|analysis|report|"
                r"meta-analysis)\s+(?:by\s+)?([^,\.]+)?(?:found|shows|suggests|indicates)",
                re.IGNORECASE,
            ),
        }

    def extract(self, text: str):
        """Extract all evidence from the given text."""
        results = []

        if self.extract_citations:
            results.extend(self._extract_citations(text))

        if self.extract_statistics:
            results.extend(self._extract_statistics(text))

        if self.extract_quotes:
            results.extend(self._extract_quotes(text))

        # Extract expert references
        results.extend(self._extract_expert_references(text))

        # Extract study references
        results.extend(self._extract_study_references(text))

        # Sort by position in text
        results.sort(key=lambda x: x.start_pos)

        return results

    def _extract_citations(self, text):
        results = []

        # APA-style citations
        for match in self._patterns["apa_citation"].finditer(text):
            author = match.group(1)
            year = match.group(2)
            context = self._get_context(text, match.start(), match.end())

            results.append(
                ExtractedEvidence(
                    evidence=Evidence(
                        type="study",  # Citations reference studies
                        content=match.group(0),
                        source=f"{author} ({year})",
                        verified=False,
                    ),
                    evidence_type=EvidenceType.CITATION,
                    credibility=CredibilityLevel.MEDIUM,  # Can be verified later
                    context=context,
                    start_pos=match.start(),
                    end_pos=match.end(),
                )
            )

        # Numbered citations
        for match in self._patterns["numbered_citation"].finditer(text):
            context = self._get_context(text, match.start(), match.end())

            results.append(
                ExtractedEvidence(
                    evidence=Evidence(
                        type="study",  # Numbered citations typically reference studies
                        content=match.group(0),
                        source=f"Reference {match.group(1)}",
                        verified=False,
                    ),
                    evidence_type=EvidenceType.CITATION,
                    credibility=CredibilityLevel.UNKNOWN,
                    context=context,
                    start_pos=match.start(),
                    end_pos=match.end(),
                )
            )

        return results

    def _extract_statistics(self, text):
        results = []

        # Percentages
        for match in self._patterns["percentage"].finditer(text):
            context = self._get_context(text, match.start(), match.end())

            results.append(
                ExtractedEvidence(
                    evidence=Evidence(
                        type="statistic",
                        content=match.group(0),
                        verified=False,
                    ),
                    evidence_type=EvidenceType.STATISTIC,
                    credibility=CredibilityLevel.UNKNOWN,
                    context=context,
                    start_pos=match.start(),
                    end_pos=match.end(),
                )
            )

        # Statistics with units
        for match in self._patterns["statistic_with_unit"].finditer(text):
            context = self._get_context(text, match.start(), match.end())

            results.append(
                ExtractedEvidence(
                    evidence=Evidence(
                        type="statistic",
                        content=match.group(0),
                        verified=False,
                    ),
                    evidence_type=EvidenceType.STATISTIC,
                    credibility=CredibilityLevel.UNKNOWN,
                    context=context,
                    start_pos=match.start(),
                    end_pos=match.end(),
                )
            )

        return results

    def _extract_quotes(self, text):
        results = []

        for pattern_name in ["double_quotes", "single_quotes"]:
            for match in self._patterns[pattern_name].finditer(text):
                quote_content = match.group(1)
                context = self._get_context(text, match.start(), match.end())

                results.append(
                    ExtractedEvidence(
                        evidence=Evidence(
                            type="quote",
                            content=quote_content,
                            verified=False,
                        ),
                        evidence_type=EvidenceType.QUOTE,
                        credibility=CredibilityLevel.UNKNOWN,
                        context=context,
                        start_pos=match.start(),
                        end_pos=match.end(),
                    )
                )

        return results

    def _extract_expert_references(self, text):
        results = []

        for match in self._patterns["expert_reference"].finditer(text):
            expert_name = match.group(1) if match.group(1) else match.group(0)
            context = self._get_context(text, match.start(), match.end())

            results.append(
                ExtractedEvidence(
                    evidence=Evidence(
                        type="expert_opinion",  # Use valid literal type
                        content=match.group(0),
                        source=expert_name.strip() if expert_name else None,
                        verified=False,
                    ),
                    evidence_type=EvidenceType.EXPERT,
                    credibility=CredibilityLevel.MEDIUM,
                    context=context,
                    start_pos=match.start(),
                    end_pos=match.end(),
                )
            )

        return results

    def _extract_study_references(self, text):
        results = []

        for match in self._patterns["study_reference"].finditer(text):
            source = match.group(1).strip() if match.group(1) else None
            context = self._get_context(text, match.start(), match.end())

            results.append(
                ExtractedEvidence(
                    evidence=Evidence(
                        type="study",
                        content=match.group(0),
                        source=source,
                        verified=False,
                    ),
                    evidence_type=EvidenceType.CITATION,
                    credibility=CredibilityLevel.MEDIUM,
                    context=context,
                    start_pos=match.start(),
                    end_pos=match.end(),
                )
            )

        return results

    def _get_context(self, text, start, end, window=50):
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]

    def assess_credibility(self, evidence, known_sources=None):
        """Assess credibility of evidence."""
        if not evidence.source:
            return CredibilityLevel.UNKNOWN

        source_lower = evidence.source.lower()

        # High credibility sources
        high_credibility_keywords = [
            "nature",
            "science",
            "lancet",
            "nejm",
            "jama",
            "pnas",
            "cell",
            "ieee",
            "acm",
            "oxford",
            "cambridge",
            "harvard",
            "mit",
            "stanford",
            "who",
            "cdc",
            "nih",
        ]

        for keyword in high_credibility_keywords:
            if keyword in source_lower:
                return CredibilityLevel.HIGH

        # Check against known sources
        if known_sources:
            for known in known_sources:
                if known.lower() in source_lower:
                    return CredibilityLevel.HIGH

        # Medium credibility if source is provided
        if evidence.source:
            return CredibilityLevel.MEDIUM

        return CredibilityLevel.UNKNOWN


class EvidenceLinker:
    """Link evidence between arguments to track support/contradiction."""

    def __init__(self):
        self._evidence_index = {}

    def index_evidence(self, argument_id: str, evidence_list: list[Evidence]) -> None:
        """Index evidence from an argument for later linking."""
        self._evidence_index[argument_id] = evidence_list

    def find_supporting(self, evidence):
        """Find evidence that supports this evidence."""
        supporting = []

        for arg_id, evidence_list in self._evidence_index.items():
            for other in evidence_list:
                if self._is_supporting(evidence, other):
                    supporting.append((arg_id, other))

        return supporting

    def find_contradicting(self, evidence):
        """Find evidence that contradicts this evidence."""
        contradicting = []

        for arg_id, evidence_list in self._evidence_index.items():
            for other in evidence_list:
                if self._is_contradicting(evidence, other):
                    contradicting.append((arg_id, other))

        return contradicting

    def _is_supporting(self, ev1, ev2):
        # Same source typically supports
        if ev1.source and ev2.source and ev1.source.lower() == ev2.source.lower():
            return True

        # Same type and overlapping content
        if ev1.type == ev2.type:
            content1 = ev1.content.lower()
            content2 = ev2.content.lower()
            # Simple overlap check - could be enhanced with NLP
            words1 = set(content1.split())
            words2 = set(content2.split())
            overlap = len(words1 & words2)
            if overlap > 3:  # Arbitrary threshold
                return True

        return False

    def _is_contradicting(self, ev1, ev2):
        # TODO: this is a simplified check - real impl would use NLP
        # Look for negation patterns
        content1 = ev1.content.lower()
        content2 = ev2.content.lower()

        negation_pairs = [
            ("increase", "decrease"),
            ("more", "less"),
            ("higher", "lower"),
            ("positive", "negative"),
            ("success", "failure"),
            ("effective", "ineffective"),
        ]

        for pos, neg in negation_pairs:
            if (pos in content1 and neg in content2) or (neg in content1 and pos in content2):
                # Additional check: same topic
                words1 = set(content1.split()) - {pos, neg}
                words2 = set(content2.split()) - {pos, neg}
                if len(words1 & words2) > 2:
                    return True

        return False

    def clear(self) -> None:
        """Clear all indexed evidence."""
        self._evidence_index.clear()
