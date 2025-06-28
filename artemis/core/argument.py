"""Utilities for constructing H-L-DAG arguments."""

import re
from dataclasses import dataclass, field

from artemis.core.types import Argument, ArgumentLevel, CausalLink, Evidence


@dataclass
class ArgumentBuilder:
    """Fluent builder for Argument objects."""

    agent: str
    level: ArgumentLevel
    content: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    causal_links: list[CausalLink] = field(default_factory=list)
    rebuts: str | None = None
    supports: str | None = None
    ethical_score: float | None = None
    thinking_trace: str | None = None

    def set_content(self, content: str):
        self.content = content
        return self

    def add_evidence(self, evidence: Evidence):
        self.evidence.append(evidence)
        return self

    def add_causal_link(self, link: CausalLink):
        self.causal_links.append(link)
        return self

    def set_rebuts(self, argument_id: str):
        self.rebuts = argument_id
        return self

    def set_supports(self, argument_id: str):
        self.supports = argument_id
        return self

    def set_ethical_score(self, score: float):
        self.ethical_score = score
        return self

    def set_thinking_trace(self, trace: str):
        self.thinking_trace = trace
        return self

    def build(self):
        return Argument(
            agent=self.agent,
            level=self.level,
            content=self.content,
            evidence=self.evidence,
            causal_links=self.causal_links,
            rebuts=self.rebuts,
            supports=self.supports,
            ethical_score=self.ethical_score,
            thinking_trace=self.thinking_trace,
        )


class ArgumentParser:
    """Parse structured arguments from LLM output."""

    # Patterns for extracting structured content
    EVIDENCE_PATTERN = re.compile(
        r"\[(?:EVIDENCE|SOURCE|CITE)\]\s*(.+?)(?=\[|$)", re.IGNORECASE | re.DOTALL
    )
    CAUSAL_PATTERN = re.compile(
        r"(?:because|therefore|thus|hence|consequently|if\s+.+?\s+then)\s+(.+?)(?:\.|$)",
        re.IGNORECASE,
    )
    CITATION_PATTERN = re.compile(
        r"\(([^)]+,\s*\d{4}[a-z]?)\)|"  # (Author, 2024)
        r"\[(\d+)\]|"  # [1]
        r"according to ([^,]+)",  # according to Source
        re.IGNORECASE,
    )

    def parse(self, content: str, agent: str, level: ArgumentLevel) -> Argument:
        """Parse content and extract structured elements."""
        builder = ArgumentBuilder(agent=agent, level=level)
        builder.set_content(content)

        # Extract evidence
        evidence = self._extract_evidence(content)
        for ev in evidence:
            builder.add_evidence(ev)

        # Extract causal links
        causal_links = self._extract_causal_links(content)
        for link in causal_links:
            builder.add_causal_link(link)

        return builder.build()

    def _extract_evidence(self, content: str):
        # NOTE: regex-based extraction is not perfect
        evidence_list = []

        # Look for explicit evidence markers
        for match in self.EVIDENCE_PATTERN.finditer(content):
            evidence_list.append(
                Evidence(
                    type="quote",
                    content=match.group(1).strip(),
                    verified=False,
                )
            )

        # Look for citation patterns
        for match in self.CITATION_PATTERN.finditer(content):
            citation = match.group(1) or match.group(2) or match.group(3)
            if citation:
                evidence_list.append(
                    Evidence(
                        type="quote",
                        content=match.group(0),
                        source=citation.strip(),
                        verified=False,
                    )
                )

        return evidence_list

    def _extract_causal_links(self, content: str):
        links = []

        for match in self.CAUSAL_PATTERN.finditer(content):
            # Extract the causal statement
            statement = match.group(1).strip()

            # Try to identify cause and effect
            # This is a simplified extraction - real implementation would use NLP
            parts = re.split(r"\s+leads to\s+|\s+causes\s+|\s+results in\s+", statement)
            if len(parts) >= 2:
                links.append(
                    CausalLink(
                        cause=parts[0].strip()[:100],  # Limit length
                        effect=parts[1].strip()[:100],
                        strength=0.5,  # Default strength
                    )
                )

        return links


class ArgumentHierarchy:
    """Manages argument hierarchy in H-L-DAG."""

    def __init__(self):
        self._arguments: dict[str, Argument] = {}
        self._children: dict[str, list[str]] = {}  # parent_id -> [child_ids]
        self._parents: dict[str, str] = {}  # child_id -> parent_id

    def add(self, argument: Argument, parent_id: str | None = None) -> None:
        """Add an argument to the hierarchy."""
        # Validate hierarchy
        if parent_id:
            parent = self._arguments.get(parent_id)
            if parent:
                self._validate_hierarchy(parent.level, argument.level)

            # Link to parent
            self._parents[argument.id] = parent_id
            if parent_id not in self._children:
                self._children[parent_id] = []
            self._children[parent_id].append(argument.id)

        self._arguments[argument.id] = argument

    def _validate_hierarchy(self, parent_level, child_level):
        level_order = {
            ArgumentLevel.STRATEGIC: 0,
            ArgumentLevel.TACTICAL: 1,
            ArgumentLevel.OPERATIONAL: 2,
        }

        if level_order[child_level] <= level_order[parent_level]:
            raise ValueError(
                f"Cannot add {child_level.value} argument as child of "
                f"{parent_level.value} argument. Children must be at a "
                f"lower hierarchical level."
            )

    def get(self, argument_id: str):
        return self._arguments.get(argument_id)

    def get_children(self, argument_id: str):
        child_ids = self._children.get(argument_id, [])
        return [self._arguments[cid] for cid in child_ids if cid in self._arguments]

    def get_parent(self, argument_id: str):
        parent_id = self._parents.get(argument_id)
        return self._arguments.get(parent_id) if parent_id else None

    def get_by_level(self, level):
        return [arg for arg in self._arguments.values() if arg.level == level]

    def get_strategic_arguments(self):
        return self.get_by_level(ArgumentLevel.STRATEGIC)

    def get_tactical_arguments(self):
        return self.get_by_level(ArgumentLevel.TACTICAL)

    def get_operational_arguments(self):
        return self.get_by_level(ArgumentLevel.OPERATIONAL)

    def get_argument_chain(self, argument_id: str):
        chain = []
        current_id: str | None = argument_id

        while current_id:
            arg = self._arguments.get(current_id)
            if arg:
                chain.insert(0, arg)
            current_id = self._parents.get(current_id)

        return chain

    @property
    def all_arguments(self):
        return list(self._arguments.values())

    def __len__(self) -> int:
        return len(self._arguments)
