"""Audit log exporter for ARTEMIS debates."""

import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from artemis.core.types import (
    DebateResult,
    SafetyAlert,
    Turn,
)
from artemis.utils.logging import get_logger

logger = get_logger(__name__)


def _md_to_html(text: str) -> str:
    """Convert basic markdown to HTML."""
    if not text:
        return ""

    # Escape HTML entities first
    text = html.escape(text)

    # Headers: ## Header -> <h4>Header</h4>
    text = re.sub(r'^## (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$', r'<h5>\1</h5>', text, flags=re.MULTILINE)

    # Bold: **text** -> <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    # Italic: *text* -> <em>text</em>
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

    # Bullet points: - item -> <li>item</li>
    text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)

    # Wrap consecutive <li> in <ul>
    text = re.sub(r'((?:<li>.*?</li>\n?)+)', r'<ul>\1</ul>', text)

    # Numbered lists: 1. item -> <li>item</li>
    text = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)

    # Line breaks for paragraphs
    text = re.sub(r'\n\n+', '</p><p>', text)
    text = re.sub(r'\n', '<br>', text)

    # Wrap in paragraph if not already structured
    if not text.startswith('<'):
        text = f'<p>{text}</p>'

    return text


@dataclass
class AuditEntry:
    """Single entry in the audit log."""

    timestamp: datetime
    debate_id: str
    event_type: str
    agent: str | None = None
    round: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditLog:
    """Complete audit log for a debate."""

    debate_id: str
    topic: str
    entries: list[AuditEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_debate_result(cls, result: DebateResult) -> "AuditLog":
        """Create audit log from a debate result."""
        entries = []

        # Add debate start entry
        entries.append(AuditEntry(
            timestamp=result.metadata.started_at,
            debate_id=result.debate_id,
            event_type="debate_started",
            details={
                "topic": result.topic,
                "agents": result.metadata.agents,
                "total_rounds": result.metadata.total_rounds,
                "jury_size": result.metadata.jury_size,
            },
        ))

        # Add entries for each turn
        for turn in result.transcript:
            entries.append(cls._turn_to_entry(turn, result.debate_id))

            # Add evaluation entry with full details
            if turn.evaluation:
                entries.append(AuditEntry(
                    timestamp=turn.timestamp,
                    debate_id=result.debate_id,
                    event_type="argument_evaluated",
                    agent=turn.agent,
                    round=turn.round,
                    details={
                        "total_score": turn.evaluation.total_score,
                        "scores": turn.evaluation.scores,
                        "weights": turn.evaluation.weights,
                        "feedback": getattr(turn.evaluation, "feedback", None),
                        "strengths": getattr(turn.evaluation, "strengths", []),
                        "weaknesses": getattr(turn.evaluation, "weaknesses", []),
                        "criteria_met": getattr(turn.evaluation, "criteria_met", []),
                    },
                ))

            # Add safety results - include ALL results for audit trail
            for safety in turn.safety_results:
                entries.append(AuditEntry(
                    timestamp=turn.timestamp,
                    debate_id=result.debate_id,
                    event_type="safety_check",
                    agent=turn.agent,
                    round=turn.round,
                    details={
                        "monitor": safety.monitor,
                        "passed": safety.severity == 0,
                        "severity": safety.severity,
                        "should_alert": safety.should_alert,
                        "analysis_notes": safety.analysis_notes,
                        "indicators": getattr(safety, "indicators", []),
                        "recommendation": getattr(safety, "recommendation", None),
                        "confidence": getattr(safety, "confidence", None),
                    },
                ))

        # Add safety alerts
        for alert in result.safety_alerts:
            entries.append(cls._alert_to_entry(alert, result.debate_id))

        # Add verdict entry
        if result.verdict:
            entries.append(AuditEntry(
                timestamp=result.metadata.ended_at or datetime.utcnow(),
                debate_id=result.debate_id,
                event_type="verdict_issued",
                details={
                    "decision": result.verdict.decision,
                    "confidence": result.verdict.confidence,
                    "reasoning": result.verdict.reasoning,
                    "unanimous": result.verdict.unanimous,
                    "score_breakdown": result.verdict.score_breakdown,
                    "dissenting_count": len(result.verdict.dissenting_opinions),
                },
            ))

        # Add debate end entry
        entries.append(AuditEntry(
            timestamp=result.metadata.ended_at or datetime.utcnow(),
            debate_id=result.debate_id,
            event_type="debate_completed",
            details={
                "final_state": result.final_state.value,
                "total_turns": result.metadata.total_turns,
            },
        ))

        # Add verification reports
        for report in result.verification_reports:
            entries.append(AuditEntry(
                timestamp=result.metadata.ended_at or datetime.utcnow(),
                debate_id=result.debate_id,
                event_type="verification_report",
                details={
                    "argument_id": report.argument_id,
                    "overall_passed": report.overall_passed,
                    "overall_score": report.overall_score,
                    "results": [
                        {
                            "rule_type": r.rule_type.value if hasattr(r.rule_type, 'value') else str(r.rule_type),
                            "passed": r.passed,
                            "score": r.score,
                            "violations": [
                                {"severity": v.severity, "description": v.description}
                                for v in r.violations
                            ],
                        }
                        for r in report.results
                    ],
                },
            ))

        # Add sub-debate entries for hierarchical debates
        for sub_debate in result.sub_debates:
            entries.append(AuditEntry(
                timestamp=result.metadata.ended_at or datetime.utcnow(),
                debate_id=result.debate_id,
                event_type="sub_debate",
                details=sub_debate,
            ))

        # Add compound verdict for hierarchical debates
        if result.compound_verdict:
            entries.append(AuditEntry(
                timestamp=result.metadata.ended_at or datetime.utcnow(),
                debate_id=result.debate_id,
                event_type="compound_verdict",
                details={
                    "final_decision": result.compound_verdict.final_decision,
                    "confidence": result.compound_verdict.confidence,
                    "reasoning": result.compound_verdict.reasoning,
                    "aggregation_method": result.compound_verdict.aggregation_method,
                    "sub_verdicts_count": len(result.compound_verdict.sub_verdicts),
                    "aggregation_weights": result.compound_verdict.aggregation_weights,
                },
            ))

        # Build metadata
        metadata = {
            "started_at": result.metadata.started_at.isoformat(),
            "ended_at": result.metadata.ended_at.isoformat() if result.metadata.ended_at else None,
            "total_rounds": result.metadata.total_rounds,
            "total_turns": result.metadata.total_turns,
            "agents": result.metadata.agents,
            "jury_size": result.metadata.jury_size,
            "safety_monitors": result.metadata.safety_monitors,
            "model_usage": result.metadata.model_usage,
            # V2 Features
            "steering_config": result.metadata.steering_config,
            "hierarchical_config": result.metadata.hierarchical_config,
            "verification_spec": result.metadata.verification_spec,
        }

        return cls(
            debate_id=result.debate_id,
            topic=result.topic,
            entries=entries,
            metadata=metadata,
        )

    @staticmethod
    def _turn_to_entry(turn: Turn, debate_id: str) -> AuditEntry:
        """Convert a turn to an audit entry with FULL content."""
        # Full evidence details with multimodal support
        evidence_full = []
        for e in turn.argument.evidence:
            ev_entry = {
                "type": e.type,
                "content": e.content,  # Full content, not truncated
                "source": e.source,
                "confidence": e.confidence,
                "verified": getattr(e, "verified", None),
            }
            # Add multimodal fields if present
            if hasattr(e, 'content_type') and e.content_type:
                ev_entry["content_type"] = e.content_type.value if hasattr(e.content_type, 'value') else str(e.content_type)
            if hasattr(e, 'url') and e.url:
                ev_entry["url"] = e.url
            if hasattr(e, 'media_type') and e.media_type:
                ev_entry["media_type"] = e.media_type
            if hasattr(e, 'alt_text') and e.alt_text:
                ev_entry["alt_text"] = e.alt_text
            if hasattr(e, 'filename') and e.filename:
                ev_entry["filename"] = e.filename
            evidence_full.append(ev_entry)

        # Full causal links
        causal_links = []
        for link in turn.argument.causal_links:
            causal_links.append({
                "cause": link.cause,
                "effect": link.effect,
                "strength": getattr(link, "strength", None),
                "mechanism": getattr(link, "mechanism", None),
            })

        return AuditEntry(
            timestamp=turn.timestamp,
            debate_id=debate_id,
            event_type="argument_generated",
            agent=turn.agent,
            round=turn.round,
            details={
                "turn_id": turn.id,
                "level": turn.argument.level.value,
                "content": turn.argument.content,  # FULL content
                "evidence": evidence_full,
                "causal_links": causal_links,
                "rebuts": turn.argument.rebuts,
                "supports": turn.argument.supports,
                "ethical_score": getattr(turn.argument, "ethical_score", None),
            },
        )

    @staticmethod
    def _alert_to_entry(alert: SafetyAlert, debate_id: str) -> AuditEntry:
        """Convert a safety alert to an audit entry."""
        return AuditEntry(
            timestamp=alert.timestamp,
            debate_id=debate_id,
            event_type="safety_alert",
            agent=alert.agent,
            details={
                "monitor": alert.monitor,
                "type": alert.type,
                "severity": alert.severity,
                "indicators": [
                    {"type": ind.type.value, "severity": ind.severity, "evidence": ind.evidence}
                    for ind in alert.indicators
                ],
                "resolved": alert.resolved,
            },
        )

    def to_json(self, path: Path | str | None = None, indent: int = 2) -> str:
        """Export audit log as JSON."""
        data = {
            "debate_id": self.debate_id,
            "topic": self.topic,
            "metadata": self.metadata,
            "entries": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type,
                    "agent": e.agent,
                    "round": e.round,
                    "details": e.details,
                }
                for e in self.entries
            ],
        }

        json_str = json.dumps(data, indent=indent, default=str)

        if path:
            Path(path).write_text(json_str)
            logger.info("Audit log exported to JSON", path=str(path))

        return json_str

    def to_markdown(self, path: Path | str | None = None) -> str:
        """Export audit log as comprehensive Markdown with full content."""
        lines = []

        # Header
        lines.append("# Debate Audit Log")
        lines.append("")
        lines.append(f"**Topic:** {self.topic}")
        lines.append(f"**Debate ID:** `{self.debate_id}`")
        lines.append("")

        # Metadata
        lines.append("## Metadata")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| Started | {self.metadata.get('started_at', 'N/A')} |")
        lines.append(f"| Ended | {self.metadata.get('ended_at', 'N/A')} |")
        lines.append(f"| Rounds | {self.metadata.get('total_rounds', 'N/A')} |")
        lines.append(f"| Turns | {self.metadata.get('total_turns', 'N/A')} |")
        lines.append(f"| Agents | {', '.join(self.metadata.get('agents', []))} |")
        lines.append(f"| Jury Size | {self.metadata.get('jury_size', 'N/A')} |")
        lines.append(f"| Safety Monitors | {', '.join(self.metadata.get('safety_monitors', [])) or 'None'} |")
        lines.append("")

        # Model usage if available
        model_usage = self.metadata.get('model_usage', {})
        if model_usage:
            lines.append("### Model Usage")
            lines.append("")
            for agent, usage in model_usage.items():
                if isinstance(usage, dict):
                    lines.append(f"- **{agent}**: {usage.get('total_tokens', 0):,} tokens")
            lines.append("")

        # V2 Features Configuration
        steering_config = self.metadata.get('steering_config')
        hierarchical_config = self.metadata.get('hierarchical_config')
        verification_spec = self.metadata.get('verification_spec')

        if steering_config or hierarchical_config or verification_spec:
            lines.append("## V2 Features Configuration")
            lines.append("")

            if steering_config:
                lines.append("### Steering Configuration")
                lines.append("")
                vector = steering_config.get('vector', {})
                if vector:
                    lines.append("| Dimension | Value |")
                    lines.append("|-----------|-------|")
                    for dim, val in vector.items():
                        lines.append(f"| {dim} | {val:.2f} |")
                    lines.append("")
                mode = steering_config.get('mode', 'prompt')
                strength = steering_config.get('strength', 1.0)
                adaptive = steering_config.get('adaptive', False)
                lines.append(f"- **Mode:** {mode}")
                lines.append(f"- **Strength:** {strength:.2f}")
                lines.append(f"- **Adaptive:** {'Yes' if adaptive else 'No'}")
                lines.append("")

            if hierarchical_config:
                lines.append("### Hierarchical Debate Configuration")
                lines.append("")
                max_depth = hierarchical_config.get('max_depth', 2)
                decomposer = hierarchical_config.get('decomposer', 'LLM')
                aggregator = hierarchical_config.get('aggregator', 'weighted_average')
                lines.append(f"- **Max Depth:** {max_depth}")
                lines.append(f"- **Decomposer:** {decomposer}")
                lines.append(f"- **Aggregator:** {aggregator}")
                lines.append("")

            if verification_spec:
                lines.append("### Verification Specification")
                lines.append("")
                rules = verification_spec.get('rules', [])
                strict_mode = verification_spec.get('strict_mode', False)
                min_score = verification_spec.get('min_score', 0.6)
                lines.append(f"- **Strict Mode:** {'Yes' if strict_mode else 'No'}")
                lines.append(f"- **Minimum Score:** {min_score:.2f}")
                if rules:
                    lines.append("- **Rules:**")
                    for rule in rules:
                        rule_type = rule.get('rule_type', 'unknown')
                        enabled = rule.get('enabled', True)
                        severity = rule.get('severity', 1.0)
                        lines.append(f"  - {rule_type} (enabled: {enabled}, severity: {severity:.2f})")
                lines.append("")

        # Full Transcript
        lines.append("## Full Transcript")
        lines.append("")

        current_round = None
        for entry in self.entries:
            if entry.event_type == "argument_generated":
                if entry.round != current_round:
                    current_round = entry.round
                    round_label = "Opening Statements" if current_round == 0 else f"Round {current_round}"
                    lines.append(f"### {round_label}")
                    lines.append("")
                    lines.append("---")
                    lines.append("")

                # Agent header with full details
                turn_id = entry.details.get('turn_id', 'N/A')
                level = entry.details.get('level', 'N/A')
                ethical_score = entry.details.get('ethical_score')
                ethical_str = f" | Ethical Score: {ethical_score:.2f}" if ethical_score else ""

                lines.append(f"#### {entry.agent}")
                lines.append("")
                lines.append(f"*Level: {level}{ethical_str} | Turn ID: `{turn_id}`*")
                lines.append("")

                # FULL argument content
                content = entry.details.get('content', '')
                lines.append(content)
                lines.append("")

                # Rebuts/Supports
                rebuts = entry.details.get('rebuts', [])
                supports = entry.details.get('supports', [])
                if rebuts:
                    lines.append(f"**Rebuts:** {', '.join(f'`{r}`' for r in rebuts)}")
                    lines.append("")
                if supports:
                    lines.append(f"**Supports:** {', '.join(f'`{s}`' for s in supports)}")
                    lines.append("")

                # Full Evidence with multimodal support
                evidence = entry.details.get('evidence', [])
                if evidence:
                    lines.append(f"**Evidence ({len(evidence)}):**")
                    lines.append("")
                    for i, e in enumerate(evidence, 1):
                        verified = " (verified)" if e.get('verified') else ""
                        content_type = e.get('content_type', 'text')
                        type_badge = f"[{e.get('type')}]"
                        if content_type and content_type != 'text':
                            type_badge = f"[{e.get('type')} - {content_type}]"
                        lines.append(f"{i}. **{type_badge}** {e.get('content')}")
                        lines.append(f"   - Source: {e.get('source', 'N/A')}")
                        lines.append(f"   - Confidence: {e.get('confidence', 0):.2f}{verified}")
                        # Multimodal fields
                        if e.get('url'):
                            lines.append(f"   - URL: {e.get('url')}")
                        if e.get('media_type'):
                            lines.append(f"   - Media Type: {e.get('media_type')}")
                        if e.get('filename'):
                            lines.append(f"   - Filename: {e.get('filename')}")
                        if e.get('alt_text'):
                            lines.append(f"   - Alt Text: {e.get('alt_text')}")
                    lines.append("")

                # Causal Links
                causal_links = entry.details.get('causal_links', [])
                if causal_links:
                    lines.append(f"**Causal Links ({len(causal_links)}):**")
                    lines.append("")
                    for link in causal_links:
                        strength = f" (strength: {link.get('strength', 'N/A')})" if link.get('strength') else ""
                        lines.append(f"- {link.get('cause')} → {link.get('effect')}{strength}")
                        if link.get('mechanism'):
                            lines.append(f"  - Mechanism: {link.get('mechanism')}")
                    lines.append("")

            # Evaluation details
            elif entry.event_type == "argument_evaluated":
                lines.append("<details>")
                lines.append(f"<summary><strong>Evaluation</strong> (Score: {entry.details.get('total_score', 0):.2f})</summary>")
                lines.append("")

                # Score breakdown
                scores = entry.details.get('scores', {})
                weights = entry.details.get('weights', {})
                if scores:
                    lines.append("| Criterion | Score | Weight |")
                    lines.append("|-----------|-------|--------|")
                    for criterion, score in scores.items():
                        weight = weights.get(criterion, 0)
                        lines.append(f"| {criterion} | {score:.2f} | {weight:.2f} |")
                    lines.append("")

                # Feedback
                feedback = entry.details.get('feedback')
                if feedback:
                    lines.append(f"**Feedback:** {feedback}")
                    lines.append("")

                # Strengths and Weaknesses
                strengths = entry.details.get('strengths', [])
                weaknesses = entry.details.get('weaknesses', [])
                if strengths:
                    lines.append("**Strengths:**")
                    for s in strengths:
                        lines.append(f"- {s}")
                    lines.append("")
                if weaknesses:
                    lines.append("**Weaknesses:**")
                    for w in weaknesses:
                        lines.append(f"- {w}")
                    lines.append("")

                lines.append("</details>")
                lines.append("")

            # Safety check details
            elif entry.event_type == "safety_check":
                passed = entry.details.get('passed', True)
                severity = entry.details.get('severity', 0)
                if not passed or severity > 0:
                    icon = "🔴" if severity > 0.7 else "🟡" if severity > 0.3 else "🟢"
                    lines.append(f"{icon} **Safety Check** ({entry.details.get('monitor', 'Unknown')})")
                    lines.append(f"- Severity: {severity:.2f}")
                    notes = entry.details.get('analysis_notes')
                    if notes:
                        lines.append(f"- Notes: {notes}")
                    recommendation = entry.details.get('recommendation')
                    if recommendation:
                        lines.append(f"- Recommendation: {recommendation}")
                    lines.append("")

        # Verdict
        verdict_entries = [e for e in self.entries if e.event_type == "verdict_issued"]
        if verdict_entries:
            verdict = verdict_entries[0]
            lines.append("## Verdict")
            lines.append("")
            lines.append(f"**Decision:** {verdict.details.get('decision', 'N/A')}")
            lines.append(f"**Confidence:** {verdict.details.get('confidence', 0):.0%}")
            lines.append(f"**Unanimous:** {'Yes' if verdict.details.get('unanimous') else 'No'}")
            lines.append("")
            lines.append("### Reasoning")
            lines.append("")
            lines.append(f"{verdict.details.get('reasoning', '')}")
            lines.append("")

            # Score breakdown
            scores = verdict.details.get('score_breakdown', {})
            if scores:
                lines.append("### Score Breakdown")
                lines.append("")
                lines.append("| Agent | Score |")
                lines.append("|-------|-------|")
                for agent, score in scores.items():
                    lines.append(f"| {agent} | {score:.2f} |")
                lines.append("")

        # Safety Alerts
        alert_entries = [e for e in self.entries if e.event_type == "safety_alert"]
        if alert_entries:
            lines.append("## Safety Alerts")
            lines.append("")
            for alert in alert_entries:
                severity = alert.details.get('severity', 0)
                severity_label = "HIGH" if severity > 0.7 else "MEDIUM" if severity > 0.4 else "LOW"
                lines.append(f"### [{severity_label}] {alert.details.get('type', 'Unknown')}")
                lines.append("")
                lines.append(f"- **Agent:** {alert.agent}")
                lines.append(f"- **Monitor:** {alert.details.get('monitor', 'N/A')}")
                lines.append(f"- **Severity:** {severity:.2f}")
                lines.append(f"- **Resolved:** {'Yes' if alert.details.get('resolved') else 'No'}")
                lines.append("")

        # Hierarchical Debate Results (V2)
        sub_debate_entries = [e for e in self.entries if e.event_type == "sub_debate"]
        compound_verdict_entries = [e for e in self.entries if e.event_type == "compound_verdict"]

        if sub_debate_entries or compound_verdict_entries:
            lines.append("## Hierarchical Debate Results")
            lines.append("")

            if sub_debate_entries:
                lines.append("### Sub-Debates")
                lines.append("")
                for i, sub in enumerate(sub_debate_entries, 1):
                    aspect = sub.details.get('aspect', f'Sub-debate {i}')
                    weight = sub.details.get('weight', 1.0)
                    sub_verdict = sub.details.get('verdict', {})
                    lines.append(f"#### {i}. {aspect}")
                    lines.append("")
                    lines.append(f"- **Weight:** {weight:.2f}")
                    if sub_verdict:
                        lines.append(f"- **Decision:** {sub_verdict.get('decision', 'N/A')}")
                        lines.append(f"- **Confidence:** {sub_verdict.get('confidence', 0):.0%}")
                        reasoning = sub_verdict.get('reasoning', '')
                        if reasoning:
                            lines.append(f"- **Reasoning:** {reasoning[:200]}...")
                    lines.append("")

            if compound_verdict_entries:
                cv = compound_verdict_entries[0]
                lines.append("### Compound Verdict (Aggregated)")
                lines.append("")
                lines.append(f"**Final Decision:** {cv.details.get('final_decision', 'N/A')}")
                lines.append(f"**Confidence:** {cv.details.get('confidence', 0):.0%}")
                lines.append(f"**Aggregation Method:** {cv.details.get('aggregation_method', 'N/A')}")
                lines.append(f"**Sub-Verdicts Count:** {cv.details.get('sub_verdicts_count', 0)}")
                lines.append("")
                weights = cv.details.get('aggregation_weights', {})
                if weights:
                    lines.append("**Aggregation Weights:**")
                    lines.append("")
                    lines.append("| Aspect | Weight |")
                    lines.append("|--------|--------|")
                    for aspect, w in weights.items():
                        lines.append(f"| {aspect} | {w:.2f} |")
                    lines.append("")
                reasoning = cv.details.get('reasoning', '')
                if reasoning:
                    lines.append(f"**Reasoning:** {reasoning}")
                    lines.append("")

        # Verification Results (V2)
        verification_entries = [e for e in self.entries if e.event_type == "verification_report"]
        if verification_entries:
            lines.append("## Verification Results")
            lines.append("")
            for ver in verification_entries:
                arg_id = ver.details.get('argument_id', 'Unknown')
                passed = ver.details.get('overall_passed', False)
                score = ver.details.get('overall_score', 0)
                status_icon = "Passed" if passed else "Failed"
                lines.append(f"### Argument `{arg_id}` - {status_icon}")
                lines.append("")
                lines.append(f"- **Overall Score:** {score:.2f}")
                lines.append(f"- **Passed:** {'Yes' if passed else 'No'}")
                lines.append("")

                results = ver.details.get('results', [])
                if results:
                    lines.append("| Rule Type | Passed | Score | Violations |")
                    lines.append("|-----------|--------|-------|------------|")
                    for r in results:
                        rule_type = r.get('rule_type', 'unknown')
                        r_passed = 'Yes' if r.get('passed') else 'No'
                        r_score = r.get('score', 0)
                        violations = r.get('violations', [])
                        violation_count = len(violations)
                        lines.append(f"| {rule_type} | {r_passed} | {r_score:.2f} | {violation_count} |")
                    lines.append("")

                    # Show violations if any
                    for r in results:
                        violations = r.get('violations', [])
                        if violations:
                            rule_type = r.get('rule_type', 'unknown')
                            lines.append(f"**{rule_type} Violations:**")
                            for v in violations:
                                sev = v.get('severity', 0)
                                desc = v.get('description', 'No description')
                                lines.append(f"- [{sev:.2f}] {desc}")
                            lines.append("")

        md_str = "\n".join(lines)

        if path:
            Path(path).write_text(md_str)
            logger.info("Audit log exported to Markdown", path=str(path))

        return md_str

    def to_html(self, path: Path | str | None = None) -> str:
        """Export audit log as comprehensive styled HTML with full content."""
        # CSS styles
        css = """
        <style>
            :root { --primary: #4a4e69; --success: #2a9d8f; --danger: #e76f51; --warning: #f4a261; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #f5f5f5; line-height: 1.6; }
            .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #1a1a2e; border-bottom: 2px solid var(--primary); padding-bottom: 10px; }
            h2 { color: var(--primary); margin-top: 30px; }
            h3 { color: #22223b; margin-top: 20px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
            h4 { color: #333; margin: 15px 0 10px 0; }
            .metadata { background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 20px 0; }
            .metadata table { width: 100%; border-collapse: collapse; }
            .metadata td { padding: 8px; border-bottom: 1px solid #dee2e6; }
            .metadata td:first-child { font-weight: bold; width: 180px; }
            .turn { background: #fff; border-left: 4px solid var(--primary); padding: 20px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-radius: 4px; }
            .turn.agent-0 { border-color: #2196F3; }
            .turn.agent-1 { border-color: #FF9800; }
            .turn.agent-2 { border-color: #9C27B0; }
            .turn-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #eee; }
            .turn-header h4 { margin: 0; color: #333; }
            .turn-meta { font-size: 0.85em; color: #666; }
            .turn-meta code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
            .turn-content { color: #333; line-height: 1.8; white-space: pre-wrap; font-family: inherit; }
            .evidence { background: #f8f9fa; padding: 15px; border-radius: 4px; margin-top: 15px; }
            .evidence h5 { margin: 0 0 10px 0; color: #555; }
            .evidence-item { padding: 10px; margin: 8px 0; background: white; border-radius: 4px; border-left: 3px solid #4CAF50; }
            .evidence-item .type { font-weight: bold; color: #4CAF50; text-transform: uppercase; font-size: 0.8em; }
            .evidence-item .content { margin: 5px 0; }
            .evidence-item .meta { font-size: 0.85em; color: #666; }
            .causal-links { background: #fff3e0; padding: 15px; border-radius: 4px; margin-top: 15px; }
            .causal-link { padding: 8px; margin: 5px 0; background: white; border-radius: 4px; display: flex; align-items: center; gap: 10px; }
            .causal-link .cause { color: #1976D2; }
            .causal-link .arrow { color: #666; }
            .causal-link .effect { color: #388E3C; }
            .evaluation { background: #e3f2fd; padding: 15px; border-radius: 4px; margin-top: 15px; }
            .evaluation summary { cursor: pointer; font-weight: bold; }
            .evaluation-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 10px 0; }
            .eval-metric { background: white; padding: 10px; border-radius: 4px; text-align: center; }
            .eval-metric .value { font-size: 1.3em; font-weight: bold; color: var(--primary); }
            .eval-metric .label { font-size: 0.8em; color: #666; }
            .strengths-weaknesses { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px; }
            .strengths { background: #e8f5e9; padding: 10px; border-radius: 4px; }
            .weaknesses { background: #ffebee; padding: 10px; border-radius: 4px; }
            .strengths h6, .weaknesses h6 { margin: 0 0 8px 0; }
            .safety-check { padding: 10px 15px; border-radius: 4px; margin: 10px 0; display: flex; align-items: center; gap: 10px; }
            .safety-check.passed { background: #e8f5e9; border-left: 4px solid #4CAF50; }
            .safety-check.warning { background: #fff3e0; border-left: 4px solid #FF9800; }
            .safety-check.failed { background: #ffebee; border-left: 4px solid #f44336; }
            .safety-icon { font-size: 1.2em; }
            .verdict { background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); padding: 25px; border-radius: 8px; margin: 30px 0; }
            .verdict-decision { font-size: 1.6em; font-weight: bold; color: #1b5e20; }
            .verdict-confidence { color: #388E3C; font-size: 1.1em; margin: 10px 0; }
            .verdict-reasoning { background: white; padding: 15px; border-radius: 4px; margin-top: 15px; }
            .scores { display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }
            .score-card { background: white; padding: 20px; border-radius: 8px; text-align: center; flex: 1; min-width: 120px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .score-value { font-size: 2em; font-weight: bold; color: var(--primary); }
            .score-label { color: #666; font-size: 0.95em; margin-top: 5px; }
            .alert { padding: 15px 20px; border-radius: 4px; margin: 15px 0; }
            .alert.high { background: #ffebee; border-left: 4px solid #f44336; }
            .alert.medium { background: #fff3e0; border-left: 4px solid #ff9800; }
            .alert.low { background: #e3f2fd; border-left: 4px solid #2196F3; }
            .alert-header { font-weight: bold; display: flex; align-items: center; gap: 10px; }
            .alert-details { margin-top: 10px; font-size: 0.95em; }
            .rebuts-supports { margin-top: 10px; font-size: 0.9em; }
            .rebuts-supports span { display: inline-block; padding: 3px 8px; border-radius: 3px; margin: 2px; }
            .rebuts span { background: #ffcdd2; color: #c62828; }
            .supports span { background: #c8e6c9; color: #2e7d32; }
            details summary { cursor: pointer; padding: 8px; background: #f5f5f5; border-radius: 4px; }
            details[open] summary { margin-bottom: 10px; }
            .toc { background: #f8f9fa; padding: 15px 20px; border-radius: 4px; margin: 20px 0; }
            .toc h4 { margin: 0 0 10px 0; }
            .toc ul { margin: 0; padding-left: 20px; }
            .toc li { margin: 5px 0; }
            .toc a { color: var(--primary); text-decoration: none; }
            .toc a:hover { text-decoration: underline; }
            /* V2 Features Styles */
            .v2-config { background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%); padding: 20px; border-radius: 8px; margin: 20px 0; }
            .v2-config h3 { color: #3f51b5; margin-top: 15px; }
            .v2-config h3:first-child { margin-top: 0; }
            .config-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }
            .config-card { background: white; padding: 15px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            .config-card h4 { margin: 0 0 10px 0; color: #3f51b5; font-size: 0.95em; }
            .config-item { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #eee; }
            .config-item:last-child { border-bottom: none; }
            .config-label { color: #666; }
            .config-value { font-weight: bold; color: #333; }
            .steering-bar { height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden; margin-top: 4px; }
            .steering-bar-fill { height: 100%; background: linear-gradient(90deg, #3f51b5, #7986cb); transition: width 0.3s; }
            .hierarchical { background: linear-gradient(135deg, #e0f2f1 0%, #b2dfdb 100%); padding: 20px; border-radius: 8px; margin: 20px 0; }
            .hierarchical h2 { color: #00695c; }
            .sub-debate { background: white; padding: 15px; border-radius: 6px; margin: 10px 0; border-left: 4px solid #26a69a; }
            .sub-debate h4 { margin: 0 0 10px 0; color: #00695c; }
            .sub-debate-meta { font-size: 0.9em; color: #666; }
            .compound-verdict { background: #004d40; color: white; padding: 20px; border-radius: 8px; margin: 15px 0; }
            .compound-verdict h3 { color: #80cbc4; margin-top: 0; }
            .compound-verdict .decision { font-size: 1.4em; font-weight: bold; }
            .verification { background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%); padding: 20px; border-radius: 8px; margin: 20px 0; }
            .verification h2 { color: #f57f17; }
            .verification-report { background: white; padding: 15px; border-radius: 6px; margin: 10px 0; }
            .verification-report.passed { border-left: 4px solid #4CAF50; }
            .verification-report.failed { border-left: 4px solid #f44336; }
            .verification-header { display: flex; justify-content: space-between; align-items: center; }
            .verification-score { font-size: 1.3em; font-weight: bold; }
            .verification-score.passed { color: #4CAF50; }
            .verification-score.failed { color: #f44336; }
            .violation-list { margin-top: 10px; }
            .violation { background: #ffebee; padding: 8px 12px; border-radius: 4px; margin: 5px 0; font-size: 0.9em; }
            .violation-severity { color: #c62828; font-weight: bold; }
            .evidence-multimodal { display: flex; align-items: center; gap: 8px; }
            .evidence-type-badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.75em; font-weight: bold; }
            .evidence-type-badge.image { background: #e3f2fd; color: #1976D2; }
            .evidence-type-badge.document { background: #fce4ec; color: #c2185b; }
            .evidence-type-badge.text { background: #e8f5e9; color: #388E3C; }
        </style>
        """

        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>Debate Audit: {html.escape(self.topic[:50])}</title>",
            css,
            "</head>",
            "<body>",
            "<div class='container'>",
            "<h1>Debate Audit Report</h1>",
            f"<p><strong>Topic:</strong> {html.escape(self.topic)}</p>",
            f"<p><strong>Debate ID:</strong> <code>{self.debate_id}</code></p>",
        ]

        # Check for v2 features
        steering_config = self.metadata.get('steering_config')
        hierarchical_config = self.metadata.get('hierarchical_config')
        verification_spec = self.metadata.get('verification_spec')
        has_v2_config = steering_config or hierarchical_config or verification_spec
        sub_debate_entries = [e for e in self.entries if e.event_type == "sub_debate"]
        compound_verdict_entries = [e for e in self.entries if e.event_type == "compound_verdict"]
        verification_entries = [e for e in self.entries if e.event_type == "verification_report"]

        # Table of contents
        html_parts.append("<div class='toc'><h4>Contents</h4><ul>")
        html_parts.append("<li><a href='#metadata'>Metadata</a></li>")
        if has_v2_config:
            html_parts.append("<li><a href='#v2-config'>V2 Features Configuration</a></li>")
        html_parts.append("<li><a href='#transcript'>Full Transcript</a></li>")
        html_parts.append("<li><a href='#verdict'>Verdict</a></li>")
        if sub_debate_entries or compound_verdict_entries:
            html_parts.append("<li><a href='#hierarchical'>Hierarchical Debate Results</a></li>")
        if verification_entries:
            html_parts.append("<li><a href='#verification'>Verification Results</a></li>")
        html_parts.append("<li><a href='#safety'>Safety Analysis</a></li>")
        html_parts.append("</ul></div>")

        # Metadata section
        html_parts.append("<div class='metadata' id='metadata'><h2>Metadata</h2><table>")
        html_parts.append(f"<tr><td>Started</td><td>{self.metadata.get('started_at', 'N/A')}</td></tr>")
        html_parts.append(f"<tr><td>Ended</td><td>{self.metadata.get('ended_at', 'N/A')}</td></tr>")
        html_parts.append(f"<tr><td>Total Rounds</td><td>{self.metadata.get('total_rounds', 'N/A')}</td></tr>")
        html_parts.append(f"<tr><td>Total Turns</td><td>{self.metadata.get('total_turns', 'N/A')}</td></tr>")
        html_parts.append(f"<tr><td>Agents</td><td>{', '.join(self.metadata.get('agents', []))}</td></tr>")
        html_parts.append(f"<tr><td>Jury Size</td><td>{self.metadata.get('jury_size', 'N/A')}</td></tr>")
        monitors = self.metadata.get('safety_monitors', [])
        html_parts.append(f"<tr><td>Safety Monitors</td><td>{', '.join(monitors) if monitors else 'None'}</td></tr>")
        html_parts.append("</table></div>")

        # V2 Features Configuration
        if has_v2_config:
            html_parts.append("<div class='v2-config' id='v2-config'>")
            html_parts.append("<h2>V2 Features Configuration</h2>")
            html_parts.append("<div class='config-grid'>")

            if steering_config:
                html_parts.append("<div class='config-card'>")
                html_parts.append("<h4>Steering Configuration</h4>")
                vector = steering_config.get('vector', {})
                for dim, val in vector.items():
                    pct = val * 100
                    html_parts.append(f"<div class='config-item'><span class='config-label'>{html.escape(dim)}</span><span class='config-value'>{val:.2f}</span></div>")
                    html_parts.append(f"<div class='steering-bar'><div class='steering-bar-fill' style='width: {pct:.0f}%'></div></div>")
                mode = steering_config.get('mode', 'prompt')
                strength = steering_config.get('strength', 1.0)
                adaptive = steering_config.get('adaptive', False)
                html_parts.append(f"<div class='config-item'><span class='config-label'>Mode</span><span class='config-value'>{html.escape(str(mode))}</span></div>")
                html_parts.append(f"<div class='config-item'><span class='config-label'>Strength</span><span class='config-value'>{strength:.2f}</span></div>")
                html_parts.append(f"<div class='config-item'><span class='config-label'>Adaptive</span><span class='config-value'>{'Yes' if adaptive else 'No'}</span></div>")
                html_parts.append("</div>")

            if hierarchical_config:
                html_parts.append("<div class='config-card'>")
                html_parts.append("<h4>Hierarchical Debate</h4>")
                max_depth = hierarchical_config.get('max_depth', 2)
                decomposer = hierarchical_config.get('decomposer', 'LLM')
                aggregator = hierarchical_config.get('aggregator', 'weighted_average')
                html_parts.append(f"<div class='config-item'><span class='config-label'>Max Depth</span><span class='config-value'>{max_depth}</span></div>")
                html_parts.append(f"<div class='config-item'><span class='config-label'>Decomposer</span><span class='config-value'>{html.escape(str(decomposer))}</span></div>")
                html_parts.append(f"<div class='config-item'><span class='config-label'>Aggregator</span><span class='config-value'>{html.escape(str(aggregator))}</span></div>")
                html_parts.append("</div>")

            if verification_spec:
                html_parts.append("<div class='config-card'>")
                html_parts.append("<h4>Verification Specification</h4>")
                strict_mode = verification_spec.get('strict_mode', False)
                min_score = verification_spec.get('min_score', 0.6)
                rules = verification_spec.get('rules', [])
                html_parts.append(f"<div class='config-item'><span class='config-label'>Strict Mode</span><span class='config-value'>{'Yes' if strict_mode else 'No'}</span></div>")
                html_parts.append(f"<div class='config-item'><span class='config-label'>Min Score</span><span class='config-value'>{min_score:.2f}</span></div>")
                html_parts.append(f"<div class='config-item'><span class='config-label'>Rules</span><span class='config-value'>{len(rules)}</span></div>")
                html_parts.append("</div>")

            html_parts.append("</div>")  # config-grid
            html_parts.append("</div>")  # v2-config

        # Full Transcript
        html_parts.append("<h2 id='transcript'>Full Transcript</h2>")

        current_round = None
        agents_list = self.metadata.get('agents', [])
        agent_index_map = {a: i for i, a in enumerate(agents_list)}

        for entry in self.entries:
            if entry.event_type == "argument_generated":
                if entry.round != current_round:
                    current_round = entry.round
                    round_label = "Opening Statements" if current_round == 0 else f"Round {current_round}"
                    html_parts.append(f"<h3>{round_label}</h3>")

                agent_idx = agent_index_map.get(entry.agent, 0)
                turn_id = entry.details.get('turn_id', 'N/A')
                level = entry.details.get('level', 'N/A')
                ethical_score = entry.details.get('ethical_score')

                html_parts.append(f"<div class='turn agent-{agent_idx}'>")
                html_parts.append("<div class='turn-header'>")
                html_parts.append(f"<h4>{html.escape(entry.agent or 'Unknown')}</h4>")
                ethical_str = f" | Ethical: {ethical_score:.2f}" if ethical_score else ""
                html_parts.append(f"<div class='turn-meta'>Level: <strong>{level}</strong>{ethical_str} | ID: <code>{turn_id}</code></div>")
                html_parts.append("</div>")

                # FULL argument content
                content = entry.details.get('content', '')
                content_html = _md_to_html(content)
                html_parts.append(f"<div class='turn-content'>{content_html}</div>")

                # Rebuts/Supports
                rebuts = entry.details.get('rebuts', [])
                supports = entry.details.get('supports', [])
                if rebuts or supports:
                    html_parts.append("<div class='rebuts-supports'>")
                    if rebuts:
                        html_parts.append("<div class='rebuts'><strong>Rebuts:</strong> ")
                        for r in rebuts:
                            html_parts.append(f"<span>{html.escape(str(r))}</span>")
                        html_parts.append("</div>")
                    if supports:
                        html_parts.append("<div class='supports'><strong>Supports:</strong> ")
                        for s in supports:
                            html_parts.append(f"<span>{html.escape(str(s))}</span>")
                        html_parts.append("</div>")
                    html_parts.append("</div>")

                # Full Evidence with multimodal support
                evidence = entry.details.get('evidence', [])
                if evidence:
                    html_parts.append(f"<div class='evidence'><h5>Evidence ({len(evidence)})</h5>")
                    for e in evidence:
                        verified = " (Verified)" if e.get('verified') else ""
                        content_type = e.get('content_type', 'text')
                        badge_class = 'text'
                        if content_type == 'image':
                            badge_class = 'image'
                        elif content_type == 'document':
                            badge_class = 'document'
                        html_parts.append("<div class='evidence-item'>")
                        html_parts.append("<div class='evidence-multimodal'>")
                        html_parts.append(f"<div class='type'>{html.escape(str(e.get('type', 'Unknown')))}</div>")
                        if content_type and content_type != 'text':
                            html_parts.append(f"<span class='evidence-type-badge {badge_class}'>{html.escape(content_type)}</span>")
                        html_parts.append("</div>")
                        html_parts.append(f"<div class='content'>{html.escape(str(e.get('content', '')))}</div>")
                        # Build meta string with multimodal fields
                        meta_parts = [f"Source: {html.escape(str(e.get('source', 'N/A')))}"]
                        meta_parts.append(f"Confidence: {e.get('confidence', 0):.2f}{verified}")
                        if e.get('url'):
                            meta_parts.append(f"URL: {html.escape(str(e.get('url')))}")
                        if e.get('media_type'):
                            meta_parts.append(f"Media: {html.escape(str(e.get('media_type')))}")
                        if e.get('filename'):
                            meta_parts.append(f"File: {html.escape(str(e.get('filename')))}")
                        html_parts.append(f"<div class='meta'>{' | '.join(meta_parts)}</div>")
                        if e.get('alt_text'):
                            html_parts.append(f"<div class='meta'><em>Alt: {html.escape(str(e.get('alt_text')))}</em></div>")
                        html_parts.append("</div>")
                    html_parts.append("</div>")

                # Causal Links
                causal_links = entry.details.get('causal_links', [])
                if causal_links:
                    html_parts.append(f"<div class='causal-links'><h5>Causal Links ({len(causal_links)})</h5>")
                    for link in causal_links:
                        strength = link.get('strength')
                        strength_str = f" (strength: {strength})" if strength else ""
                        html_parts.append("<div class='causal-link'>")
                        html_parts.append(f"<span class='cause'>{html.escape(str(link.get('cause', '')))}</span>")
                        html_parts.append("<span class='arrow'>→</span>")
                        html_parts.append(f"<span class='effect'>{html.escape(str(link.get('effect', '')))}</span>")
                        html_parts.append(f"<span class='meta'>{strength_str}</span>")
                        html_parts.append("</div>")
                    html_parts.append("</div>")

                html_parts.append("</div>")  # Close turn

            # Evaluation details
            elif entry.event_type == "argument_evaluated":
                total_score = entry.details.get('total_score', 0)
                html_parts.append("<div class='evaluation'>")
                html_parts.append(f"<details><summary>Evaluation Score: {total_score:.2f}</summary>")

                # Score grid
                scores = entry.details.get('scores', {})
                weights = entry.details.get('weights', {})
                if scores:
                    html_parts.append("<div class='evaluation-grid'>")
                    for criterion, score in scores.items():
                        weight = weights.get(criterion, 0)
                        html_parts.append(f"<div class='eval-metric'><div class='value'>{score:.2f}</div><div class='label'>{html.escape(criterion)} (w: {weight:.2f})</div></div>")
                    html_parts.append("</div>")

                # Feedback
                feedback = entry.details.get('feedback')
                if feedback:
                    html_parts.append(f"<p><strong>Feedback:</strong> {html.escape(feedback)}</p>")

                # Strengths and Weaknesses
                strengths = entry.details.get('strengths', [])
                weaknesses = entry.details.get('weaknesses', [])
                if strengths or weaknesses:
                    html_parts.append("<div class='strengths-weaknesses'>")
                    if strengths:
                        html_parts.append("<div class='strengths'><h6>Strengths</h6><ul>")
                        for s in strengths:
                            html_parts.append(f"<li>{html.escape(str(s))}</li>")
                        html_parts.append("</ul></div>")
                    if weaknesses:
                        html_parts.append("<div class='weaknesses'><h6>Weaknesses</h6><ul>")
                        for w in weaknesses:
                            html_parts.append(f"<li>{html.escape(str(w))}</li>")
                        html_parts.append("</ul></div>")
                    html_parts.append("</div>")

                html_parts.append("</details></div>")

            # Safety check inline
            elif entry.event_type == "safety_check":
                entry.details.get('passed', True)
                severity = entry.details.get('severity', 0)
                monitor = entry.details.get('monitor', 'Unknown')
                notes = entry.details.get('analysis_notes', '')

                if severity > 0.3:
                    if severity > 0.7:
                        css_class = "failed"
                        icon = "⛔"
                    elif severity > 0.3:
                        css_class = "warning"
                        icon = "⚠️"
                    else:
                        css_class = "passed"
                        icon = "✅"

                    html_parts.append(f"<div class='safety-check {css_class}'>")
                    html_parts.append(f"<span class='safety-icon'>{icon}</span>")
                    html_parts.append(f"<div><strong>{html.escape(monitor)}</strong> - Severity: {severity:.2f}")
                    if notes:
                        html_parts.append(f"<br><small>{html.escape(notes)}</small>")
                    html_parts.append("</div></div>")

        # Verdict
        verdict_entries = [e for e in self.entries if e.event_type == "verdict_issued"]
        if verdict_entries:
            verdict = verdict_entries[0]
            html_parts.append("<div class='verdict' id='verdict'>")
            html_parts.append("<h2>Verdict</h2>")
            html_parts.append(f"<div class='verdict-decision'>{html.escape(str(verdict.details.get('decision', 'N/A')))}</div>")
            confidence = verdict.details.get('confidence', 0)
            unanimous = verdict.details.get('unanimous', False)
            html_parts.append(f"<div class='verdict-confidence'>Confidence: {confidence:.0%} | {'Unanimous' if unanimous else 'Not Unanimous'}</div>")

            reasoning = verdict.details.get('reasoning', '')
            if reasoning:
                html_parts.append(f"<div class='verdict-reasoning'><h4>Reasoning</h4><p>{html.escape(reasoning)}</p></div>")

            scores = verdict.details.get('score_breakdown', {})
            if scores:
                html_parts.append("<h4>Final Scores</h4><div class='scores'>")
                for agent, score in scores.items():
                    html_parts.append(f"<div class='score-card'><div class='score-value'>{score:.2f}</div><div class='score-label'>{html.escape(agent)}</div></div>")
                html_parts.append("</div>")

            dissenting = verdict.details.get('dissenting_count', 0)
            if dissenting > 0:
                html_parts.append(f"<p><em>{dissenting} juror(s) dissented.</em></p>")

            html_parts.append("</div>")

        # Hierarchical Debate Results (V2)
        if sub_debate_entries or compound_verdict_entries:
            html_parts.append("<div class='hierarchical' id='hierarchical'>")
            html_parts.append("<h2>Hierarchical Debate Results</h2>")

            if sub_debate_entries:
                html_parts.append("<h3>Sub-Debates</h3>")
                for i, sub in enumerate(sub_debate_entries, 1):
                    aspect = sub.details.get('aspect', f'Sub-debate {i}')
                    weight = sub.details.get('weight', 1.0)
                    sub_verdict = sub.details.get('verdict', {})
                    html_parts.append("<div class='sub-debate'>")
                    html_parts.append(f"<h4>{i}. {html.escape(str(aspect))}</h4>")
                    html_parts.append(f"<div class='sub-debate-meta'>Weight: {weight:.2f}</div>")
                    if sub_verdict:
                        decision = sub_verdict.get('decision', 'N/A')
                        confidence = sub_verdict.get('confidence', 0)
                        reasoning = sub_verdict.get('reasoning', '')
                        html_parts.append(f"<p><strong>Decision:</strong> {html.escape(str(decision))} ({confidence:.0%} confidence)</p>")
                        if reasoning:
                            html_parts.append(f"<p><em>{html.escape(str(reasoning)[:200])}...</em></p>")
                    html_parts.append("</div>")

            if compound_verdict_entries:
                cv = compound_verdict_entries[0]
                html_parts.append("<div class='compound-verdict'>")
                html_parts.append("<h3>Compound Verdict (Aggregated)</h3>")
                final_decision = cv.details.get('final_decision', 'N/A')
                confidence = cv.details.get('confidence', 0)
                method = cv.details.get('aggregation_method', 'N/A')
                sub_count = cv.details.get('sub_verdicts_count', 0)
                html_parts.append(f"<div class='decision'>{html.escape(str(final_decision))}</div>")
                html_parts.append(f"<p>Confidence: {confidence:.0%} | Method: {html.escape(str(method))} | Sub-verdicts: {sub_count}</p>")
                weights = cv.details.get('aggregation_weights', {})
                if weights:
                    html_parts.append("<p><strong>Aggregation Weights:</strong></p><ul>")
                    for aspect, w in weights.items():
                        html_parts.append(f"<li>{html.escape(str(aspect))}: {w:.2f}</li>")
                    html_parts.append("</ul>")
                reasoning = cv.details.get('reasoning', '')
                if reasoning:
                    html_parts.append(f"<p><em>{html.escape(str(reasoning))}</em></p>")
                html_parts.append("</div>")

            html_parts.append("</div>")  # hierarchical

        # Verification Results (V2)
        if verification_entries:
            html_parts.append("<div class='verification' id='verification'>")
            html_parts.append("<h2>Verification Results</h2>")
            for ver in verification_entries:
                arg_id = ver.details.get('argument_id', 'Unknown')
                passed = ver.details.get('overall_passed', False)
                score = ver.details.get('overall_score', 0)
                status_class = "passed" if passed else "failed"
                html_parts.append(f"<div class='verification-report {status_class}'>")
                html_parts.append("<div class='verification-header'>")
                html_parts.append(f"<h4>Argument <code>{html.escape(str(arg_id))}</code></h4>")
                html_parts.append(f"<div class='verification-score {status_class}'>{score:.2f}</div>")
                html_parts.append("</div>")

                results = ver.details.get('results', [])
                if results:
                    html_parts.append("<table style='width:100%; margin-top:10px; border-collapse:collapse;'>")
                    html_parts.append("<tr><th style='text-align:left; padding:5px; border-bottom:1px solid #ddd;'>Rule</th><th style='padding:5px; border-bottom:1px solid #ddd;'>Passed</th><th style='padding:5px; border-bottom:1px solid #ddd;'>Score</th></tr>")
                    for r in results:
                        rule_type = r.get('rule_type', 'unknown')
                        r_passed = r.get('passed', False)
                        r_score = r.get('score', 0)
                        passed_icon = "✅" if r_passed else "❌"
                        html_parts.append(f"<tr><td style='padding:5px;'>{html.escape(str(rule_type))}</td><td style='text-align:center; padding:5px;'>{passed_icon}</td><td style='text-align:center; padding:5px;'>{r_score:.2f}</td></tr>")
                    html_parts.append("</table>")

                    # Show violations
                    all_violations = []
                    for r in results:
                        for v in r.get('violations', []):
                            all_violations.append((r.get('rule_type', 'unknown'), v))

                    if all_violations:
                        html_parts.append("<div class='violation-list'>")
                        html_parts.append("<h5>Violations</h5>")
                        for rule_type, v in all_violations:
                            sev = v.get('severity', 0)
                            desc = v.get('description', 'No description')
                            html_parts.append(f"<div class='violation'><span class='violation-severity'>[{sev:.2f}]</span> <strong>{html.escape(str(rule_type))}:</strong> {html.escape(str(desc))}</div>")
                        html_parts.append("</div>")

                html_parts.append("</div>")  # verification-report
            html_parts.append("</div>")  # verification

        # Safety Alerts Section
        alert_entries = [e for e in self.entries if e.event_type == "safety_alert"]
        safety_checks = [e for e in self.entries if e.event_type == "safety_check"]

        html_parts.append("<h2 id='safety'>Safety Analysis</h2>")

        if not alert_entries and not any(e.details.get('severity', 0) > 0.3 for e in safety_checks):
            html_parts.append("<p style='color: #388E3C;'>✅ No safety concerns detected during this debate.</p>")
        else:
            if alert_entries:
                html_parts.append("<h3>Critical Alerts</h3>")
                for alert in alert_entries:
                    severity = alert.details.get('severity', 0)
                    severity_class = "high" if severity > 0.7 else "medium" if severity > 0.4 else "low"
                    resolved = alert.details.get('resolved', False)

                    html_parts.append(f"<div class='alert {severity_class}'>")
                    html_parts.append("<div class='alert-header'>")
                    html_parts.append(f"<span>{'🔴' if severity > 0.7 else '🟡' if severity > 0.4 else '🔵'}</span>")
                    html_parts.append(f"<strong>{html.escape(str(alert.details.get('type', 'Unknown')))}</strong>")
                    html_parts.append(f"<span> - {html.escape(str(alert.agent))}</span>")
                    html_parts.append("</div>")

                    html_parts.append("<div class='alert-details'>")
                    html_parts.append(f"<p><strong>Monitor:</strong> {html.escape(str(alert.details.get('monitor', 'N/A')))}</p>")
                    html_parts.append(f"<p><strong>Severity:</strong> {severity:.2f}</p>")
                    html_parts.append(f"<p><strong>Status:</strong> {'Resolved' if resolved else 'Unresolved'}</p>")

                    indicators = alert.details.get('indicators', [])
                    if indicators:
                        html_parts.append("<p><strong>Indicators:</strong></p><ul>")
                        for ind in indicators:
                            html_parts.append(f"<li>{html.escape(str(ind.get('type', '')))}: {html.escape(str(ind.get('evidence', '')))}</li>")
                        html_parts.append("</ul>")

                    html_parts.append("</div></div>")

        # Close document
        html_parts.extend([
            "</div>",
            "<script>",
            "// Auto-expand details if there are safety issues",
            "document.querySelectorAll('.evaluation details').forEach(d => {",
            "  if(d.querySelector('.weaknesses li')) d.open = true;",
            "});",
            "</script>",
            "</body>",
            "</html>",
        ])

        html_str = "\n".join(html_parts)

        if path:
            Path(path).write_text(html_str)
            logger.info("Audit log exported to HTML", path=str(path))

        return html_str


def export_debate_audit(
    result: DebateResult,
    output_dir: Path | str | None = None,
    formats: list[str] | None = None,
) -> dict[str, str]:
    """
    Export debate result to audit log files.

    Args:
        result: The debate result to export.
        output_dir: Directory to save files. If None, returns strings only.
        formats: List of formats to export. Options: 'json', 'markdown', 'html'.
                 Defaults to all formats.

    Returns:
        Dict mapping format to content (or file path if output_dir provided).
    """
    formats = formats or ["json", "markdown", "html"]
    audit_log = AuditLog.from_debate_result(result)

    outputs = {}

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"debate_{result.debate_id}"

        if "json" in formats:
            path = output_dir / f"{base_name}.json"
            audit_log.to_json(path)
            outputs["json"] = str(path)

        if "markdown" in formats:
            path = output_dir / f"{base_name}.md"
            audit_log.to_markdown(path)
            outputs["markdown"] = str(path)

        if "html" in formats:
            path = output_dir / f"{base_name}.html"
            audit_log.to_html(path)
            outputs["html"] = str(path)
    else:
        if "json" in formats:
            outputs["json"] = audit_log.to_json()
        if "markdown" in formats:
            outputs["markdown"] = audit_log.to_markdown()
        if "html" in formats:
            outputs["html"] = audit_log.to_html()

    return outputs
