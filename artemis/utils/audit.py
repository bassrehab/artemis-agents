"""Audit log exporter for ARTEMIS debates."""

import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from artemis.core.types import DebateResult, SafetyAlert, Turn
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
        # Full evidence details
        evidence_full = []
        for e in turn.argument.evidence:
            evidence_full.append({
                "type": e.type,
                "content": e.content,  # Full content, not truncated
                "source": e.source,
                "confidence": e.confidence,
                "verified": getattr(e, "verified", None),
            })

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

                # Full Evidence
                evidence = entry.details.get('evidence', [])
                if evidence:
                    lines.append(f"**Evidence ({len(evidence)}):**")
                    lines.append("")
                    for i, e in enumerate(evidence, 1):
                        verified = " (verified)" if e.get('verified') else ""
                        lines.append(f"{i}. **[{e.get('type')}]** {e.get('content')}")
                        lines.append(f"   - Source: {e.get('source', 'N/A')}")
                        lines.append(f"   - Confidence: {e.get('confidence', 0):.2f}{verified}")
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

        # Table of contents
        html_parts.append("<div class='toc'><h4>Contents</h4><ul>")
        html_parts.append("<li><a href='#metadata'>Metadata</a></li>")
        html_parts.append("<li><a href='#transcript'>Full Transcript</a></li>")
        html_parts.append("<li><a href='#verdict'>Verdict</a></li>")
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

                # Full Evidence
                evidence = entry.details.get('evidence', [])
                if evidence:
                    html_parts.append(f"<div class='evidence'><h5>Evidence ({len(evidence)})</h5>")
                    for e in evidence:
                        verified = " (Verified)" if e.get('verified') else ""
                        html_parts.append("<div class='evidence-item'>")
                        html_parts.append(f"<div class='type'>{html.escape(str(e.get('type', 'Unknown')))}</div>")
                        html_parts.append(f"<div class='content'>{html.escape(str(e.get('content', '')))}</div>")
                        html_parts.append(f"<div class='meta'>Source: {html.escape(str(e.get('source', 'N/A')))} | Confidence: {e.get('confidence', 0):.2f}{verified}</div>")
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
