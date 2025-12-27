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

            # Add evaluation entry if present
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
                    },
                ))

            # Add safety results if any
            for safety in turn.safety_results:
                if safety.severity > 0:
                    entries.append(AuditEntry(
                        timestamp=turn.timestamp,
                        debate_id=result.debate_id,
                        event_type="safety_check",
                        agent=turn.agent,
                        round=turn.round,
                        details={
                            "monitor": safety.monitor,
                            "severity": safety.severity,
                            "should_alert": safety.should_alert,
                            "notes": safety.analysis_notes,
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
        """Convert a turn to an audit entry."""
        evidence_summary = []
        for e in turn.argument.evidence:
            evidence_summary.append({
                "type": e.type,
                "content": e.content[:100] + "..." if len(e.content) > 100 else e.content,
                "source": e.source,
                "confidence": e.confidence,
            })

        return AuditEntry(
            timestamp=turn.timestamp,
            debate_id=debate_id,
            event_type="argument_generated",
            agent=turn.agent,
            round=turn.round,
            details={
                "level": turn.argument.level.value,
                "content_length": len(turn.argument.content),
                "content_preview": turn.argument.content[:200] + "..." if len(turn.argument.content) > 200 else turn.argument.content,
                "evidence_count": len(turn.argument.evidence),
                "evidence": evidence_summary,
                "causal_links_count": len(turn.argument.causal_links),
                "rebuts": turn.argument.rebuts,
                "supports": turn.argument.supports,
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
        """Export audit log as Markdown."""
        lines = []

        # Header
        lines.append(f"# Debate Audit Log")
        lines.append(f"")
        lines.append(f"**Topic:** {self.topic}")
        lines.append(f"**Debate ID:** `{self.debate_id}`")
        lines.append(f"")

        # Metadata
        lines.append(f"## Metadata")
        lines.append(f"")
        lines.append(f"| Field | Value |")
        lines.append(f"|-------|-------|")
        lines.append(f"| Started | {self.metadata.get('started_at', 'N/A')} |")
        lines.append(f"| Ended | {self.metadata.get('ended_at', 'N/A')} |")
        lines.append(f"| Rounds | {self.metadata.get('total_rounds', 'N/A')} |")
        lines.append(f"| Turns | {self.metadata.get('total_turns', 'N/A')} |")
        lines.append(f"| Agents | {', '.join(self.metadata.get('agents', []))} |")
        lines.append(f"| Jury Size | {self.metadata.get('jury_size', 'N/A')} |")
        lines.append(f"")

        # Transcript
        lines.append(f"## Transcript")
        lines.append(f"")

        current_round = None
        for entry in self.entries:
            if entry.event_type == "argument_generated":
                if entry.round != current_round:
                    current_round = entry.round
                    round_label = "Opening" if current_round == 0 else f"Round {current_round}"
                    lines.append(f"### {round_label}")
                    lines.append(f"")

                lines.append(f"**{entry.agent}** ({entry.details.get('level', 'N/A')})")
                lines.append(f"")
                lines.append(f"> {entry.details.get('content_preview', '')}")
                lines.append(f"")

                # Evidence
                evidence = entry.details.get('evidence', [])
                if evidence:
                    lines.append(f"*Evidence ({len(evidence)}):*")
                    for e in evidence[:3]:  # Limit to 3
                        lines.append(f"- [{e.get('type')}] {e.get('content')} (source: {e.get('source', 'N/A')})")
                    lines.append(f"")

        # Verdict
        verdict_entries = [e for e in self.entries if e.event_type == "verdict_issued"]
        if verdict_entries:
            verdict = verdict_entries[0]
            lines.append(f"## Verdict")
            lines.append(f"")
            lines.append(f"**Decision:** {verdict.details.get('decision', 'N/A')}")
            lines.append(f"**Confidence:** {verdict.details.get('confidence', 0):.0%}")
            lines.append(f"**Unanimous:** {'Yes' if verdict.details.get('unanimous') else 'No'}")
            lines.append(f"")
            lines.append(f"### Reasoning")
            lines.append(f"")
            lines.append(f"{verdict.details.get('reasoning', '')}")
            lines.append(f"")

            # Score breakdown
            scores = verdict.details.get('score_breakdown', {})
            if scores:
                lines.append(f"### Score Breakdown")
                lines.append(f"")
                lines.append(f"| Agent | Score |")
                lines.append(f"|-------|-------|")
                for agent, score in scores.items():
                    lines.append(f"| {agent} | {score:.2f} |")
                lines.append(f"")

        # Safety Alerts
        alert_entries = [e for e in self.entries if e.event_type == "safety_alert"]
        if alert_entries:
            lines.append(f"## Safety Alerts")
            lines.append(f"")
            for alert in alert_entries:
                severity = alert.details.get('severity', 0)
                severity_label = "HIGH" if severity > 0.7 else "MEDIUM" if severity > 0.4 else "LOW"
                lines.append(f"### [{severity_label}] {alert.details.get('type', 'Unknown')}")
                lines.append(f"")
                lines.append(f"- **Agent:** {alert.agent}")
                lines.append(f"- **Monitor:** {alert.details.get('monitor', 'N/A')}")
                lines.append(f"- **Severity:** {severity:.2f}")
                lines.append(f"- **Resolved:** {'Yes' if alert.details.get('resolved') else 'No'}")
                lines.append(f"")

        md_str = "\n".join(lines)

        if path:
            Path(path).write_text(md_str)
            logger.info("Audit log exported to Markdown", path=str(path))

        return md_str

    def to_html(self, path: Path | str | None = None) -> str:
        """Export audit log as styled HTML."""
        # CSS styles
        css = """
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #1a1a2e; border-bottom: 2px solid #4a4e69; padding-bottom: 10px; }
            h2 { color: #4a4e69; margin-top: 30px; }
            h3 { color: #22223b; }
            .metadata { background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 20px 0; }
            .metadata table { width: 100%; border-collapse: collapse; }
            .metadata td { padding: 8px; border-bottom: 1px solid #dee2e6; }
            .metadata td:first-child { font-weight: bold; width: 150px; }
            .turn { background: #fff; border-left: 4px solid #4a4e69; padding: 15px; margin: 15px 0; }
            .turn.pro { border-color: #2a9d8f; }
            .turn.con { border-color: #e76f51; }
            .turn-header { font-weight: bold; margin-bottom: 10px; }
            .turn-content { color: #333; line-height: 1.6; }
            .evidence { background: #e9ecef; padding: 10px; border-radius: 4px; margin-top: 10px; font-size: 0.9em; }
            .verdict { background: #d4edda; padding: 20px; border-radius: 4px; margin: 20px 0; }
            .verdict-decision { font-size: 1.4em; font-weight: bold; color: #155724; }
            .verdict-confidence { color: #666; }
            .alert { padding: 15px; border-radius: 4px; margin: 10px 0; }
            .alert.high { background: #f8d7da; border-left: 4px solid #dc3545; }
            .alert.medium { background: #fff3cd; border-left: 4px solid #ffc107; }
            .alert.low { background: #d1ecf1; border-left: 4px solid #17a2b8; }
            .scores { display: flex; gap: 20px; margin: 15px 0; }
            .score-card { background: #f8f9fa; padding: 15px; border-radius: 4px; text-align: center; flex: 1; }
            .score-value { font-size: 1.5em; font-weight: bold; color: #4a4e69; }
            .score-label { color: #666; font-size: 0.9em; }
            .collapsible { cursor: pointer; padding: 10px; background: #e9ecef; border: none; width: 100%; text-align: left; font-weight: bold; }
            .collapsible:after { content: '\\25BC'; float: right; }
            .collapsible.active:after { content: '\\25B2'; }
            .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1; }
        </style>
        """

        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>Debate Audit: {self.topic[:50]}</title>",
            css,
            "</head>",
            "<body>",
            "<div class='container'>",
            f"<h1>Debate Audit Log</h1>",
            f"<p><strong>Topic:</strong> {self.topic}</p>",
            f"<p><strong>ID:</strong> <code>{self.debate_id}</code></p>",
        ]

        # Metadata section
        html_parts.append("<div class='metadata'><h2>Metadata</h2><table>")
        html_parts.append(f"<tr><td>Started</td><td>{self.metadata.get('started_at', 'N/A')}</td></tr>")
        html_parts.append(f"<tr><td>Ended</td><td>{self.metadata.get('ended_at', 'N/A')}</td></tr>")
        html_parts.append(f"<tr><td>Rounds</td><td>{self.metadata.get('total_rounds', 'N/A')}</td></tr>")
        html_parts.append(f"<tr><td>Turns</td><td>{self.metadata.get('total_turns', 'N/A')}</td></tr>")
        html_parts.append(f"<tr><td>Agents</td><td>{', '.join(self.metadata.get('agents', []))}</td></tr>")
        html_parts.append(f"<tr><td>Jury Size</td><td>{self.metadata.get('jury_size', 'N/A')}</td></tr>")
        html_parts.append("</table></div>")

        # Transcript
        html_parts.append("<h2>Transcript</h2>")
        current_round = None
        for entry in self.entries:
            if entry.event_type == "argument_generated":
                if entry.round != current_round:
                    current_round = entry.round
                    round_label = "Opening" if current_round == 0 else f"Round {current_round}"
                    html_parts.append(f"<h3>{round_label}</h3>")

                agent_class = "pro" if "pro" in (entry.agent or "").lower() else "con"
                content_html = _md_to_html(entry.details.get('content_preview', ''))
                html_parts.append(f"<div class='turn {agent_class}'>")
                html_parts.append(f"<div class='turn-header'>{entry.agent} ({entry.details.get('level', 'N/A')})</div>")
                html_parts.append(f"<div class='turn-content'>{content_html}</div>")

                evidence = entry.details.get('evidence', [])
                if evidence:
                    html_parts.append(f"<div class='evidence'><strong>Evidence ({len(evidence)}):</strong><ul>")
                    for e in evidence[:3]:
                        html_parts.append(f"<li>[{e.get('type')}] {e.get('content')}</li>")
                    html_parts.append("</ul></div>")
                html_parts.append("</div>")

        # Verdict
        verdict_entries = [e for e in self.entries if e.event_type == "verdict_issued"]
        if verdict_entries:
            verdict = verdict_entries[0]
            html_parts.append("<div class='verdict'>")
            html_parts.append("<h2>Verdict</h2>")
            html_parts.append(f"<div class='verdict-decision'>{verdict.details.get('decision', 'N/A')}</div>")
            html_parts.append(f"<div class='verdict-confidence'>Confidence: {verdict.details.get('confidence', 0):.0%}</div>")
            html_parts.append(f"<p>{verdict.details.get('reasoning', '')}</p>")

            scores = verdict.details.get('score_breakdown', {})
            if scores:
                html_parts.append("<div class='scores'>")
                for agent, score in scores.items():
                    html_parts.append(f"<div class='score-card'><div class='score-value'>{score:.2f}</div><div class='score-label'>{agent}</div></div>")
                html_parts.append("</div>")
            html_parts.append("</div>")

        # Safety Alerts
        alert_entries = [e for e in self.entries if e.event_type == "safety_alert"]
        if alert_entries:
            html_parts.append("<h2>Safety Alerts</h2>")
            for alert in alert_entries:
                severity = alert.details.get('severity', 0)
                severity_class = "high" if severity > 0.7 else "medium" if severity > 0.4 else "low"
                html_parts.append(f"<div class='alert {severity_class}'>")
                html_parts.append(f"<strong>{alert.details.get('type', 'Unknown')}</strong> - {alert.agent}")
                html_parts.append(f"<br>Severity: {severity:.2f} | Monitor: {alert.details.get('monitor', 'N/A')}")
                html_parts.append("</div>")

        # Close
        html_parts.extend([
            "</div>",
            "<script>",
            "document.querySelectorAll('.collapsible').forEach(btn => {",
            "  btn.addEventListener('click', function() {",
            "    this.classList.toggle('active');",
            "    var content = this.nextElementSibling;",
            "    content.style.display = content.style.display === 'block' ? 'none' : 'block';",
            "  });",
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
