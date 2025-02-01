"""
ARTEMIS Exceptions

Custom exception hierarchy for the ARTEMIS framework.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from artemis.core.types import SafetyAlert


class ArtemisError(Exception):
    """Base exception for all ARTEMIS errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details: dict[str, Any] = details or {}
        super().__init__(message)


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(ArtemisError):
    """Invalid configuration provided."""

    pass


class DebateConfigError(ConfigurationError):
    """Invalid debate configuration."""

    pass


class ModelConfigError(ConfigurationError):
    """Invalid model configuration."""

    pass


# =============================================================================
# Agent Errors
# =============================================================================


class AgentError(ArtemisError):
    """Agent-related error."""

    def __init__(self, message: str, agent_name: str | None = None, **kwargs: Any) -> None:
        self.agent_name = agent_name
        super().__init__(message, details={"agent": agent_name, **kwargs})


class ArgumentGenerationError(AgentError):
    """Failed to generate argument."""

    pass


class ReasoningError(AgentError):
    """Error during reasoning/thinking phase."""

    pass


# =============================================================================
# Model/Provider Errors
# =============================================================================


class ModelError(ArtemisError):
    """LLM provider error."""

    def __init__(self, message: str, provider: str | None = None, **kwargs: Any) -> None:
        self.provider = provider
        super().__init__(message, details={"provider": provider, **kwargs})


class ModelNotFoundError(ModelError):
    """Requested model not found or not available."""

    pass


class RateLimitError(ModelError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: float | None = None, **kwargs: Any) -> None:
        self.retry_after = retry_after
        super().__init__(message, details={"retry_after": retry_after, **kwargs})


class TokenLimitError(ModelError):
    """Token limit exceeded."""

    def __init__(
        self,
        message: str,
        tokens_used: int | None = None,
        tokens_limit: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.tokens_used = tokens_used
        self.tokens_limit = tokens_limit
        super().__init__(
            message,
            details={
                "tokens_used": tokens_used,
                "tokens_limit": tokens_limit,
                **kwargs,
            },
        )


class ProviderConnectionError(ModelError):
    """Failed to connect to model provider."""

    pass


# =============================================================================
# Evaluation Errors
# =============================================================================


class EvaluationError(ArtemisError):
    """Evaluation error."""

    pass


class CriteriaError(EvaluationError):
    """Invalid or missing evaluation criteria."""

    pass


class CausalGraphError(EvaluationError):
    """Error in causal graph construction or analysis."""

    pass


# =============================================================================
# Jury Errors
# =============================================================================


class JuryError(ArtemisError):
    """Jury-related error."""

    pass


class DeliberationError(JuryError):
    """Error during jury deliberation."""

    pass


class ConsensusError(JuryError):
    """Failed to reach consensus."""

    def __init__(self, message: str, dissent_ratio: float | None = None, **kwargs: Any) -> None:
        self.dissent_ratio = dissent_ratio
        super().__init__(message, details={"dissent_ratio": dissent_ratio, **kwargs})


# =============================================================================
# Safety Errors
# =============================================================================


class SafetyError(ArtemisError):
    """Safety monitoring error."""

    pass


class SafetyHaltError(SafetyError):
    """Debate halted due to safety violation."""

    def __init__(self, message: str, alert: "SafetyAlert | None" = None):
        self.alert = alert
        details = {}
        if alert:
            details = {
                "alert_id": alert.id,
                "agent": alert.agent,
                "type": alert.type,
                "severity": alert.severity,
            }
        super().__init__(message, details=details)


class MonitorError(SafetyError):
    """Error in safety monitor execution."""

    def __init__(self, message: str, monitor: str | None = None, **kwargs: Any) -> None:
        self.monitor = monitor
        super().__init__(message, details={"monitor": monitor, **kwargs})


# =============================================================================
# Debate Lifecycle Errors
# =============================================================================


class DebateError(ArtemisError):
    """Debate orchestration error."""

    pass


class StateTransitionError(DebateError):
    """Invalid state transition attempted."""

    def __init__(
        self,
        message: str,
        from_state: str | None = None,
        to_state: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            message,
            details={"from_state": from_state, "to_state": to_state, **kwargs},
        )


class TimeoutError(DebateError):
    """Operation timed out."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(
            message,
            details={
                "operation": operation,
                "timeout_seconds": timeout_seconds,
                **kwargs,
            },
        )


class RoundError(DebateError):
    """Error during debate round execution."""

    def __init__(self, message: str, round_number: int | None = None, **kwargs: Any) -> None:
        self.round_number = round_number
        super().__init__(message, details={"round_number": round_number, **kwargs})


# =============================================================================
# Integration Errors
# =============================================================================


class IntegrationError(ArtemisError):
    """Framework integration error."""

    def __init__(self, message: str, framework: str | None = None, **kwargs: Any) -> None:
        self.framework = framework
        super().__init__(message, details={"framework": framework, **kwargs})


class MCPError(IntegrationError):
    """MCP server/protocol error."""

    pass
