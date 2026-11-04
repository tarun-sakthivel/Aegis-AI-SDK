"""
sentinel_sre/core/models.py
─────────────────────────────────────────────────────────────────────────────
Core Pydantic schemas shared across the SDK.

Schemas
-------
UnifiedTelemetrySchema
    Canonical representation of an intercepted error event.  Produced by the
    ASGI middleware and consumed by the LangGraph orchestration layer.

TriageResult
    Structured output of the LLM triage node.  Validated by Pydantic so that
    downstream routing edges can trust the criticality score without
    defensive checks.
"""

from __future__ import annotations

import traceback
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Coarse severity bucket derived from HTTP status or exception class."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TriageDecision(str, Enum):
    """Routing decision emitted by the triage node."""

    REMEDIATE = "remediate"   # criticality >= threshold → action node
    NOTIFY = "notify"          # criticality <  threshold → notification node
    ESCALATE = "escalate"      # LLM returned ambiguous / max-severity result


# ---------------------------------------------------------------------------
# Unified Telemetry Schema
# ---------------------------------------------------------------------------


class HTTPContext(BaseModel):
    """Optional HTTP-layer metadata attached when the error originates from a
    web request."""

    model_config = ConfigDict(frozen=True)

    method: str = Field(..., description="HTTP method (GET, POST, …)")
    path: str = Field(..., description="Request path, e.g. /api/v1/infer")
    status_code: int = Field(..., ge=100, le=599)
    client_host: str | None = Field(default=None)
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Sanitised request headers (auth tokens must be redacted upstream).",
    )
    query_params: dict[str, str] = Field(default_factory=dict)


class UnifiedTelemetrySchema(BaseModel):
    """Canonical error event produced by the ASGI middleware or any SDK
    instrumentation point.

    Design goals
    ────────────
    • Immutable after creation (``frozen=True``) so that graph nodes receive a
      stable snapshot.
    • ``event_id`` is auto-generated; callers should not set it manually.
    • ``formatted_traceback`` is derived automatically from *exc* at
      construction time and is therefore always consistent.
    """

    model_config = ConfigDict(frozen=True)

    # ── Identity ────────────────────────────────────────────────────────────
    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Globally unique identifier for this telemetry event.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Wall-clock time (UTC) at which the exception was caught.",
    )

    # ── Error payload ────────────────────────────────────────────────────────
    exception_type: str = Field(
        ...,
        description="Fully qualified exception class name, e.g. "
        "'builtins.RuntimeError' or 'torch.cuda.OutOfMemoryError'.",
    )
    exception_message: str = Field(
        ...,
        description="str(exception) – the human-readable error message.",
    )
    formatted_traceback: str = Field(
        ...,
        description="Full traceback string produced by traceback.format_exc().",
    )

    # ── Origin ───────────────────────────────────────────────────────────────
    service_name: str = Field(
        default="unknown-service",
        description="Logical name of the owning micro-service.",
    )
    hostname: str | None = Field(
        default=None,
        description="Hostname of the pod / VM that produced the event.",
    )
    environment: str = Field(
        default="production",
        description="Deployment environment label (production / staging / …).",
    )

    # ── HTTP context (optional) ──────────────────────────────────────────────
    http_context: HTTPContext | None = Field(
        default=None,
        description="Present when the error was intercepted by the ASGI middleware.",
    )

    # ── Derived / enrichment ─────────────────────────────────────────────────
    severity_hint: Severity = Field(
        default=Severity.HIGH,
        description="Coarse severity bucket. Refined by the LLM triage node.",
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value tags for routing / filtering.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary structured metadata (flame-graph URLs, span IDs, …).",
    )

    # ── Validators ───────────────────────────────────────────────────────────
    @field_validator("exception_type", mode="before")
    @classmethod
    def _normalise_exception_type(cls, v: Any) -> str:
        """Accept either a string or a live exception / type object."""
        if isinstance(v, type) and issubclass(v, BaseException):
            return f"{v.__module__}.{v.__qualname__}"
        if isinstance(v, BaseException):
            t = type(v)
            return f"{t.__module__}.{t.__qualname__}"
        return str(v)

    @field_validator("formatted_traceback", mode="before")
    @classmethod
    def _ensure_traceback_string(cls, v: Any) -> str:
        if not isinstance(v, str):
            return "".join(traceback.format_exception(type(v), v, v.__traceback__))
        return v

    @model_validator(mode="after")
    def _derive_severity_from_http(self) -> "UnifiedTelemetrySchema":
        """Upgrade severity_hint when HTTP 5xx is detected."""
        if self.http_context and self.http_context.status_code >= 500:
            # pydantic frozen models require object.__setattr__ for internal mutation
            object.__setattr__(self, "severity_hint", Severity.CRITICAL)
        return self

    # ── Convenience factories ─────────────────────────────────────────────────
    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        service_name: str = "unknown-service",
        http_context: HTTPContext | None = None,
        **kwargs: Any,
    ) -> "UnifiedTelemetrySchema":
        """Construct a telemetry event directly from a live exception object."""
        return cls(
            exception_type=type(exc),  # validator normalises
            exception_message=str(exc),
            formatted_traceback="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
            service_name=service_name,
            http_context=http_context,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Triage Result
# ---------------------------------------------------------------------------


class RemediationSuggestion(BaseModel):
    """A single self-healing action proposed by the LLM."""

    model_config = ConfigDict(frozen=True)

    action_id: str = Field(
        ...,
        description="Identifier matching a developer-registered remediation handler.",
    )
    rationale: str = Field(
        ...,
        description="One-sentence explanation of why this action is recommended.",
    )
    estimated_risk: str = Field(
        default="low",
        pattern=r"^(low|medium|high)$",
        description="Risk assessment for executing this remediation.",
    )


class TriageResult(BaseModel):
    """Structured output of the LangGraph triage node.

    The LLM is instructed to return a JSON object that conforms exactly to
    this schema.  Pydantic validation at the boundary guarantees downstream
    routing logic never encounters missing or type-mismatched fields.
    """

    model_config = ConfigDict(frozen=True)

    # ── Scores ──────────────────────────────────────────────────────────────
    criticality: int = Field(
        ...,
        ge=1,
        le=10,
        description="LLM-assigned criticality score (1 = informational, 10 = service outage).",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the criticality assessment.",
    )

    # ── Classification ───────────────────────────────────────────────────────
    error_category: str = Field(
        ...,
        description="Semantic category, e.g. 'OOM', 'DB_CONNECTION_TIMEOUT', 'UNHANDLED_5XX'.",
    )
    decision: TriageDecision = Field(
        ...,
        description="Routing decision derived by the triage node.",
    )

    # ── Retrieval context ────────────────────────────────────────────────────
    runbook_id: str | None = Field(
        default=None,
        description="ID / filename of the matched runbook, if any.",
    )
    runbook_excerpt: str | None = Field(
        default=None,
        description="Verbatim excerpt from the runbook used for grounding.",
    )

    # ── Remediation ───────────────────────────────────────────────────────────
    suggested_remediations: list[RemediationSuggestion] = Field(
        default_factory=list,
        description="Ordered list of recommended self-healing actions.",
    )

    # ── Narrative ────────────────────────────────────────────────────────────
    summary: str = Field(
        ...,
        description="Human-readable triage summary (≤ 3 sentences).",
    )
    raw_llm_output: str = Field(
        default="",
        description="Verbatim LLM response retained for auditing / replay.",
    )

    # ── Metadata ─────────────────────────────────────────────────────────────
    triage_latency_ms: float | None = Field(
        default=None,
        description="Wall-clock milliseconds consumed by the triage node.",
    )

    @model_validator(mode="after")
    def _align_decision_with_criticality(self) -> "TriageResult":
        """Promote decision to ESCALATE when criticality is at maximum."""
        if self.criticality == 10 and self.decision != TriageDecision.ESCALATE:
            object.__setattr__(self, "decision", TriageDecision.ESCALATE)
        return self
