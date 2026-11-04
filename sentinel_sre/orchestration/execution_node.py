"""
sentinel_sre/orchestration/execution_node.py
─────────────────────────────────────────────────────────────────────────────
Execution node for the LangGraph action pipeline.

Responsibilities
────────────────
1. Receives the ``GraphState`` after a successful triage.
2. Looks up all matching remediation handlers from ``SentinelAgent``'s
   registry using the ``TriageResult.error_category`` and
   ``TriageResult.criticality``.
3. Executes each matching handler concurrently with:
   - Per-handler timeout enforcement.
   - Structured success / failure tracking per handler.
   - Circuit-breaker logic: a handler that fails ``max_handler_failures``
     times in a rolling window is quarantined and skipped for
     ``quarantine_seconds``.
4. Returns an enriched ``GraphState`` with an ``ExecutionReport`` summarising
   what ran, what succeeded, what failed, and total wall-clock time.

This module is **imported and wired into the graph by graph.py** – it does not
construct or modify the graph itself.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sentinel_sre.core.models import TriageResult, UnifiedTelemetrySchema

if TYPE_CHECKING:
    from sentinel_sre.core.agent import RemediationRegistration, SentinelAgent

logger = logging.getLogger("sentinel_sre.execution")


# ---------------------------------------------------------------------------
# Circuit breaker state (module-level, shared across pipeline runs)
# ---------------------------------------------------------------------------

@dataclass
class _HandlerBreaker:
    """Per-handler circuit breaker state.

    Tracks recent failures and enforces a quarantine window when the failure
    threshold is exceeded.
    """
    failure_count: int = 0
    quarantine_until: float = 0.0  # epoch seconds

    def is_open(self, *, max_failures: int, quarantine_seconds: float) -> bool:
        """Return ``True`` if the breaker is OPEN (handler should be skipped)."""
        now = time.monotonic()
        if now < self.quarantine_until:
            return True
        # Quarantine expired – reset counter
        if self.failure_count >= max_failures and now >= self.quarantine_until:
            self.failure_count = 0
        return False

    def record_failure(self, *, max_failures: int, quarantine_seconds: float) -> None:
        self.failure_count += 1
        if self.failure_count >= max_failures:
            self.quarantine_until = time.monotonic() + quarantine_seconds
            logger.warning(
                "sentinel_sre.execution: circuit breaker OPEN for %.0fs "
                "after %d consecutive failures.",
                quarantine_seconds,
                self.failure_count,
            )

    def record_success(self) -> None:
        self.failure_count = 0
        self.quarantine_until = 0.0


# Keyed by handler name (``RemediationRegistration.name``)
_breakers: dict[str, _HandlerBreaker] = {}


def _get_breaker(name: str) -> _HandlerBreaker:
    if name not in _breakers:
        _breakers[name] = _HandlerBreaker()
    return _breakers[name]


# ---------------------------------------------------------------------------
# Per-handler execution result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HandlerResult:
    """Outcome of executing a single remediation handler.

    Attributes
    ----------
    handler_name:
        The ``RemediationRegistration.name`` of the executed handler.
    success:
        ``True`` if the handler completed without raising.
    skipped:
        ``True`` if the handler was skipped (circuit-breaker open).
    error_message:
        Populated when ``success=False`` and ``skipped=False``.
    duration_ms:
        Wall-clock execution time in milliseconds.
    """

    handler_name: str
    success: bool
    skipped: bool = False
    error_message: str | None = None
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Execution report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionReport:
    """Aggregated result of the action node execution.

    Attributes
    ----------
    event_id:
        ID of the originating ``UnifiedTelemetrySchema``.
    triage_category:
        ``TriageResult.error_category`` for the triggering incident.
    total_handlers:
        Number of matching handlers found.
    executed:
        Handlers that were attempted (not skipped).
    succeeded:
        Handlers that completed without error.
    failed:
        Handlers that raised an exception.
    skipped:
        Handlers bypassed due to open circuit breakers.
    total_duration_ms:
        Wall-clock time for the entire action node.
    handler_results:
        Per-handler ``HandlerResult`` objects.
    """

    event_id: str
    triage_category: str
    total_handlers: int
    executed: int
    succeeded: int
    failed: int
    skipped: int
    total_duration_ms: float
    handler_results: tuple[HandlerResult, ...]

    def to_summary(self) -> str:
        lines = [
            f"ExecutionReport for event={self.event_id!r}",
            f"  category       : {self.triage_category}",
            f"  handlers found : {self.total_handlers}",
            f"  executed       : {self.executed} "
            f"(✓{self.succeeded} ✗{self.failed} ⊘{self.skipped})",
            f"  total time     : {self.total_duration_ms:.1f} ms",
        ]
        for r in self.handler_results:
            status = "✓" if r.success else ("⊘" if r.skipped else "✗")
            lines.append(
                f"    {status} {r.handler_name} ({r.duration_ms:.1f} ms)"
                + (f" – {r.error_message}" if r.error_message else "")
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Execution node function
# ---------------------------------------------------------------------------


async def execute_remediations(
    *,
    event: UnifiedTelemetrySchema,
    triage: TriageResult,
    agent: "SentinelAgent",
    handler_timeout: float = 30.0,
    max_handler_failures: int = 3,
    quarantine_seconds: float = 300.0,
) -> ExecutionReport:
    """Core execution logic for the LangGraph ``action_node``.

    This function is extracted from the node so it can be unit-tested
    independently of the LangGraph runtime.

    Parameters
    ----------
    event:
        The original telemetry event.
    triage:
        Validated ``TriageResult`` from the triage node.
    agent:
        The owning ``SentinelAgent`` – provides access to the remediation
        registry via ``agent._registrations``.
    handler_timeout:
        Maximum seconds a single handler may run before being cancelled.
        Default: 30 s.
    max_handler_failures:
        Number of consecutive failures before a handler's circuit breaker
        opens.  Default: 3.
    quarantine_seconds:
        Seconds a handler is quarantined after its circuit breaker opens.
        Default: 300 s (5 minutes).

    Returns
    -------
    An ``ExecutionReport`` summarising the execution outcome.
    """
    t0 = time.perf_counter()

    # ── 1. Find matching handlers ─────────────────────────────────────────────
    matching: list[RemediationRegistration] = [
        reg for reg in agent._registrations if reg.matches(triage)
    ]

    logger.info(
        "sentinel_sre.execution: event=%s category=%r criticality=%d → "
        "%d matching handler(s).",
        event.event_id,
        triage.error_category,
        triage.criticality,
        len(matching),
    )

    if not matching:
        return ExecutionReport(
            event_id=event.event_id,
            triage_category=triage.error_category,
            total_handlers=0,
            executed=0,
            succeeded=0,
            failed=0,
            skipped=0,
            total_duration_ms=0.0,
            handler_results=(),
        )

    # ── 2. Execute each handler concurrently ─────────────────────────────────
    async def _run_one(reg: RemediationRegistration) -> HandlerResult:
        breaker = _get_breaker(reg.name)

        # Circuit breaker check
        if breaker.is_open(
            max_failures=max_handler_failures,
            quarantine_seconds=quarantine_seconds,
        ):
            remaining = breaker.quarantine_until - time.monotonic()
            logger.warning(
                "sentinel_sre.execution: handler %r is quarantined "
                "(%.0f s remaining); skipping.",
                reg.name,
                max(remaining, 0),
            )
            return HandlerResult(
                handler_name=reg.name,
                success=False,
                skipped=True,
                error_message=f"Circuit breaker open ({remaining:.0f}s remaining)",
            )

        handler_t0 = time.perf_counter()
        try:
            await asyncio.wait_for(
                reg.handler(event, triage),
                timeout=handler_timeout,
            )
            duration_ms = (time.perf_counter() - handler_t0) * 1_000
            breaker.record_success()
            logger.info(
                "sentinel_sre.execution: handler %r succeeded in %.1f ms.",
                reg.name,
                duration_ms,
            )
            return HandlerResult(
                handler_name=reg.name,
                success=True,
                duration_ms=duration_ms,
            )

        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - handler_t0) * 1_000
            msg = f"Handler timed out after {handler_timeout:.0f}s"
            breaker.record_failure(
                max_failures=max_handler_failures,
                quarantine_seconds=quarantine_seconds,
            )
            logger.error(
                "sentinel_sre.execution: handler %r timed out (%.1f ms).",
                reg.name,
                duration_ms,
            )
            return HandlerResult(
                handler_name=reg.name,
                success=False,
                error_message=msg,
                duration_ms=duration_ms,
            )

        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - handler_t0) * 1_000
            msg = f"{type(exc).__name__}: {exc}"
            breaker.record_failure(
                max_failures=max_handler_failures,
                quarantine_seconds=quarantine_seconds,
            )
            logger.exception(
                "sentinel_sre.execution: handler %r failed: %s",
                reg.name,
                msg,
            )
            return HandlerResult(
                handler_name=reg.name,
                success=False,
                error_message=msg,
                duration_ms=duration_ms,
            )

    results: list[HandlerResult] = await asyncio.gather(
        *[_run_one(reg) for reg in matching]
    )

    total_ms = (time.perf_counter() - t0) * 1_000
    succeeded = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    executed = len(results) - skipped

    report = ExecutionReport(
        event_id=event.event_id,
        triage_category=triage.error_category,
        total_handlers=len(matching),
        executed=executed,
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        total_duration_ms=round(total_ms, 2),
        handler_results=tuple(results),
    )

    logger.info(
        "sentinel_sre.execution: %s",
        report.to_summary().replace("\n", " | "),
    )
    return report


# ---------------------------------------------------------------------------
# LangGraph action_node wrapper
# ---------------------------------------------------------------------------


async def action_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node that executes developer-registered remediation handlers.

    Accepts the ``GraphState`` dict produced by the triage node and returns a
    partial state update containing the ``ExecutionReport`` and a human-readable
    ``action_taken`` summary.

    This function **replaces** the inline ``action_node`` that was previously
    defined in ``graph.py``.  ``graph.py`` should import and wire this function
    instead of its own implementation.

    Parameters
    ----------
    state:
        The accumulated LangGraph ``GraphState``.

    Returns
    -------
    A partial ``GraphState`` dict with keys:
    - ``action_taken``: human-readable summary string.
    - ``action_successful``: ``True`` when no handler failed or was skipped.
    - ``execution_report``: the ``ExecutionReport`` dataclass instance.
    - ``notification_sent``: ``False`` (action node does not send notifications).
    """
    from sentinel_sre.core.agent import SentinelAgent  # lazy import avoids cycle

    event: UnifiedTelemetrySchema = state["event"]
    triage: TriageResult | None = state.get("triage_result")
    agent: SentinelAgent = state["agent_ref"]

    if triage is None:
        logger.error(
            "sentinel_sre.execution: action_node reached with no triage_result "
            "for event %s.",
            event.event_id,
        )
        return {
            "action_taken": "ERROR: triage_result missing – no remediations dispatched.",
            "action_successful": False,
            "notification_sent": False,
        }

    # Pull per-graph execution config from state (set by build_graph)
    handler_timeout: float = state.get("_handler_timeout", 30.0)  # type: ignore[assignment]
    max_failures: int = state.get("_max_handler_failures", 3)  # type: ignore[assignment]
    quarantine_secs: float = state.get("_quarantine_seconds", 300.0)  # type: ignore[assignment]

    report = await execute_remediations(
        event=event,
        triage=triage,
        agent=agent,
        handler_timeout=handler_timeout,
        max_handler_failures=max_failures,
        quarantine_seconds=quarantine_secs,
    )

    # Successful when nothing failed or was quarantined. Zero matching
    # handlers is still a completed action (nothing was registered to run).
    action_successful = report.failed == 0 and report.skipped == 0

    return {
        "action_taken": report.to_summary(),
        "action_successful": action_successful,
        "execution_report": report,
        "notification_sent": False,
    }
