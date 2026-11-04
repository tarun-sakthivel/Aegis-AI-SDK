"""
sentinel_sre/core/agent.py
─────────────────────────────────────────────────────────────────────────────
``SentinelAgent`` – the central coordination object for the sentinel-sre SDK.

Responsibilities
────────────────
1. Owns the ``asyncio.Queue`` consumed by the ASGI middleware / log handler.
2. Runs a background async worker that drains the queue and feeds events into
   the LangGraph orchestration pipeline.
3. Provides the ``@register_remediation`` decorator so application developers
   can register async self-healing functions without subclassing anything.
4. Exposes ``start()`` / ``stop()`` lifecycle methods for use with FastAPI
   lifespan context managers.

Usage
─────
    from sentinel_sre.core.agent import SentinelAgent

    sentinel = SentinelAgent(
        runbook_dir="./runbooks",
        criticality_threshold=8,
    )

    @sentinel.register_remediation(trigger_pattern="OOM", min_criticality=7)
    async def handle_oom(event, triage):
        # restart the worker pool, flush caches, etc.
        ...

    # FastAPI lifespan
    @asynccontextmanager
    async def lifespan(app):
        await sentinel.start()
        yield
        await sentinel.stop()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(SentinelMiddleware, event_queue=sentinel.event_queue)
"""

from __future__ import annotations

import asyncio
import fnmatch
import functools
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sentinel_sre.core.models import TriageDecision, TriageResult, UnifiedTelemetrySchema

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from sentinel_sre.ingestion.context_graph import TopologyGraph
    from sentinel_sre.retrieval.hybrid_engine import HybridRetriever

logger = logging.getLogger("sentinel_sre.agent")

# Type alias for a remediation handler coroutine.
RemediationHandler = Callable[
    [UnifiedTelemetrySchema, TriageResult],
    Awaitable[None],
]


# ---------------------------------------------------------------------------
# Remediation registration metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemediationRegistration:
    """Metadata stored for each ``@register_remediation``-decorated function.

    Attributes
    ----------
    handler:
        The async callable to invoke when the trigger fires.
    trigger_pattern:
        A glob-style pattern matched against ``TriageResult.error_category``.
        Examples: ``"OOM"``, ``"DB_*"``, ``"*TIMEOUT*"``.
    min_criticality:
        The handler is only invoked when the triage score is **≥** this value.
    name:
        Human-readable identifier (defaults to ``handler.__qualname__``).
    """

    handler: RemediationHandler
    trigger_pattern: str
    min_criticality: int
    name: str

    def matches(self, triage: TriageResult) -> bool:
        """Return ``True`` when this registration should fire for *triage*."""
        category_match = fnmatch.fnmatchcase(
            triage.error_category.upper(), self.trigger_pattern.upper()
        )
        criticality_match = triage.criticality >= self.min_criticality
        return category_match and criticality_match


# ---------------------------------------------------------------------------
# SentinelAgent
# ---------------------------------------------------------------------------


class SentinelAgent:
    """Central coordination object for the sentinel-sre SDK.

    Parameters
    ----------
    runbook_dir:
        Path to the directory containing ``.md`` runbook files.  The hybrid
        retrieval engine (BM25 + vector search) scans this directory at startup.
    criticality_threshold:
        Events with ``TriageResult.criticality >= criticality_threshold`` are
        routed to the action node; below this value they go to the notification
        node.
    queue_maxsize:
        Maximum number of unprocessed telemetry events held in memory.
        ``0`` means unlimited (not recommended in production).
    service_name:
        Logical service name embedded in emitted telemetry events.
    llm_model:
        Identifier for the LLM used by the triage node
        (e.g., ``"gpt-4o"``, ``"gemini-1.5-pro"``).
    concurrency:
        Maximum number of events processed concurrently by the worker pool.
    topology:
        Optional ``TopologyGraph`` injected into triage so the LLM sees the
        service blast radius.  Load it with ``TopologyGraph.load_from_file``.
    """

    def __init__(
        self,
        *,
        runbook_dir: str | Path = "./runbooks",
        criticality_threshold: int = 8,
        queue_maxsize: int = 1_000,
        service_name: str = "unknown-service",
        llm_model: str = "gpt-4o",
        concurrency: int = 4,
        retriever: "HybridRetriever | None" = None,
        embeddings: "Embeddings | None" = None,
        topology: "TopologyGraph | None" = None,
    ) -> None:
        self.runbook_dir = Path(runbook_dir).expanduser().resolve()
        self.criticality_threshold = criticality_threshold
        self.service_name = service_name
        self.llm_model = llm_model
        self.concurrency = concurrency

        # Public queue – shared with the ASGI middleware and log handler.
        self.event_queue: asyncio.Queue[UnifiedTelemetrySchema] = asyncio.Queue(
            maxsize=queue_maxsize
        )

        # Optional pre-built retriever or embeddings for auto-construction.
        # If neither is supplied, build_graph will create a FakeEmbeddings fallback.
        self._retriever: "HybridRetriever | None" = retriever
        self._embeddings: "Embeddings | None" = embeddings
        self._topology: "TopologyGraph | None" = topology

        # Internal state
        self._registrations: list[RemediationRegistration] = []
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._semaphore: asyncio.Semaphore | None = None
        self._running: bool = False

        # Lazy-imported graph to avoid circular imports at module level.
        self._graph: Any | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background worker pool and compile the LangGraph pipeline.

        Must be called before any events can be processed.  Idempotent – safe
        to call multiple times.
        """
        if self._running:
            logger.warning("SentinelAgent.start() called on an already-running agent.")
            return

        logger.info(
            "sentinel_sre: starting agent (runbook_dir=%s, threshold=%d, workers=%d)",
            self.runbook_dir,
            self.criticality_threshold,
            self.concurrency,
        )

        self._semaphore = asyncio.Semaphore(self.concurrency)
        await self._compile_graph()
        self._running = True

        for i in range(self.concurrency):
            task = asyncio.create_task(
                self._worker(worker_id=i), name=f"sentinel-worker-{i}"
            )
            self._worker_tasks.append(task)

        logger.info("sentinel_sre: %d worker(s) started.", self.concurrency)

    async def stop(self, timeout: float = 10.0) -> None:
        """Gracefully drain the queue and stop all worker tasks.

        Parameters
        ----------
        timeout:
            Seconds to wait for in-flight events to complete before cancelling.
        """
        if not self._running:
            return

        logger.info("sentinel_sre: stopping agent (draining queue)…")
        self._running = False

        # Allow workers to finish current items.
        try:
            await asyncio.wait_for(self.event_queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "sentinel_sre: queue drain timed out after %.1fs; "
                "cancelling worker tasks.",
                timeout,
            )

        for task in self._worker_tasks:
            task.cancel()

        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        logger.info("sentinel_sre: agent stopped.")

    # ------------------------------------------------------------------
    # Decorator API
    # ------------------------------------------------------------------

    def register_remediation(
        self,
        trigger_pattern: str,
        *,
        min_criticality: int = 1,
    ) -> Callable[[RemediationHandler], RemediationHandler]:
        """Class method decorator that registers an async self-healing function.

        Parameters
        ----------
        trigger_pattern:
            Glob-style pattern matched against ``TriageResult.error_category``.
            Use ``"*"`` to match all categories.
        min_criticality:
            Minimum criticality score (inclusive) required to invoke this
            handler.  Defaults to ``1`` (always fires when pattern matches).

        Example
        -------
        .. code-block:: python

            @sentinel.register_remediation("OOM", min_criticality=7)
            async def handle_oom(
                event: UnifiedTelemetrySchema,
                triage: TriageResult,
            ) -> None:
                await restart_inference_workers()
                await clear_model_cache()
        """
        if not 1 <= min_criticality <= 10:
            raise ValueError(
                f"min_criticality must be between 1 and 10, got {min_criticality}."
            )

        def decorator(fn: RemediationHandler) -> RemediationHandler:
            registration = RemediationRegistration(
                handler=fn,
                trigger_pattern=trigger_pattern,
                min_criticality=min_criticality,
                name=getattr(fn, "__qualname__", repr(fn)),
            )
            self._registrations.append(registration)
            logger.debug(
                "sentinel_sre: registered remediation '%s' (pattern=%r, min_criticality=%d)",
                registration.name,
                trigger_pattern,
                min_criticality,
            )

            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                return await fn(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    # ------------------------------------------------------------------
    # Remediation dispatch
    # ------------------------------------------------------------------

    async def dispatch_remediations(
        self,
        event: UnifiedTelemetrySchema,
        triage: TriageResult,
    ) -> None:
        """Invoke all registered handlers whose pattern and criticality match.

        Handlers are invoked concurrently; individual failures are logged but
        do not abort other handlers.

        Parameters
        ----------
        event:
            The original telemetry event that triggered the pipeline.
        triage:
            The ``TriageResult`` produced by the LangGraph triage node.
        """
        matching = [r for r in self._registrations if r.matches(triage)]

        if not matching:
            logger.info(
                "sentinel_sre: no registered remediations matched "
                "(category=%r, criticality=%d).",
                triage.error_category,
                triage.criticality,
            )
            return

        logger.info(
            "sentinel_sre: dispatching %d remediation handler(s) for event %s.",
            len(matching),
            event.event_id,
        )

        async def _safe_invoke(reg: RemediationRegistration) -> None:
            try:
                logger.debug("sentinel_sre: invoking handler %r", reg.name)
                await reg.handler(event, triage)
                logger.info("sentinel_sre: handler %r completed successfully.", reg.name)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "sentinel_sre: handler %r raised an unhandled exception.", reg.name
                )

        await asyncio.gather(*[_safe_invoke(r) for r in matching])

    # ------------------------------------------------------------------
    # Internal worker
    # ------------------------------------------------------------------

    async def _worker(self, *, worker_id: int) -> None:
        """Long-running coroutine that drains the event queue."""
        logger.debug("sentinel_sre: worker-%d started.", worker_id)
        while self._running or not self.event_queue.empty():
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            assert self._semaphore is not None  # set in start()
            async with self._semaphore:
                try:
                    await self._process_event(event)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "sentinel_sre: worker-%d failed to process event %s.",
                        worker_id,
                        event.event_id,
                    )
                finally:
                    self.event_queue.task_done()

        logger.debug("sentinel_sre: worker-%d exiting.", worker_id)

    async def _process_event(self, event: UnifiedTelemetrySchema) -> None:
        """Run the LangGraph pipeline for a single telemetry event."""
        from sentinel_sre.orchestration.graph import run_pipeline  # lazy import

        logger.info(
            "sentinel_sre: processing event %s (%s)",
            event.event_id,
            event.exception_type,
        )
        await run_pipeline(
            event=event,
            agent=self,
            graph=self._graph,
        )

    async def _compile_graph(self) -> None:
        """Compile the LangGraph state machine (deferred to avoid import cycles).

        If a ``HybridRetriever`` was not injected at construction time but
        ``embeddings`` were provided, a new retriever is built here.  The
        retriever is **not** automatically indexed – callers must run
        ``RunbookLoader.load_and_index()`` before or after calling
        ``SentinelAgent.start()``.
        """
        from sentinel_sre.orchestration.graph import build_graph  # lazy import

        retriever = self._retriever
        if retriever is None and self._embeddings is not None:
            from sentinel_sre.retrieval.hybrid_engine import HybridRetriever  # lazy
            retriever = HybridRetriever(embeddings=self._embeddings)
            self._retriever = retriever
            logger.info(
                "sentinel_sre: created HybridRetriever from provided embeddings (%s).",
                type(self._embeddings).__name__,
            )

        self._graph = build_graph(
            criticality_threshold=self.criticality_threshold,
            llm_model=self.llm_model,
            runbook_dir=self.runbook_dir,
            retriever=retriever,
            topology=self._topology,
        )
        logger.info("sentinel_sre: LangGraph pipeline compiled.")
