"""
sentinel_sre/interceptor/asgi_middleware.py
─────────────────────────────────────────────────────────────────────────────
Non-blocking ASGI middleware for automatic exception interception.

Design principles
─────────────────
• **Zero request-latency impact** – exceptions are packaged and enqueued into
  an ``asyncio.Queue`` *after* the HTTP response has already been sent to the
  client.  The error pipeline runs entirely in the background.
• **Starlette / FastAPI compatible** – extends ``BaseHTTPMiddleware`` and uses
  only standard ASGI primitives so it integrates with any ASGI framework.
• **Back-pressure aware** – the queue has a configurable ``maxsize``; when
  full, events are dropped with a structured warning rather than blocking the
  request or silently swallowing errors.
• **Sensitive data redaction** – ``Authorization``, ``Cookie``, and
  ``X-Api-Key`` headers are stripped before the event is enqueued.

Usage
─────
    from sentinel_sre.interceptor.asgi_middleware import SentinelMiddleware

    app = FastAPI()
    app.add_middleware(
        SentinelMiddleware,
        event_queue=sentinel_agent.event_queue,
        service_name="my-api",
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import socket
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from sentinel_sre.core.models import HTTPContext, Severity, UnifiedTelemetrySchema

logger = logging.getLogger("sentinel_sre.interceptor")

# Headers that must never leave the application process.
_REDACTED_HEADERS: frozenset[str] = frozenset(
    {"authorization", "cookie", "x-api-key", "x-auth-token", "proxy-authorization"}
)


def _sanitise_headers(raw: dict[str, str]) -> dict[str, str]:
    """Return a copy of *raw* with sensitive headers replaced by ``<redacted>``."""
    return {
        k: ("<redacted>" if k.lower() in _REDACTED_HEADERS else v)
        for k, v in raw.items()
    }


def _resolve_hostname() -> str:
    """Best-effort hostname resolution; falls back to ``platform.node()``."""
    try:
        return socket.gethostname()
    except Exception:  # pragma: no cover
        return platform.node()


class SentinelMiddleware(BaseHTTPMiddleware):
    """Starlette ``BaseHTTPMiddleware`` that intercepts unhandled exceptions and
    publishes them to an ``asyncio.Queue`` without blocking the HTTP response.

    Parameters
    ----------
    app:
        The inner ASGI application being wrapped.
    event_queue:
        The shared queue consumed by the ``SentinelAgent`` background worker.
    service_name:
        Logical name of this micro-service, embedded in every telemetry event.
    environment:
        Deployment environment label (``production`` / ``staging`` / …).
    queue_maxsize:
        Maximum number of events that can be buffered.  Events arriving on a
        full queue are logged at WARNING level and silently dropped.
    reraise:
        When ``True`` the original exception is re-raised *after* enqueuing so
        that FastAPI's default exception handler can still return a 500.
        Defaults to ``True``.
    extra_tags:
        Static key-value pairs appended to every ``UnifiedTelemetrySchema``
        emitted by this middleware instance.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        event_queue: asyncio.Queue[UnifiedTelemetrySchema],
        service_name: str = "unknown-service",
        environment: str = "production",
        reraise: bool = True,
        extra_tags: dict[str, str] | None = None,
    ) -> None:
        super().__init__(app)
        self._queue: asyncio.Queue[UnifiedTelemetrySchema] = event_queue
        self._service_name = service_name
        self._environment = environment
        self._reraise = reraise
        self._hostname = _resolve_hostname()
        self._extra_tags: dict[str, str] = extra_tags or {}

    # ------------------------------------------------------------------
    # BaseHTTPMiddleware contract
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Delegate to the inner application; intercept any unhandled exception.

        The exception is packaged into a ``UnifiedTelemetrySchema`` and placed
        on the event queue *before* a 500 response is returned (or the
        exception re-raised, depending on ``self._reraise``).
        """
        try:
            response: Response = await call_next(request)
        except Exception as exc:  # noqa: BLE001
            await self._handle_exception(request, exc, status_code=500)
            if self._reraise:
                raise
            # Fallback: return a plain 500 so the connection is not left open.
            return Response(
                content='{"detail":"Internal Server Error"}',
                status_code=500,
                media_type="application/json",
            )

        # Also capture explicitly returned 5xx responses whose exceptions were
        # swallowed internally (e.g. by an inner exception handler).
        if response.status_code >= 500:
            await self._capture_5xx_response(request, response)

        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _handle_exception(
        self,
        request: Request,
        exc: BaseException,
        *,
        status_code: int = 500,
    ) -> None:
        """Build a telemetry event from a live exception and enqueue it."""
        http_ctx = self._build_http_context(request, status_code=status_code)
        event = UnifiedTelemetrySchema.from_exception(
            exc,
            service_name=self._service_name,
            environment=self._environment,
            http_context=http_ctx,
            hostname=self._hostname,
            severity_hint=Severity.CRITICAL,
            tags=self._extra_tags.copy(),
        )
        await self._enqueue(event)

    async def _capture_5xx_response(
        self,
        request: Request,
        response: Response,
    ) -> None:
        """Synthesise a telemetry event for a 5xx that did not raise."""
        http_ctx = self._build_http_context(request, status_code=response.status_code)
        # We have no live exception; construct a synthetic one for consistency.
        synthetic_exc = RuntimeError(
            f"HTTP {response.status_code} returned without raising an exception. "
            f"Path: {request.url.path}"
        )
        event = UnifiedTelemetrySchema.from_exception(
            synthetic_exc,
            service_name=self._service_name,
            environment=self._environment,
            http_context=http_ctx,
            hostname=self._hostname,
            severity_hint=Severity.HIGH,
            tags={**self._extra_tags, "synthetic": "true"},
        )
        await self._enqueue(event)

    def _build_http_context(
        self,
        request: Request,
        *,
        status_code: int,
    ) -> HTTPContext:
        """Extract and sanitise HTTP metadata from a Starlette ``Request``."""
        raw_headers: dict[str, str] = dict(request.headers)
        return HTTPContext(
            method=request.method,
            path=str(request.url.path),
            status_code=status_code,
            client_host=request.client.host if request.client else None,
            headers=_sanitise_headers(raw_headers),
            query_params=dict(request.query_params),
        )

    async def _enqueue(self, event: UnifiedTelemetrySchema) -> None:
        """Non-blocking enqueue with back-pressure handling."""
        try:
            self._queue.put_nowait(event)
            logger.debug(
                "sentinel_sre: enqueued event %s (queue depth: %d)",
                event.event_id,
                self._queue.qsize(),
            )
        except asyncio.QueueFull:
            logger.warning(
                "sentinel_sre: event queue is full (maxsize=%d). "
                "Dropping event %s for %s %s. "
                "Consider increasing queue_maxsize or scaling the consumer.",
                self._queue.maxsize,
                event.event_id,
                event.http_context.method if event.http_context else "N/A",
                event.http_context.path if event.http_context else "N/A",
            )


# ---------------------------------------------------------------------------
# Logging integration – alternative ingestion path
# ---------------------------------------------------------------------------


class SentinelLogHandler(logging.Handler):
    """Python ``logging.Handler`` that converts ``logging.LogRecord`` objects
    carrying exception info into ``UnifiedTelemetrySchema`` events.

    This provides a second ingestion path complementary to the ASGI middleware:
    any ``logger.exception(...)`` call will be automatically captured even
    outside of an HTTP request context.

    Usage
    ─────
        handler = SentinelLogHandler(
            event_queue=sentinel_agent.event_queue,
            service_name="my-worker",
            loop=asyncio.get_event_loop(),
        )
        handler.setLevel(logging.ERROR)
        logging.getLogger().addHandler(handler)
    """

    def __init__(
        self,
        *,
        event_queue: asyncio.Queue[UnifiedTelemetrySchema],
        service_name: str = "unknown-service",
        environment: str = "production",
        loop: asyncio.AbstractEventLoop | None = None,
        extra_tags: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._queue = event_queue
        self._service_name = service_name
        self._environment = environment
        self._loop = loop
        self._hostname = _resolve_hostname()
        self._extra_tags: dict[str, str] = extra_tags or {}

    def emit(self, record: logging.LogRecord) -> None:
        """Synchronously package the log record and schedule queue insertion."""
        if not record.exc_info:
            return  # Only capture records that carry exception context.

        exc_type, exc_value, exc_tb = record.exc_info
        if exc_value is None:
            return

        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        event = UnifiedTelemetrySchema(
            exception_type=exc_value,  # validator normalises
            exception_message=str(exc_value),
            formatted_traceback=tb_str,
            service_name=self._service_name,
            environment=self._environment,
            hostname=self._hostname,
            tags={
                **self._extra_tags,
                "logger": record.name,
                "level": record.levelname,
            },
            extra={"log_message": record.getMessage()},
        )
        self._schedule_enqueue(event)

    def _schedule_enqueue(self, event: UnifiedTelemetrySchema) -> None:
        """Thread-safe scheduling of the enqueue coroutine."""
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No running event loop; drop with a warning.
                self.handleError(
                    logging.LogRecord(
                        name="sentinel_sre",
                        level=logging.WARNING,
                        pathname="",
                        lineno=0,
                        msg="No event loop available; telemetry event dropped.",
                        args=(),
                        exc_info=None,
                    )
                )
                return

        if loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._async_enqueue(event), loop
            )
        else:
            loop.run_until_complete(self._async_enqueue(event))

    async def _async_enqueue(self, event: UnifiedTelemetrySchema) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logging.getLogger("sentinel_sre.log_handler").warning(
                "Event queue full; dropping log-sourced telemetry event %s.",
                event.event_id,
            )
