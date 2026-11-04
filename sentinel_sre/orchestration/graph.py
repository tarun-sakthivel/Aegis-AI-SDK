"""
sentinel_sre/orchestration/graph.py
─────────────────────────────────────────────────────────────────────────────
LangGraph orchestration pipeline for incident triage and self-healing.

Pipeline overview
─────────────────

                 ┌──────────────┐
     event ───► │  triage_node │
                 └──────┬───────┘
                        │
              ┌─────────▼──────────┐
              │  conditional_edge  │
              │  (criticality cmp) │
              └──┬─────────────┬───┘
     threshold  │             │  below threshold
                ▼             ▼
        ┌─────────────┐  ┌─────────────┐
        │ action_node │  │ notify_node │
        └──────┬──────┘  └──────┬──────┘
               │                │
               └────────┬───────┘
                        ▼
                      END

Nodes
─────
triage_node
    Performs hybrid retrieval (BM25 + vector search via pgvector) over the
    local runbook corpus, then calls an LLM to produce a ``TriageResult``.

action_node
    Looks up developer-registered remediation handlers on the owning
    ``SentinelAgent`` and executes them asynchronously.

notify_node
    Fires Slack / Jira notifications via MCP tool calls (or fallback HTTP).

State
─────
``GraphState`` is a ``TypedDict`` that travels through the graph.  LangGraph
merges partial updates from each node into the accumulated state.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from sentinel_sre.core.models import TriageDecision, TriageResult, UnifiedTelemetrySchema
from sentinel_sre.orchestration.execution_node import ExecutionReport
from sentinel_sre.orchestration.execution_node import action_node as _action_node_impl
from sentinel_sre.orchestration.triage_node import triage_node as _triage_node_impl
from sentinel_sre.retrieval.hybrid_engine import HybridRetriever, RetrievedChunk

if TYPE_CHECKING:
    from sentinel_sre.core.agent import SentinelAgent
    from sentinel_sre.ingestion.context_graph import TopologyGraph

logger = logging.getLogger("sentinel_sre.orchestration")


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    """Mutable state carried through the LangGraph execution.

    Every field is ``Optional`` / has a default because LangGraph nodes return
    *partial* updates – only the keys they modify need to be present in the
    returned dict.
    """

    # ── Input (populated before the graph is invoked) ───────────────────────
    event: UnifiedTelemetrySchema
    """The original telemetry event that triggered the pipeline."""

    agent_ref: Any
    """Reference to the ``SentinelAgent`` instance (for dispatching remediations)."""

    criticality_threshold: int
    """Threshold passed through from agent configuration."""

    # ── Retrieval engine (injected by build_graph) ──────────────────────────
    _retriever: HybridRetriever
    """Shared ``HybridRetriever`` instance; set once at graph compile time."""

    # ── Retrieval results (populated by triage_node) ──────────────────────────
    retrieved_runbook_id: str | None
    """Filename / ID of the best-matching runbook chunk."""

    retrieved_runbook_excerpt: str | None
    """Verbatim excerpt used for LLM grounding."""

    retrieved_chunks: list[RetrievedChunk]
    """Full ranked list of retrieved chunks (for downstream inspection/logging)."""

    # ── Triage (populated by triage_node) ────────────────────────────────────
    triage_result: TriageResult | None
    """Validated LLM triage output, or ``None`` if triage failed."""

    triage_error: str | None
    """Error string when triage fails; otherwise ``None``."""

    # ── Injected dependencies (set by build_graph / run_pipeline) ───────────
    _llm: Any
    """LangChain chat model used by the triage node."""

    _topology: Any
    """Optional ``TopologyGraph`` used for blast-radius context."""

    _handler_timeout: float
    _max_handler_failures: int
    _quarantine_seconds: float
    _blast_radius_hops: int
    _mcp_client: Any

    # ── Action / notify outputs ──────────────────────────────────────────────
    action_taken: str | None
    """Human-readable summary of the execution node outcome."""

    action_successful: bool
    """True when every attempted remediation handler succeeded."""

    execution_report: ExecutionReport | None
    """Structured per-handler execution report from the action node."""

    notification_sent: bool
    """Whether the notify node emitted an alert."""


# ---------------------------------------------------------------------------
# Node adapters
# ---------------------------------------------------------------------------
# triage_node and action_node are imported from their dedicated modules
# and adapted here to satisfy LangGraph's TypedDict state contract.


async def triage_node(state: GraphState) -> GraphState:  # type: ignore[return]
    """Thin adapter – delegates to ``sentinel_sre.orchestration.triage_node``."""
    partial = await _triage_node_impl(state)  # type: ignore[arg-type]
    return GraphState(**partial)  # type: ignore[return-value]


async def action_node(state: GraphState) -> GraphState:  # type: ignore[return]
    """Thin adapter – delegates to ``sentinel_sre.orchestration.execution_node``.

    Invoked when ``triage_result.criticality >= criticality_threshold``
    or when ``triage_result.decision == TriageDecision.ESCALATE``.
    """
    partial = await _action_node_impl(state)  # type: ignore[arg-type]
    return GraphState(**partial)  # type: ignore[return-value]


async def notify_node(state: GraphState) -> GraphState:
    """Notification node – fires Slack / Jira alerts via MCP or fallback HTTP.

    Invoked when ``triage_result.criticality < criticality_threshold``, i.e.
    the incident warrants awareness but not automated remediation.
    """
    event: UnifiedTelemetrySchema = state["event"]
    triage: TriageResult | None = state.get("triage_result")

    if triage is None:
        logger.error(
            "sentinel_sre: notify_node reached with no triage_result for event %s.",
            event.event_id,
        )
        return GraphState(notification_sent=False)

    logger.info(
        "sentinel_sre: notify_node activated for event %s "
        "(criticality=%d – below threshold).",
        event.event_id,
        triage.criticality,
    )

    # MCP tool integration point.  The MCP server exposes tools named
    # ``slack_post_message`` and ``jira_create_issue``.  When a configured MCP
    # client is attached to the agent, those tools are invoked here.  This stub
    # demonstrates the call contract; wire in your MCP client accordingly.
    notification_payload = {
        "event_id": event.event_id,
        "service": event.service_name,
        "environment": event.environment,
        "criticality": triage.criticality,
        "category": triage.error_category,
        "summary": triage.summary,
        "runbook": triage.runbook_id,
        "timestamp": event.timestamp.isoformat(),
    }

    mcp_client = state.get("_mcp_client")  # type: ignore[assignment]
    if mcp_client is not None:
        try:
            await mcp_client.call_tool(
                "slack_post_message",
                arguments={
                    "channel": "#incidents",
                    "text": (
                        f"🚨 *[sentinel-sre]* `{event.service_name}` | "
                        f"criticality *{triage.criticality}/10* | "
                        f"{triage.summary}"
                    ),
                    "metadata": notification_payload,
                },
            )
            logger.info("sentinel_sre: Slack notification sent for event %s.", event.event_id)
        except Exception:  # noqa: BLE001
            logger.exception(
                "sentinel_sre: Slack MCP call failed for event %s; "
                "falling back to structured log.",
                event.event_id,
            )
            logger.warning("sentinel_sre: INCIDENT ALERT %s", json.dumps(notification_payload))
    else:
        # No MCP client configured – emit a structured log that can be scraped
        # by an external alerting system (e.g., Datadog, Grafana Loki).
        logger.warning("sentinel_sre: INCIDENT ALERT %s", json.dumps(notification_payload))

    return GraphState(notification_sent=True)


# ---------------------------------------------------------------------------
# Conditional routing edge
# ---------------------------------------------------------------------------


def _route_after_triage(state: GraphState) -> str:
    """LangGraph conditional edge function.

    Returns the name of the next node based on the triage result.
    """
    triage: TriageResult | None = state.get("triage_result")
    threshold: int = state.get("criticality_threshold", 8)

    if triage is None:
        # Triage failed – default to notify so the error is not silently dropped.
        logger.warning(
            "sentinel_sre: triage_result is None; routing to notify_node as fallback."
        )
        return "notify_node"

    if triage.decision == TriageDecision.ESCALATE or triage.criticality >= threshold:
        return "action_node"

    return "notify_node"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(
    *,
    criticality_threshold: int,
    llm_model: str,
    runbook_dir: Path,
    retriever: HybridRetriever | None = None,
    topology: "TopologyGraph | None" = None,
    llm: Any | None = None,
    handler_timeout: float = 30.0,
    max_handler_failures: int = 3,
    quarantine_seconds: float = 300.0,
    blast_radius_hops: int = 2,
) -> Any:
    """Compile and return the LangGraph ``CompiledGraph``.

    The compiled graph is stateless and thread-safe; it can be shared across
    multiple worker coroutines.

    Parameters
    ----------
    criticality_threshold:
        Forwarded into ``GraphState`` so the routing edge can compare against
        ``triage_result.criticality`` without capturing it in a closure.
    llm_model:
        OpenAI-compatible model identifier used by the triage node.
    runbook_dir:
        Path to the runbook corpus.  Used only for logging context; actual
        indexing is performed by ``RunbookLoader`` before ``build_graph`` is
        called.
    retriever:
        A pre-built and pre-indexed ``HybridRetriever`` instance.  When
        ``None`` (the default for backward compatibility), an empty retriever
        with ``FakeEmbeddings`` is created so the graph can still compile; a
        WARNING is emitted.
    topology:
        Optional service graph used by the triage node for blast radius.
    llm:
        LangChain chat model.  When omitted, a ``ChatOpenAI`` client is
        built from ``llm_model`` (requires ``OPENAI_API_KEY``).
    """
    from langchain_core.embeddings import FakeEmbeddings  # lightweight fallback

    if topology is not None:
        logger.info(
            "sentinel_sre: topology graph attached (%r).",
            topology,
        )

    if retriever is None:
        logger.warning(
            "sentinel_sre: build_graph called without a HybridRetriever. "
            "Creating an empty retriever with FakeEmbeddings. "
            "Call RunbookLoader.load_and_index() before SentinelAgent.start()."
        )
        retriever = HybridRetriever(embeddings=FakeEmbeddings(size=768))

    if llm is None:
        llm = ChatOpenAI(
            model=llm_model,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    # ── State machine definition ─────────────────────────────────────────────
    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node("triage_node", triage_node)
    workflow.add_node("action_node", action_node)
    workflow.add_node("notify_node", notify_node)

    # Edges
    workflow.add_edge(START, "triage_node")
    workflow.add_conditional_edges(
        "triage_node",
        _route_after_triage,
        {
            "action_node": "action_node",
            "notify_node": "notify_node",
        },
    )
    workflow.add_edge("action_node", END)
    workflow.add_edge("notify_node", END)

    compiled = workflow.compile()

    # Attach pre-built artefacts as graph-level metadata so ``run_pipeline``
    # can inject them into the initial state.
    compiled._sentinel_meta = {  # type: ignore[attr-defined]
        "_retriever": retriever,
        "_llm": llm,
        "_topology": topology,
        "_handler_timeout": handler_timeout,
        "_max_handler_failures": max_handler_failures,
        "_quarantine_seconds": quarantine_seconds,
        "_blast_radius_hops": blast_radius_hops,
        "criticality_threshold": criticality_threshold,
    }

    logger.info(
        "sentinel_sre: LangGraph compiled "
        "(nodes: triage → [action | notify] → END, "
        "retriever=%r).",
        retriever,
    )
    return compiled


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


async def run_pipeline(
    *,
    event: UnifiedTelemetrySchema,
    agent: "SentinelAgent",
    graph: Any,
) -> GraphState:
    """Invoke the compiled LangGraph pipeline for a single telemetry event.

    Parameters
    ----------
    event:
        The telemetry event to process.
    agent:
        The owning ``SentinelAgent`` instance (passed through state so nodes
        can call ``agent.dispatch_remediations``).
    graph:
        The compiled LangGraph graph returned by ``build_graph``.

    Returns
    -------
    The final accumulated ``GraphState`` after all nodes have executed.
    """
    meta: dict[str, Any] = getattr(graph, "_sentinel_meta", {})

    initial_state: GraphState = GraphState(
        event=event,
        agent_ref=agent,
        triage_result=None,
        triage_error=None,
        retrieved_runbook_id=None,
        retrieved_runbook_excerpt=None,
        retrieved_chunks=[],
        action_taken=None,
        execution_report=None,
        notification_sent=False,
        **meta,
    )

    final_state: GraphState = await graph.ainvoke(initial_state)
    return final_state
