"""
sentinel_sre/orchestration/triage_node.py
─────────────────────────────────────────────────────────────────────────────
LangGraph triage node: the LLM "brain" of the incident response pipeline.

This module implements ``run_triage`` – the async function called by the
LangGraph ``triage_node`` edge.  It is intentionally split from ``graph.py``
so it can be unit-tested independently and swapped for alternative LLM
providers without touching the graph topology.

Prompt design
─────────────
The system prompt is structured as three context blocks, ordered by signal
quality:

1. **Error trace** – most specific, highest signal.  Full exception type,
   message, and trimmed stack trace from ``UnifiedTelemetrySchema``.

2. **Runbook SOP** – retrieved via ``HybridRetriever``, provides authoritative
   remediation procedures written by the team.

3. **Blast radius** – topology context from ``TopologyGraph``, tells the LLM
   which upstream callers and downstream dependencies are at risk.

The LLM is then asked to respond **only** with a JSON object conforming to
the ``TriageResult`` Pydantic schema, enforced via LangChain's
``.with_structured_output()`` method.  Pydantic re-validates the parsed object
at the boundary so downstream routing is never exposed to type mismatches.

Fallback strategy
─────────────────
If ``.with_structured_output()`` fails (e.g. the model doesn't support tool-
calling), the node falls back to a ``JsonOutputParser`` + manual Pydantic
validation path, logging the degradation.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from sentinel_sre.core.models import (
    TriageDecision,
    TriageResult,
    UnifiedTelemetrySchema,
)

logger = logging.getLogger("sentinel_sre.triage")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are an expert Staff Site Reliability Engineer performing automated incident triage.
You have been given three pieces of context:

━━━ CONTEXT 1 – ERROR EVENT ━━━
Service      : {service_name}
Environment  : {environment}
Exception    : {exception_type}
Message      : {exception_message}
Traceback:
{formatted_traceback}

━━━ CONTEXT 2 – RUNBOOK / SOP ━━━
Runbook ID   : {runbook_id}
Excerpt:
{runbook_excerpt}

━━━ CONTEXT 3 – SERVICE BLAST RADIUS ━━━
{blast_radius_context}

━━━ YOUR TASK ━━━
Analyse the error event in light of the runbook SOP and blast radius context.
Produce a triage assessment as a **strict JSON object** matching this schema:

{{
  "criticality": <int 1-10>,
  "confidence": <float 0.0-1.0>,
  "error_category": <string>,
  "decision": <"remediate" | "notify" | "escalate">,
  "runbook_id": <string | null>,
  "runbook_excerpt": <string | null>,
  "suggested_remediations": [
    {{"action_id": "<str>", "rationale": "<str>", "estimated_risk": "low|medium|high"}}
  ],
  "summary": <string, ≤3 sentences>,
  "raw_llm_output": ""
}}

ROUTING RULES (MANDATORY):
- criticality == 10                        → decision MUST be "escalate"
- criticality >= {criticality_threshold}   → decision MUST be "remediate"
- criticality <  {criticality_threshold}   → decision MUST be "notify"

SCORING GUIDANCE:
 1-3  : Informational / minor degradation, no user impact.
 4-6  : Partial degradation, subset of users affected.
 7-9  : Major degradation, most users affected or data-loss risk.
   10 : Complete service outage or unrecoverable state.

Consider the blast radius: a failure in a core shared service warrants a
higher criticality than the same error in an isolated worker.

Output ONLY the JSON object. No markdown fences, no explanations.
"""


# ---------------------------------------------------------------------------
# Main triage function
# ---------------------------------------------------------------------------


async def run_triage(
    state: dict[str, Any],
    llm_client: Any,
    *,
    criticality_threshold: int = 8,
    max_traceback_chars: int = 3_000,
    max_runbook_chars: int = 2_000,
) -> dict[str, Any]:
    """Execute LLM-powered triage and return a partial ``GraphState`` update.

    This is the **canonical entry point** for the triage node.  It is called
    by the LangGraph node function (defined in ``graph.py``) and returns a
    dict that LangGraph merges into the accumulated state.

    Parameters
    ----------
    state:
        The current ``GraphState`` dict.  Must contain ``event``,
        ``_retriever``, and optionally ``_topology``.
    llm_client:
        A LangChain chat model instance (``ChatOpenAI``, ``ChatAnthropic``,
        ``ChatGoogleGenerativeAI``, etc.).  The function attempts to use
        ``.with_structured_output(TriageResult)`` and falls back to
        ``JsonOutputParser`` if the model does not support tool-calling.
    criticality_threshold:
        Score threshold forwarded into the prompt so the LLM knows the
        routing boundary.
    max_traceback_chars:
        Maximum characters of the stack trace included in the prompt.
        Prevents prompt budget exhaustion on very long tracebacks.
    max_runbook_chars:
        Maximum characters of the runbook excerpt included in the prompt.

    Returns
    -------
    A partial ``GraphState`` dict containing:
    - ``triage_result``: validated ``TriageResult`` or ``None`` on failure.
    - ``triage_error``: error string on failure, else ``None``.
    - ``retrieved_runbook_id``, ``retrieved_runbook_excerpt``: from retriever.
    - ``retrieved_chunks``: full ranked chunk list.
    """
    t0 = time.perf_counter()

    event: UnifiedTelemetrySchema = state["event"]
    retriever = state.get("_retriever")
    topology = state.get("_topology")          # TopologyGraph | None
    threshold: int = state.get("criticality_threshold", criticality_threshold)

    # ── 1. Hybrid retrieval ───────────────────────────────────────────────────
    from sentinel_sre.retrieval.hybrid_engine import HybridRetriever, RetrievedChunk

    retrieved_chunks: list[RetrievedChunk] = []
    runbook_id: str | None = None
    runbook_excerpt: str | None = None

    if retriever is not None and isinstance(retriever, HybridRetriever) and retriever.is_ready:
        query = f"{event.exception_type} {event.exception_message}"
        try:
            retrieved_chunks = await retriever.retrieve(query, top_k=5)
            if retrieved_chunks:
                best = retrieved_chunks[0]
                runbook_id = best.source
                runbook_excerpt = best.content[:max_runbook_chars]
                logger.debug(
                    "sentinel_sre.triage: top retrieval – doc_id=%r rrf=%.4f "
                    "bm25_rank=%s vector_rank=%s",
                    best.doc_id, best.rrf_score, best.bm25_rank, best.vector_rank,
                )
        except Exception:  # noqa: BLE001
            logger.exception(
                "sentinel_sre.triage: retrieval failed for event %s; "
                "continuing without runbook context.",
                event.event_id,
            )
    else:
        logger.warning(
            "sentinel_sre.triage: no indexed HybridRetriever available; "
            "LLM will triage without runbook context."
        )

    # ── 2. Blast-radius context ───────────────────────────────────────────────
    blast_radius_context = "No topology graph configured."
    if topology is not None:
        try:
            from sentinel_sre.ingestion.context_graph import TopologyGraph
            if isinstance(topology, TopologyGraph) and not topology.is_empty:
                blast = topology.get_blast_radius(
                    event.service_name,
                    max_hops=state.get("_blast_radius_hops", 2),  # type: ignore[arg-type]
                )
                blast_radius_context = blast.to_llm_context()
        except Exception:  # noqa: BLE001
            logger.exception(
                "sentinel_sre.triage: blast-radius query failed; "
                "proceeding without topology context."
            )

    # ── 3. Build prompt variables ─────────────────────────────────────────────
    prompt_vars = {
        "service_name": event.service_name,
        "environment": event.environment,
        "exception_type": event.exception_type,
        "exception_message": event.exception_message,
        "formatted_traceback": event.formatted_traceback[-max_traceback_chars:],
        "runbook_id": runbook_id or "N/A",
        "runbook_excerpt": runbook_excerpt or "No matching runbook found.",
        "blast_radius_context": blast_radius_context,
        "criticality_threshold": threshold,
    }

    # ── 4. LLM call with structured output (tool-calling path) ───────────────
    triage_result: TriageResult | None = None
    triage_error: str | None = None
    raw_output: str = ""

    try:
        triage_result, raw_output = await _invoke_with_structured_output(
            llm_client=llm_client,
            prompt_vars=prompt_vars,
            runbook_id=runbook_id,
            runbook_excerpt=runbook_excerpt,
        )
    except Exception as exc:  # noqa: BLE001
        # Fallback: attempt JSON parsing without structured output
        logger.warning(
            "sentinel_sre.triage: structured output failed (%s: %s); "
            "falling back to JSON parser.",
            type(exc).__name__,
            exc,
        )
        try:
            triage_result, raw_output = await _invoke_with_json_fallback(
                llm_client=llm_client,
                prompt_vars=prompt_vars,
                runbook_id=runbook_id,
                runbook_excerpt=runbook_excerpt,
            )
        except Exception as fallback_exc:  # noqa: BLE001
            triage_error = f"{type(fallback_exc).__name__}: {fallback_exc}"
            logger.exception(
                "sentinel_sre.triage: both triage paths failed for event %s: %s",
                event.event_id,
                triage_error,
            )

    latency_ms = (time.perf_counter() - t0) * 1_000

    # Stamp latency onto the result (requires bypassing frozen model)
    if triage_result is not None:
        object.__setattr__(triage_result, "triage_latency_ms", round(latency_ms, 2))
        object.__setattr__(triage_result, "raw_llm_output", raw_output)
        logger.info(
            "sentinel_sre.triage: complete – criticality=%d decision=%s "
            "confidence=%.2f latency=%.1fms",
            triage_result.criticality,
            triage_result.decision,
            triage_result.confidence,
            latency_ms,
        )
    else:
        logger.error(
            "sentinel_sre.triage: triage failed for event %s after %.1fms: %s",
            event.event_id,
            latency_ms,
            triage_error,
        )

    return {
        "triage_result": triage_result,
        "triage_error": triage_error,
        "retrieved_runbook_id": runbook_id,
        "retrieved_runbook_excerpt": runbook_excerpt,
        "retrieved_chunks": retrieved_chunks,
    }


# ---------------------------------------------------------------------------
# LLM invocation strategies
# ---------------------------------------------------------------------------


async def _invoke_with_structured_output(
    *,
    llm_client: Any,
    prompt_vars: dict[str, Any],
    runbook_id: str | None,
    runbook_excerpt: str | None,
) -> tuple[TriageResult, str]:
    """Primary path: use ``.with_structured_output(TriageResult)``.

    LangChain will inject a tool/function-call schema derived from the Pydantic
    model, forcing the model to emit a valid JSON blob without any prompt
    engineering tricks.

    Returns
    -------
    ``(TriageResult, raw_json_string)``
    """
    structured_llm = llm_client.with_structured_output(
        TriageResult,
        method="json_schema",          # preferred; falls back to function_calling
        include_raw=True,              # returns {"raw": AIMessage, "parsed": TriageResult}
    )

    messages = [
        SystemMessage(content=_SYSTEM_TEMPLATE.format(**prompt_vars)),
        HumanMessage(content="Produce the triage JSON now."),
    ]

    response: dict[str, Any] = await structured_llm.ainvoke(messages)

    parsed: TriageResult = response["parsed"]
    raw_msg = response.get("raw")
    raw_output = (
        raw_msg.content if hasattr(raw_msg, "content") else json.dumps(parsed.model_dump())
    )

    # Merge retrieval context into the parsed result (LLM may not echo it back)
    if parsed.runbook_id is None and runbook_id:
        object.__setattr__(parsed, "runbook_id", runbook_id)
    if parsed.runbook_excerpt is None and runbook_excerpt:
        object.__setattr__(parsed, "runbook_excerpt", runbook_excerpt[:500])

    return parsed, raw_output


async def _invoke_with_json_fallback(
    *,
    llm_client: Any,
    prompt_vars: dict[str, Any],
    runbook_id: str | None,
    runbook_excerpt: str | None,
) -> tuple[TriageResult, str]:
    """Fallback path: plain chat completion + ``JsonOutputParser`` + Pydantic.

    Used when the model does not support tool-calling (e.g. some open-source
    models served via Ollama).

    Returns
    -------
    ``(TriageResult, raw_json_string)``
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_TEMPLATE),
            ("human", "Produce the triage JSON now."),
        ]
    )
    chain = prompt | llm_client | JsonOutputParser()

    llm_dict: dict[str, Any] = await chain.ainvoke(prompt_vars)
    raw_output = json.dumps(llm_dict)

    # Ensure retrieval fields are present
    llm_dict.setdefault("runbook_id", runbook_id)
    llm_dict.setdefault("runbook_excerpt", runbook_excerpt)
    llm_dict.setdefault("raw_llm_output", raw_output)

    triage_result = TriageResult(**llm_dict)
    return triage_result, raw_output


# ---------------------------------------------------------------------------
# LangGraph node adapter
# ---------------------------------------------------------------------------


async def triage_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node function that wraps ``run_triage``.

    This is the function registered with ``workflow.add_node("triage_node", ...)``.
    It extracts the LLM client and threshold from the injected state keys set
    by ``build_graph`` and delegates to ``run_triage``.

    Parameters
    ----------
    state:
        The accumulated ``GraphState``.  Must contain ``_llm``,
        ``criticality_threshold``, ``event``, and optionally ``_retriever``
        and ``_topology``.

    Returns
    -------
    Partial ``GraphState`` update from ``run_triage``.
    """
    llm_client = state["_llm"]
    threshold: int = state.get("criticality_threshold", 8)  # type: ignore[assignment]
    return await run_triage(
        state,
        llm_client,
        criticality_threshold=threshold,
    )
