"""End-to-end checks for topology, triage, and remediation execution.

Uses a fake LangChain chat model so the pipeline can run without an API key.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from sentinel_sre.core.agent import RemediationRegistration, SentinelAgent
from sentinel_sre.core.models import (
    RemediationSuggestion,
    TriageDecision,
    TriageResult,
    UnifiedTelemetrySchema,
)
from sentinel_sre.ingestion.context_graph import TopologyGraph
from sentinel_sre.orchestration import execution_node as execution_mod
from sentinel_sre.orchestration.execution_node import action_node
from sentinel_sre.orchestration.graph import build_graph, run_pipeline
from sentinel_sre.orchestration.triage_node import run_triage
from sentinel_sre.retrieval.hybrid_engine import HybridRetriever

_MARKDOWN = """
## Services
- fastapi
- celery_worker
- postgres
- redis

## Dependencies
- fastapi -> celery_worker
- fastapi -> redis
- celery_worker -> postgres
"""

_JSON = """
{
  "services": ["fastapi", "celery_worker", "postgres"],
  "edges": [
    {"from": "fastapi", "to": "celery_worker"},
    {"from": "celery_worker", "to": "postgres"}
  ]
}
"""


def _event(service: str = "celery_worker") -> UnifiedTelemetrySchema:
    try:
        raise RuntimeError("worker lost connection to broker")
    except RuntimeError as exc:
        return UnifiedTelemetrySchema.from_exception(exc, service_name=service)


def _triage(
    *,
    criticality: int = 9,
    category: str = "BROKER_CONNECTION",
    decision: TriageDecision = TriageDecision.REMEDIATE,
) -> TriageResult:
    return TriageResult(
        criticality=criticality,
        confidence=0.91,
        error_category=category,
        decision=decision,
        summary="Celery cannot reach the broker.",
        suggested_remediations=[
            RemediationSuggestion(
                action_id="restart_celery",
                rationale="Restart the worker so it reconnects.",
                estimated_risk="low",
            )
        ],
    )


class _RawMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _Structured:
    def __init__(self, result: TriageResult, sink: dict[str, Any]) -> None:
        self._result = result
        self._sink = sink

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        self._sink["system"] = messages[0].content
        return {"parsed": self._result, "raw": _RawMessage(self._result.model_dump_json())}


class _FakeLLM:
    """Minimal stand-in for a LangChain chat model with structured output."""

    def __init__(self, result: TriageResult) -> None:
        self.result = result
        self.calls: dict[str, Any] = {}
        self.structured_kwargs: dict[str, Any] = {}

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _Structured:
        self.structured_kwargs = {"schema": schema, **kwargs}
        return _Structured(self.result, self.calls)


@pytest.fixture(autouse=True)
def _reset_breakers() -> None:
    execution_mod._breakers.clear()


def test_markdown_and_json_build_directed_edges() -> None:
    from_md = TopologyGraph(name="md")
    from_md.load_from_markdown(_MARKDOWN)
    assert ("fastapi", "celery_worker") in list(from_md._graph.edges())
    assert ("celery_worker", "postgres") in list(from_md._graph.edges())

    from_json = TopologyGraph(name="json")
    from_json.load_from_json(_JSON)
    blast = from_json.get_blast_radius("celery_worker", max_hops=2)
    assert blast.upstream_callers == ["fastapi"]
    assert blast.downstream_dependencies == ["postgres"]
    assert "fastapi" in blast.to_llm_context()

    unknown = from_json.get_blast_radius("does-not-exist")
    assert unknown.upstream_callers == []
    assert unknown.downstream_dependencies == []


def test_blast_radius_respects_max_hops() -> None:
    topo = TopologyGraph()
    topo.load_from_markdown(_MARKDOWN)
    one_hop = topo.get_blast_radius("fastapi", max_hops=1)
    assert "celery_worker" in one_hop.downstream_dependencies
    assert "postgres" not in one_hop.downstream_dependencies

    two_hops = topo.get_blast_radius("fastapi", max_hops=2)
    assert "postgres" in two_hops.downstream_dependencies


@pytest.mark.asyncio
async def test_load_from_file_and_action_node(tmp_path) -> None:
    path = tmp_path / "architecture.md"
    path.write_text(_MARKDOWN, encoding="utf-8")
    topo = TopologyGraph()
    await topo.load_from_file(path)
    assert not topo.is_empty

    json_path = tmp_path / "architecture.json"
    json_path.write_text(_JSON, encoding="utf-8")
    topo_json = TopologyGraph()
    await topo_json.load_from_file(json_path)
    assert topo_json.edge_count == 2

    seen: list[str] = []

    async def restart_worker(event: UnifiedTelemetrySchema, triage: TriageResult) -> None:
        seen.append(f"{event.service_name}:{triage.error_category}")

    agent = SentinelAgent(runbook_dir=tmp_path, topology=topo)
    agent.register_remediation("BROKER_*", min_criticality=7)(restart_worker)

    state = await action_node(
        {
            "event": _event(),
            "triage_result": _triage(),
            "agent_ref": agent,
            "_handler_timeout": 5.0,
        }
    )
    assert state["action_successful"] is True
    assert seen == ["celery_worker:BROKER_CONNECTION"]
    assert state["execution_report"].succeeded == 1


@pytest.mark.asyncio
async def test_action_node_reports_handler_failure() -> None:
    async def boom(event: UnifiedTelemetrySchema, triage: TriageResult) -> None:
        raise RuntimeError("restart refused")

    agent = SentinelAgent(runbook_dir=".")
    agent.register_remediation("*", min_criticality=1)(boom)

    state = await action_node(
        {
            "event": _event(),
            "triage_result": _triage(),
            "agent_ref": agent,
        }
    )
    assert state["action_successful"] is False
    assert state["execution_report"].failed == 1


@pytest.mark.asyncio
async def test_run_triage_injects_trace_runbook_and_blast_radius(tmp_path) -> None:
    topo = TopologyGraph()
    topo.load_from_markdown(_MARKDOWN)

    retriever = HybridRetriever(embeddings=FakeEmbeddings(size=32))
    await retriever.add_documents(
        [
            Document(
                page_content=(
                    "SOP: celery broker connection lost. "
                    "Restart the celery_worker and verify postgres."
                ),
                metadata={"doc_id": "broker.md#chunk0", "source": "broker.md"},
            )
        ]
    )

    expected = _triage()
    llm = _FakeLLM(expected)
    event = _event()
    update = await run_triage(
        {
            "event": event,
            "_retriever": retriever,
            "_topology": topo,
            "_blast_radius_hops": 2,
            "criticality_threshold": 8,
        },
        llm,
    )

    assert update["triage_result"] is expected
    assert llm.structured_kwargs["schema"] is TriageResult
    system = llm.calls["system"]
    assert "worker lost connection to broker" in system
    assert "celery broker connection lost" in system.lower() or "broker" in system.lower()
    assert "fastapi" in system
    assert "postgres" in system
    assert update["triage_result"].criticality == 9


@pytest.mark.asyncio
async def test_compiled_graph_routes_to_action_and_runs_handler(tmp_path) -> None:
    topo = TopologyGraph()
    topo.load_from_json(_JSON)
    retriever = HybridRetriever(embeddings=FakeEmbeddings(size=32))
    await retriever.add_documents(
        [
            Document(
                page_content="Restart celery_worker when the broker connection drops.",
                metadata={"doc_id": "sop.md#chunk0", "source": "sop.md"},
            )
        ]
    )

    ran: list[str] = []

    async def restart_worker(event: UnifiedTelemetrySchema, triage: TriageResult) -> None:
        ran.append(triage.error_category)

    agent = SentinelAgent(
        runbook_dir=tmp_path,
        criticality_threshold=8,
        topology=topo,
        retriever=retriever,
        llm_model="fake",
    )
    agent.register_remediation("BROKER_*", min_criticality=8)(restart_worker)

    graph = build_graph(
        criticality_threshold=8,
        llm_model="fake",
        runbook_dir=tmp_path,
        retriever=retriever,
        topology=topo,
        llm=_FakeLLM(_triage()),
    )

    final = await run_pipeline(event=_event(), agent=agent, graph=graph)
    assert final["action_successful"] is True
    assert ran == ["BROKER_CONNECTION"]
    assert final["notification_sent"] is False
    assert final["retrieved_runbook_id"] == "sop.md"
    assert final["triage_result"].decision == TriageDecision.REMEDIATE


@pytest.mark.asyncio
async def test_compiled_graph_routes_below_threshold_to_notify(tmp_path) -> None:
    graph = build_graph(
        criticality_threshold=8,
        llm_model="fake",
        runbook_dir=tmp_path,
        topology=TopologyGraph(),
        llm=_FakeLLM(
            _triage(criticality=3, decision=TriageDecision.NOTIFY, category="MINOR")
        ),
    )
    agent = SentinelAgent(runbook_dir=tmp_path, criticality_threshold=8)

    async def should_not_run(event: UnifiedTelemetrySchema, triage: TriageResult) -> None:
        raise AssertionError("remediation must not run below threshold")

    agent.register_remediation("*", min_criticality=1)(should_not_run)

    final = await run_pipeline(event=_event(), agent=agent, graph=graph)
    assert final["notification_sent"] is True
    assert final.get("action_successful") is None


def test_registration_matches_pattern() -> None:
    reg = RemediationRegistration(
        handler=lambda event, triage: asyncio.sleep(0),  # type: ignore[arg-type, return-value]
        trigger_pattern="BROKER_*",
        min_criticality=7,
        name="restart",
    )
    assert reg.matches(_triage())
    assert not reg.matches(_triage(criticality=2))
    assert not reg.matches(_triage(category="OOM"))
