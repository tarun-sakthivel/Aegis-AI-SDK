"""
sentinel_sre/ingestion/context_graph.py
─────────────────────────────────────────────────────────────────────────────
Service topology graph for blast-radius analysis during incident triage.

The ``TopologyGraph`` maintains a directed graph of microservice dependencies
using ``networkx``.  It is intentionally in-process and requires no external
database or service-mesh agent – topology is loaded from either a JSON spec
or a Markdown architecture document that developers already maintain.

Blast-radius semantics
──────────────────────
Given a failing service S, the LLM triage node needs to know:

• **Upstream callers** (services that call S) – they may start returning errors
  to their own clients because S is degraded.
• **Downstream dependencies** (services S calls) – the root cause may lie
  further down the call chain.

Both sets are bounded by ``max_hops`` to prevent the context from exploding
on large graphs.

Input formats
─────────────
JSON (recommended for programmatic tooling):

    {
      "services": ["api", "worker", "db", "cache"],
      "edges": [
        {"from": "api",    "to": "worker"},
        {"from": "api",    "to": "cache"},
        {"from": "worker", "to": "db"}
      ]
    }

Markdown (convenient for teams using architecture-as-docs):

    ## Services
    - api
    - worker
    - db

    ## Dependencies
    - api -> worker
    - api -> cache
    - worker -> db

Both ``->`` and ``→`` arrow styles are accepted in Markdown.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger("sentinel_sre.context_graph")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlastRadius:
    """Result of a ``get_blast_radius`` query.

    Attributes
    ----------
    origin_service:
        The service name the query was anchored on.
    upstream_callers:
        Services that have a directed path **to** ``origin_service`` within
        ``max_hops``.  These are at risk of cascading failures.
    downstream_dependencies:
        Services that ``origin_service`` has a directed path **to** within
        ``max_hops``.  The root cause may live here.
    max_hops:
        The hop limit used for this query.
    all_affected:
        Union of ``upstream_callers`` and ``downstream_dependencies``.
    """

    origin_service: str
    upstream_callers: list[str]
    downstream_dependencies: list[str]
    max_hops: int

    @property
    def all_affected(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for svc in self.upstream_callers + self.downstream_dependencies:
            if svc not in seen:
                seen.add(svc)
                result.append(svc)
        return result

    def to_llm_context(self) -> str:
        """Render a compact, human-readable string for LLM prompt injection."""
        lines: list[str] = [
            f"=== BLAST RADIUS (origin: {self.origin_service}, max_hops={self.max_hops}) ===",
            f"Upstream callers  : {', '.join(self.upstream_callers) or 'none'}",
            f"Downstream deps   : {', '.join(self.downstream_dependencies) or 'none'}",
            f"All affected      : {', '.join(self.all_affected) or 'none'}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_service": self.origin_service,
            "upstream_callers": self.upstream_callers,
            "downstream_dependencies": self.downstream_dependencies,
            "max_hops": self.max_hops,
            "all_affected": self.all_affected,
        }


# ---------------------------------------------------------------------------
# Markdown parser helpers
# ---------------------------------------------------------------------------

# Accepts:  "api -> worker",  "api → worker",  "api-->worker"  (spaces optional)
_EDGE_RE = re.compile(
    r"^\s*[-*]?\s*([a-zA-Z0-9_.\-]+)\s*[-–—]{0,2}[>→]\s*([a-zA-Z0-9_.\-]+)\s*$"
)
_SERVICE_LIST_RE = re.compile(r"^\s*[-*]\s*([a-zA-Z0-9_.\-]+)\s*$")


def _parse_markdown(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Extract service names and directed edges from a Markdown topology doc.

    Returns
    -------
    (services, edges)
        ``services`` – list of explicit service names (may be empty if only
        edges section is present; services will then be inferred from edges).
        ``edges`` – list of ``(from_service, to_service)`` tuples.
    """
    services: list[str] = []
    edges: list[tuple[str, str]] = []

    in_services_section = False
    in_deps_section = False

    for line in text.splitlines():
        heading_lower = line.strip().lower().lstrip("#").strip()

        # Detect section headers
        if line.strip().startswith("#"):
            if any(kw in heading_lower for kw in ("service", "component", "node")):
                in_services_section = True
                in_deps_section = False
                continue
            elif any(kw in heading_lower for kw in ("depend", "edge", "link", "connect", "relation")):
                in_deps_section = True
                in_services_section = False
                continue
            else:
                # Unknown heading – leave current section active but don't reset
                pass

        # Try edge first (works in both sections)
        edge_match = _EDGE_RE.match(line)
        if edge_match:
            src, dst = edge_match.group(1), edge_match.group(2)
            edges.append((src, dst))
            continue

        # Service name bullet in services section
        if in_services_section:
            svc_match = _SERVICE_LIST_RE.match(line)
            if svc_match:
                services.append(svc_match.group(1))

    return services, edges


# ---------------------------------------------------------------------------
# TopologyGraph
# ---------------------------------------------------------------------------


class TopologyGraph:
    """In-process directed service dependency graph.

    Parameters
    ----------
    name:
        Human-readable label for this topology (used in log messages).

    Example
    -------
    .. code-block:: python

        topo = TopologyGraph()
        await topo.load_from_file(Path("./architecture.json"))

        blast = topo.get_blast_radius("worker", max_hops=2)
        print(blast.to_llm_context())
    """

    def __init__(self, name: str = "default") -> None:
        self._name = name
        self._graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_from_dict(self, spec: dict[str, Any]) -> None:
        """Populate the graph from a plain Python dictionary.

        Expected schema::

            {
              "services": ["api", "worker", ...],   # optional – inferred from edges if absent
              "edges": [
                {"from": "api", "to": "worker"},
                ...
              ]
            }

        ``metadata`` keys on each edge (e.g. ``"protocol"``, ``"timeout_ms"``)
        are stored as edge attributes and surfaced in ``get_blast_radius``.
        """
        # Add explicit service nodes
        for svc in spec.get("services", []):
            self._graph.add_node(str(svc))

        # Add edges
        for edge in spec.get("edges", []):
            src = str(edge["from"])
            dst = str(edge["to"])
            attrs = {k: v for k, v in edge.items() if k not in ("from", "to")}
            self._graph.add_edge(src, dst, **attrs)

        logger.info(
            "sentinel_sre.context_graph [%s]: loaded %d node(s), %d edge(s) from dict.",
            self._name,
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    def load_from_json(self, json_text: str) -> None:
        """Parse *json_text* and delegate to ``load_from_dict``."""
        try:
            spec = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid topology JSON: {exc}") from exc
        self.load_from_dict(spec)

    def load_from_markdown(self, markdown_text: str) -> None:
        """Parse a Markdown architecture document and build the graph.

        Supports both explicit ``## Services`` / ``## Dependencies`` section
        conventions and free-form arrow notation (``A -> B``) anywhere in the
        document.
        """
        services, edges = _parse_markdown(markdown_text)

        for svc in services:
            self._graph.add_node(svc)
        for src, dst in edges:
            self._graph.add_edge(src, dst)

        logger.info(
            "sentinel_sre.context_graph [%s]: parsed Markdown – "
            "%d explicit service(s), %d edge(s) → graph has %d node(s), %d edge(s).",
            self._name,
            len(services),
            len(edges),
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    async def load_from_file(self, path: str | Path) -> None:
        """Load topology from a ``.json`` or ``.md`` / ``.txt`` file.

        File format is inferred from the extension.  ``.json`` files are parsed
        as JSON topology specs; all other extensions are parsed as Markdown.

        Parameters
        ----------
        path:
            Absolute or relative path to the topology file.
        """
        import asyncio

        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"sentinel_sre.context_graph: topology file not found: {path}"
            )

        text: str = await asyncio.to_thread(path.read_text, encoding="utf-8")

        if path.suffix.lower() == ".json":
            self.load_from_json(text)
        else:
            self.load_from_markdown(text)

        logger.info(
            "sentinel_sre.context_graph [%s]: loaded topology from %s.",
            self._name,
            path,
        )

    # ------------------------------------------------------------------
    # Blast-radius query
    # ------------------------------------------------------------------

    def get_blast_radius(
        self,
        service_name: str,
        *,
        max_hops: int = 2,
    ) -> BlastRadius:
        """Return the upstream callers and downstream dependencies of *service_name*.

        Uses BFS (via ``nx.single_source_shortest_path``) bounded by
        ``max_hops`` in both directions.

        Parameters
        ----------
        service_name:
            The name of the failing / degraded service to anchor the query on.
        max_hops:
            Maximum number of hops to traverse in each direction.  ``1`` means
            only direct neighbours.  ``2`` (default) includes neighbours-of-neighbours.

        Returns
        -------
        A ``BlastRadius`` dataclass.  When ``service_name`` is not in the graph
        a ``BlastRadius`` with empty lists is returned and a WARNING is logged.
        """
        if service_name not in self._graph:
            logger.warning(
                "sentinel_sre.context_graph: service %r not found in topology "
                "(%d known services). Returning empty blast radius.",
                service_name,
                self._graph.number_of_nodes(),
            )
            return BlastRadius(
                origin_service=service_name,
                upstream_callers=[],
                downstream_dependencies=[],
                max_hops=max_hops,
            )

        # Downstream: traverse forward edges (what does service_name call?)
        downstream_map = nx.single_source_shortest_path_length(
            self._graph, service_name, cutoff=max_hops
        )
        downstream = sorted(
            svc for svc in downstream_map if svc != service_name
        )

        # Upstream: traverse edges on the REVERSED graph (who calls service_name?)
        reversed_graph = self._graph.reverse(copy=False)
        upstream_map = nx.single_source_shortest_path_length(
            reversed_graph, service_name, cutoff=max_hops
        )
        upstream = sorted(
            svc for svc in upstream_map if svc != service_name
        )

        logger.debug(
            "sentinel_sre.context_graph: blast_radius(%r, max_hops=%d) → "
            "upstream=%s, downstream=%s.",
            service_name,
            max_hops,
            upstream,
            downstream,
        )

        return BlastRadius(
            origin_service=service_name,
            upstream_callers=upstream,
            downstream_dependencies=downstream,
            max_hops=max_hops,
        )

    # ------------------------------------------------------------------
    # Graph mutation helpers (useful for dynamic topology updates)
    # ------------------------------------------------------------------

    def add_service(self, name: str, **attrs: Any) -> None:
        """Add a service node to the graph at runtime."""
        self._graph.add_node(name, **attrs)
        logger.debug("sentinel_sre.context_graph: added service %r.", name)

    def add_dependency(
        self,
        source: str,
        target: str,
        **attrs: Any,
    ) -> None:
        """Add a directed dependency edge ``source → target``."""
        self._graph.add_edge(source, target, **attrs)
        logger.debug(
            "sentinel_sre.context_graph: added edge %r → %r.", source, target
        )

    def remove_service(self, name: str) -> None:
        """Remove a service and all its edges."""
        if name in self._graph:
            self._graph.remove_node(name)
            logger.debug("sentinel_sre.context_graph: removed service %r.", name)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def services(self) -> list[str]:
        """Sorted list of all service names in the graph."""
        return sorted(self._graph.nodes())

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    @property
    def is_empty(self) -> bool:
        return self._graph.number_of_nodes() == 0

    def to_adjacency_dict(self) -> dict[str, list[str]]:
        """Return ``{service: [dependencies]}`` for serialisation."""
        return {node: sorted(self._graph.successors(node)) for node in self._graph.nodes()}

    def __repr__(self) -> str:
        return (
            f"TopologyGraph(name={self._name!r}, "
            f"services={self._graph.number_of_nodes()}, "
            f"edges={self._graph.number_of_edges()})"
        )
