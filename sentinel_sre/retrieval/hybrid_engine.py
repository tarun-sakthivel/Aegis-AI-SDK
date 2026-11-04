"""
sentinel_sre/retrieval/hybrid_engine.py
─────────────────────────────────────────────────────────────────────────────
Hybrid Retrieval Engine: BM25 (sparse) + InMemoryVectorStore (dense) fused
via Reciprocal Rank Fusion (RRF).

Design rationale
────────────────
• **BM25** (``rank_bm25.BM25Okapi``) excels at exact keyword / error-code
  matching, e.g. "CUDA OOM", "psycopg OperationalError", "HTTP 503".
• **Dense vector search** (LangChain ``InMemoryVectorStore``) captures
  semantic similarity even when terminology differs between the error message
  and the runbook.
• **RRF fusion** (Cormack et al., 2009) combines both ranked lists without
  requiring score normalisation, using the formula:
      RRF(d) = Σ  1 / (k + rank_i(d))
  where k=60 is the standard constant.

The retriever operates purely in-process and requires no external service
(no Postgres, no Redis).  Corpus documents are ingested via ``add_documents``
and indices are rebuilt atomically using a read-write lock so live queries are
never blocked on index construction.

Thread / async safety
─────────────────────
• BM25 index rebuild is CPU-bound and synchronous; it runs in an executor
  thread via ``asyncio.to_thread`` so the event loop is never blocked.
• Vector store operations are async-native via ``langchain_core``'s
  ``InMemoryVectorStore``.
• A ``asyncio.Lock`` guards index swaps so concurrent retrievals see either
  the old or the new index atomically.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from rank_bm25 import BM25Okapi

logger = logging.getLogger("sentinel_sre.retrieval")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievedChunk:
    """A single result returned by the hybrid retriever.

    Attributes
    ----------
    doc_id:
        Unique identifier for the source document chunk (usually
        ``<filename>#chunk<n>``).
    content:
        The raw text content of the chunk.
    source:
        Original file path or logical name the chunk was loaded from.
    rrf_score:
        Fused RRF score (higher = better match).  Comparable only within a
        single query result set.
    bm25_rank:
        1-based BM25 rank, or ``None`` if BM25 did not return this chunk.
    vector_rank:
        1-based vector-search rank, or ``None`` if vector search did not
        return this chunk.
    metadata:
        Arbitrary key-value metadata forwarded from the source ``Document``.
    """

    doc_id: str
    content: str
    source: str
    rrf_score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tokeniser helpers
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenise(text: str) -> list[str]:
    """Case-fold and split text into alphanumeric tokens.

    Preserves underscores so that identifiers like ``cuda_oom`` and
    ``connection_refused`` are kept intact.
    """
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# BM25 index wrapper
# ---------------------------------------------------------------------------


@dataclass
class _BM25State:
    """Encapsulates one generation of the BM25 index.

    Swapped atomically on ``add_documents`` so in-flight queries always see a
    consistent snapshot.
    """

    index: BM25Okapi
    doc_ids: list[str]
    contents: dict[str, str]  # doc_id → raw text


def _build_bm25_state(documents: list[Document]) -> _BM25State:
    """Build a fresh ``_BM25State`` from a list of LangChain ``Document`` objects.

    This is intentionally a pure function so it can be safely executed in a
    thread pool without shared mutable state.
    """
    doc_ids: list[str] = []
    tokenised_corpus: list[list[str]] = []
    contents: dict[str, str] = {}

    for doc in documents:
        doc_id: str = doc.metadata.get("doc_id", doc.metadata.get("source", id(doc)))
        doc_ids.append(doc_id)
        tokenised_corpus.append(_tokenise(doc.page_content))
        contents[doc_id] = doc.page_content

    return _BM25State(
        index=BM25Okapi(tokenised_corpus),
        doc_ids=doc_ids,
        contents=contents,
    )


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Hybrid BM25 + dense vector retriever with RRF fusion.

    Parameters
    ----------
    embeddings:
        A LangChain ``Embeddings`` implementation used to build and query the
        vector store.  Typical choices:
        - ``langchain_openai.OpenAIEmbeddings`` (cloud)
        - ``langchain_community.embeddings.HuggingFaceEmbeddings`` (local)
        - ``langchain_core.embeddings.FakeEmbeddings`` (testing)
    rrf_k:
        RRF constant.  60 is the standard default from the original paper.
        Lower values amplify the effect of top-ranked documents.
    bm25_candidates:
        Number of BM25 candidates to retrieve before RRF fusion.
    vector_candidates:
        Number of vector-search candidates to retrieve before RRF fusion.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        rrf_k: int = 60,
        bm25_candidates: int = 10,
        vector_candidates: int = 10,
    ) -> None:
        self._embeddings = embeddings
        self._rrf_k = rrf_k
        self._bm25_candidates = bm25_candidates
        self._vector_candidates = vector_candidates

        # Mutable state protected by a lock.
        self._lock = asyncio.Lock()
        self._bm25_state: _BM25State | None = None
        self._vector_store: InMemoryVectorStore | None = None
        self._document_count: int = 0

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    async def add_documents(self, documents: list[Document]) -> None:
        """Ingest a batch of documents and atomically rebuild both indices.

        Safe to call multiple times; each call **replaces** the existing index
        with a fresh one built from all provided documents (suitable for the
        startup ingestion pattern used by ``RunbookLoader``).

        Parameters
        ----------
        documents:
            A list of LangChain ``Document`` objects, each carrying a
            ``page_content`` string and a ``metadata`` dict that MUST contain
            at least a ``"doc_id"`` key (set by ``RunbookLoader``).
        """
        if not documents:
            logger.warning("sentinel_sre.retrieval: add_documents called with empty list.")
            return

        t0 = time.perf_counter()

        # ── Build BM25 index in a thread (CPU-bound) ─────────────────────────
        new_bm25_state: _BM25State = await asyncio.to_thread(
            _build_bm25_state, documents
        )

        # ── Build vector store (async I/O bound for remote embeddings) ────────
        new_vector_store = InMemoryVectorStore(embedding=self._embeddings)
        await new_vector_store.aadd_documents(documents)

        # ── Atomic swap under lock ────────────────────────────────────────────
        async with self._lock:
            self._bm25_state = new_bm25_state
            self._vector_store = new_vector_store
            self._document_count = len(documents)

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        logger.info(
            "sentinel_sre.retrieval: indexed %d chunk(s) in %.1f ms "
            "(BM25 vocab=%d tokens, vector_store=InMemory).",
            len(documents),
            elapsed_ms,
            sum(len(_tokenise(d.page_content)) for d in documents),
        )

    async def add_documents_incremental(self, documents: list[Document]) -> None:
        """Append new documents to the existing indices without full rebuild.

        Useful for hot-loading new runbooks at runtime without a service
        restart.  BM25 does not support incremental updates natively, so the
        full BM25 index is still rebuilt; however the vector store is updated
        incrementally using ``aadd_documents``.

        Parameters
        ----------
        documents:
            New documents to append.  Existing documents are retained.
        """
        if not documents:
            return

        async with self._lock:
            existing_contents = (
                list(self._bm25_state.contents.values()) if self._bm25_state else []
            )

        # Reconstruct full corpus for BM25.
        existing_docs = [
            Document(page_content=content) for content in existing_contents
        ]
        all_docs = existing_docs + documents

        new_bm25_state: _BM25State = await asyncio.to_thread(
            _build_bm25_state, all_docs
        )

        # Vector store supports native incremental additions.
        async with self._lock:
            if self._vector_store is None:
                self._vector_store = InMemoryVectorStore(embedding=self._embeddings)
            await self._vector_store.aadd_documents(documents)
            self._bm25_state = new_bm25_state
            self._document_count += len(documents)

        logger.info(
            "sentinel_sre.retrieval: incrementally added %d chunk(s) "
            "(total corpus size: %d).",
            len(documents),
            self._document_count,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Execute hybrid retrieval and return the top-k fused results.

        Both BM25 and vector searches are run concurrently via
        ``asyncio.gather``.  Results are fused with RRF and returned ordered
        by descending RRF score.

        Parameters
        ----------
        query:
            The free-text retrieval query (typically the exception type +
            message from a ``UnifiedTelemetrySchema``).
        top_k:
            Maximum number of ``RetrievedChunk`` objects to return.

        Returns
        -------
        A list of ``RetrievedChunk`` objects, ordered by descending RRF score.
        An empty list is returned when no documents have been ingested.

        Raises
        ------
        RuntimeError
            If the retriever is queried before any documents have been
            ingested (i.e., ``add_documents`` has never been called).
        """
        async with self._lock:
            bm25_state = self._bm25_state
            vector_store = self._vector_store

        if bm25_state is None or vector_store is None:
            logger.warning(
                "sentinel_sre.retrieval: retrieve() called before any documents "
                "were ingested; returning empty result set."
            )
            return []

        t0 = time.perf_counter()

        # ── Run both retrievers concurrently ──────────────────────────────────
        bm25_task = asyncio.to_thread(
            self._bm25_retrieve_sync,
            query,
            bm25_state,
            self._bm25_candidates,
        )
        vector_task = self._vector_retrieve_async(
            query, vector_store, self._vector_candidates
        )

        bm25_hits, vector_hits = await asyncio.gather(bm25_task, vector_task)

        # ── RRF fusion ────────────────────────────────────────────────────────
        fused = self._reciprocal_rank_fusion(
            bm25_hits, vector_hits, k=self._rrf_k
        )
        results = fused[:top_k]

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        logger.debug(
            "sentinel_sre.retrieval: query=%r → %d BM25 hit(s), %d vector hit(s), "
            "%d fused result(s) in %.1f ms.",
            query[:80],
            len(bm25_hits),
            len(vector_hits),
            len(results),
            elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------
    # Individual retriever backends
    # ------------------------------------------------------------------

    @staticmethod
    def _bm25_retrieve_sync(
        query: str,
        state: _BM25State,
        top_k: int,
    ) -> list[tuple[str, str, float]]:
        """Synchronous BM25 retrieval – intended to run in a thread pool.

        Returns
        -------
        List of ``(doc_id, content, score)`` tuples ordered by descending
        BM25 score.
        """
        query_tokens = _tokenise(query)
        scores: list[float] = state.index.get_scores(query_tokens).tolist()
        ranked = sorted(
            enumerate(scores), key=lambda t: t[1], reverse=True
        )[:top_k]
        results: list[tuple[str, str, float]] = []
        for idx, score in ranked:
            if score <= 0.0:
                # BM25 returns 0 for completely non-matching documents;
                # exclude them to avoid polluting the fused ranking.
                break
            doc_id = state.doc_ids[idx]
            results.append((doc_id, state.contents[doc_id], score))
        return results

    @staticmethod
    async def _vector_retrieve_async(
        query: str,
        store: InMemoryVectorStore,
        top_k: int,
    ) -> list[tuple[str, str, float]]:
        """Async dense-vector retrieval via ``InMemoryVectorStore``.

        Returns
        -------
        List of ``(doc_id, content, score)`` tuples ordered by descending
        cosine similarity score.
        """
        try:
            raw: list[tuple[Document, float]] = (
                await store.asimilarity_search_with_score(query, k=top_k)
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "sentinel_sre.retrieval: vector search failed; "
                "returning empty vector hits."
            )
            return []

        results: list[tuple[str, str, float]] = []
        for doc, score in raw:
            doc_id = doc.metadata.get("doc_id", doc.metadata.get("source", "unknown"))
            results.append((doc_id, doc.page_content, float(score)))
        return results

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[tuple[str, str, float]],
        vector_results: list[tuple[str, str, float]],
        *,
        k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Fuse BM25 and vector ranked lists using Reciprocal Rank Fusion.

        RRF score for a document d across retriever lists L₁…Lₙ is defined as:
            RRF(d) = Σᵢ 1 / (k + rankᵢ(d))

        where rankᵢ(d) is the 1-based position of d in list Lᵢ, and documents
        absent from a list contribute 0 to the sum (they are simply not in that
        list's ranking).

        Parameters
        ----------
        bm25_results:
            Ranked list of ``(doc_id, content, score)`` from BM25 retrieval.
            Index 0 = highest-scoring BM25 document.
        vector_results:
            Ranked list of ``(doc_id, content, score)`` from vector retrieval.
            Index 0 = highest-scoring vector document.
        k:
            RRF constant.  Defaults to ``self._rrf_k`` (60).

        Returns
        -------
        List of ``RetrievedChunk`` objects ordered by descending RRF score.
        """
        rrf_k = k if k is not None else self._rrf_k
        rrf_scores: dict[str, float] = {}
        bm25_ranks: dict[str, int] = {}
        vector_ranks: dict[str, int] = {}
        contents: dict[str, str] = {}

        for rank, (doc_id, content, _) in enumerate(bm25_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            bm25_ranks[doc_id] = rank
            contents[doc_id] = content

        for rank, (doc_id, content, _) in enumerate(vector_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            vector_ranks[doc_id] = rank
            contents.setdefault(doc_id, content)

        ordered = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        chunks: list[RetrievedChunk] = []
        for doc_id, rrf_score in ordered:
            content = contents[doc_id]
            # Extract source from the doc_id convention "<source>#chunk<n>".
            source = doc_id.split("#chunk")[0] if "#chunk" in doc_id else doc_id
            chunks.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    content=content,
                    source=source,
                    rrf_score=rrf_score,
                    bm25_rank=bm25_ranks.get(doc_id),
                    vector_rank=vector_ranks.get(doc_id),
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        """Total number of chunks currently indexed."""
        return self._document_count

    @property
    def is_ready(self) -> bool:
        """``True`` if at least one document has been ingested."""
        return self._bm25_state is not None and self._vector_store is not None

    def __repr__(self) -> str:
        return (
            f"HybridRetriever("
            f"docs={self._document_count}, "
            f"rrf_k={self._rrf_k}, "
            f"bm25_candidates={self._bm25_candidates}, "
            f"vector_candidates={self._vector_candidates})"
        )
