"""
sentinel_sre/ingestion/document_loader.py
─────────────────────────────────────────────────────────────────────────────
Runbook ingestion pipeline that loads ``.md`` and ``.txt`` files from a
directory tree, chunks them into retrieval-sized segments, and feeds them into
the ``HybridRetriever``.

Design goals
────────────
• **Zero-dependency chunking** – the splitter is implemented in pure Python
  with no additional third-party libraries beyond what is already declared in
  ``pyproject.toml``.
• **Idempotent** – calling ``load_and_index()`` replaces the retriever's index
  with a fresh one, making it safe to call on startup or whenever the runbook
  directory changes.
• **Async-first** – file I/O is dispatched via ``asyncio.to_thread`` so that
  large corpus scans do not block the event loop.
• **Rich metadata** – every chunk carries ``source``, ``doc_id``,
  ``chunk_index``, ``total_chunks``, and ``file_type`` metadata fields so the
  triage node can display meaningful runbook citations.

Chunking strategy
─────────────────
Documents are split on Markdown headings (``# / ## / ###``) first, then any
remaining blocks longer than ``chunk_size`` tokens are split with a sliding
window (``chunk_overlap`` token overlap).  This preserves semantic coherence
better than naïve fixed-size splitting.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document

from sentinel_sre.retrieval.hybrid_engine import HybridRetriever

logger = logging.getLogger("sentinel_sre.ingestion")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt"})

# Regex that matches any Markdown ATX heading (# … ######).
_HEADING_RE = re.compile(r"^#{1,6}\s+.+", re.MULTILINE)


# ---------------------------------------------------------------------------
# Chunking utilities
# ---------------------------------------------------------------------------


def _split_on_headings(text: str) -> list[str]:
    """Split a Markdown document at heading boundaries.

    Each returned segment starts with its heading (if any).  Empty segments
    and pure-whitespace segments are dropped.
    """
    # Find all heading start positions.
    boundaries = [m.start() for m in _HEADING_RE.finditer(text)]

    if not boundaries:
        return [text.strip()] if text.strip() else []

    segments: list[str] = []

    # Text before the first heading (preamble).
    preamble = text[: boundaries[0]].strip()
    if preamble:
        segments.append(preamble)

    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)

    return segments


def _token_count(text: str) -> int:
    """Rough token estimate: whitespace-delimited words.

    This is intentionally cheap; a production implementation would use the
    model's actual tokeniser (e.g. ``tiktoken``).
    """
    return len(text.split())


def _sliding_window_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Iterator[str]:
    """Yield fixed-size chunks with overlap from *text*.

    Splits on whitespace boundaries to avoid cutting mid-word.
    """
    words = text.split()
    step = max(1, chunk_size - chunk_overlap)
    start = 0
    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        yield " ".join(chunk_words)
        if start + chunk_size >= len(words):
            break
        start += step


def _chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split *text* into retrieval-sized chunks.

    Strategy (applied in order):
    1. Split on Markdown headings.
    2. For any heading-segment still larger than ``chunk_size`` tokens,
       apply sliding-window splitting.
    3. If a heading-segment is smaller than 50% of ``chunk_size``, merge it
       with the previous chunk to avoid micro-chunks.
    """
    heading_segments = _split_on_headings(text)
    if not heading_segments:
        return []

    raw_chunks: list[str] = []
    for segment in heading_segments:
        if _token_count(segment) > chunk_size:
            raw_chunks.extend(
                _sliding_window_split(segment, chunk_size, chunk_overlap)
            )
        else:
            raw_chunks.append(segment)

    # Merge micro-chunks into their predecessor.
    min_tokens = chunk_size // 2
    merged: list[str] = []
    for chunk in raw_chunks:
        if merged and _token_count(chunk) < min_tokens:
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)

    return [c for c in merged if c.strip()]


# ---------------------------------------------------------------------------
# Loader configuration
# ---------------------------------------------------------------------------


@dataclass
class LoaderConfig:
    """Configuration for the ``RunbookLoader``.

    Attributes
    ----------
    runbook_dir:
        Root directory to scan for ``.md`` / ``.txt`` runbook files.
        Sub-directories are traversed recursively.
    chunk_size:
        Target chunk size in whitespace-delimited tokens.  Controls the
        trade-off between retrieval granularity and context window usage.
        Default: 300 tokens (~225 words).
    chunk_overlap:
        Number of tokens of overlap between consecutive sliding-window chunks.
        Prevents context loss at chunk boundaries.  Default: 50 tokens.
    glob_patterns:
        File glob patterns to include.  Defaults to ``["**/*.md", "**/*.txt"]``.
    encoding:
        Text encoding used when reading files.  Default: ``"utf-8"``.
    skip_patterns:
        List of glob patterns relative to ``runbook_dir`` to exclude.
        Example: ``["**/draft_*", "**/archive/**"]``.
    """

    runbook_dir: str | Path = "./docs/runbooks"
    chunk_size: int = 300
    chunk_overlap: int = 50
    glob_patterns: list[str] = field(
        default_factory=lambda: ["**/*.md", "**/*.txt"]
    )
    encoding: str = "utf-8"
    skip_patterns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RunbookLoader
# ---------------------------------------------------------------------------


class RunbookLoader:
    """Scans a runbook directory, chunks documents, and populates a
    ``HybridRetriever`` index on startup.

    Parameters
    ----------
    retriever:
        The ``HybridRetriever`` instance to populate.  Typically the same
        instance referenced by the ``SentinelAgent`` and the LangGraph pipeline.
    config:
        ``LoaderConfig`` controlling directory path, chunking params, and
        file inclusion/exclusion rules.

    Example
    -------
    .. code-block:: python

        from langchain_openai import OpenAIEmbeddings
        from sentinel_sre.retrieval.hybrid_engine import HybridRetriever
        from sentinel_sre.ingestion.document_loader import RunbookLoader, LoaderConfig

        retriever = HybridRetriever(embeddings=OpenAIEmbeddings())
        loader = RunbookLoader(
            retriever=retriever,
            config=LoaderConfig(runbook_dir="./docs/runbooks", chunk_size=300),
        )

        # Call once at startup (inside an async context):
        await loader.load_and_index()
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        config: LoaderConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._config = config or LoaderConfig()
        self._runbook_dir = Path(self._config.runbook_dir).expanduser().resolve()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def load_and_index(self) -> int:
        """Discover, load, chunk, and index all runbook files.

        Performs all file I/O in a thread pool so the event loop is not
        blocked.  After this coroutine returns, ``self._retriever`` is ready
        to serve queries.

        Returns
        -------
        The total number of document chunks indexed.

        Raises
        ------
        FileNotFoundError
            If ``runbook_dir`` does not exist.
        """
        if not self._runbook_dir.exists():
            raise FileNotFoundError(
                f"sentinel_sre.ingestion: runbook_dir does not exist: "
                f"{self._runbook_dir}"
            )

        logger.info(
            "sentinel_sre.ingestion: scanning %s for runbooks …", self._runbook_dir
        )

        # File discovery is fast; run in executor to stay non-blocking.
        file_paths: list[Path] = await asyncio.to_thread(self._discover_files)

        if not file_paths:
            logger.warning(
                "sentinel_sre.ingestion: no runbook files found in %s "
                "(patterns: %s).",
                self._runbook_dir,
                self._config.glob_patterns,
            )
            return 0

        logger.info(
            "sentinel_sre.ingestion: found %d file(s); loading …", len(file_paths)
        )

        # Load and chunk all files concurrently.
        load_tasks = [
            asyncio.to_thread(self._load_and_chunk_file, fp) for fp in file_paths
        ]
        results: list[list[Document] | None] = await asyncio.gather(
            *load_tasks, return_exceptions=False
        )

        documents: list[Document] = []
        for chunk_list in results:
            if chunk_list:
                documents.extend(chunk_list)

        if not documents:
            logger.warning(
                "sentinel_sre.ingestion: all files were empty or unreadable."
            )
            return 0

        await self._retriever.add_documents(documents)

        logger.info(
            "sentinel_sre.ingestion: indexed %d chunks from %d file(s).",
            len(documents),
            len(file_paths),
        )
        return len(documents)

    async def reload(self) -> int:
        """Trigger a full re-scan and re-index of the runbook directory.

        Useful for picking up new or updated runbooks at runtime without a
        service restart.  Equivalent to calling ``load_and_index()`` again.
        """
        logger.info("sentinel_sre.ingestion: reloading runbook corpus …")
        return await self.load_and_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_files(self) -> list[Path]:
        """Return all matching file paths under ``runbook_dir``.

        Executed in a thread pool by the caller.
        """
        skip_patterns = self._config.skip_patterns
        found: list[Path] = []

        for pattern in self._config.glob_patterns:
            for path in self._runbook_dir.glob(pattern):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                    continue
                relative = path.relative_to(self._runbook_dir)
                if any(relative.match(sp) for sp in skip_patterns):
                    logger.debug(
                        "sentinel_sre.ingestion: skipping %s (matches skip_pattern).", path
                    )
                    continue
                found.append(path)

        # Deterministic ordering for reproducible index content.
        return sorted(set(found))

    def _load_and_chunk_file(self, path: Path) -> list[Document] | None:
        """Read a single file, chunk it, and return a list of ``Document`` objects.

        Returns ``None`` on read errors (logged at WARNING level).  Executed
        in a thread pool by the caller.
        """
        try:
            text = path.read_text(encoding=self._config.encoding, errors="replace")
        except OSError:
            logger.warning(
                "sentinel_sre.ingestion: could not read %s; skipping.", path
            )
            return None

        if not text.strip():
            logger.debug("sentinel_sre.ingestion: %s is empty; skipping.", path)
            return None

        chunks = _chunk_text(
            text,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )

        if not chunks:
            return None

        # Derive a stable logical name relative to the runbook root.
        try:
            relative_name = str(path.relative_to(self._runbook_dir))
        except ValueError:
            relative_name = path.name

        documents: list[Document] = []
        total = len(chunks)
        for idx, chunk_text in enumerate(chunks):
            doc_id = f"{relative_name}#chunk{idx}"
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "doc_id": doc_id,
                        "source": relative_name,
                        "chunk_index": idx,
                        "total_chunks": total,
                        "file_type": path.suffix.lstrip(".").lower(),
                        "absolute_path": str(path),
                    },
                )
            )

        logger.debug(
            "sentinel_sre.ingestion: %s → %d chunk(s) "
            "(avg %.0f tokens/chunk).",
            relative_name,
            total,
            sum(_token_count(c) for c in chunks) / max(total, 1),
        )
        return documents

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def runbook_dir(self) -> Path:
        """Resolved absolute path of the runbook directory."""
        return self._runbook_dir

    @property
    def retriever(self) -> HybridRetriever:
        """The ``HybridRetriever`` instance being populated."""
        return self._retriever

    def __repr__(self) -> str:
        return (
            f"RunbookLoader("
            f"dir={self._runbook_dir}, "
            f"chunk_size={self._config.chunk_size}, "
            f"indexed={self._retriever.document_count} chunks)"
        )
