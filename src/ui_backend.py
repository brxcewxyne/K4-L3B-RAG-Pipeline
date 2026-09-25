"""Chat preview and live Task 5-7 retrieval adapter."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Future connection point
# ---------------------------------------------------------------------------
# Replace the body of `ask_question` with a call to
# `src.task10_generation.generate_with_citation(query, top_k=top_k)`.
# Keep the return shape compatible with `GenerationResult`
# (answer: str, sources: list[SearchResult], retrieval_source: str).
# ---------------------------------------------------------------------------

BACKEND_CONNECTED: bool = False
BACKEND_ENTRYPOINT: str = "src.task10_generation:generate_with_citation"

# Retrieval-flow trace boundary (Tasks 5-7).
# get_retrieval_trace() renders the [Query -> Dense + BM25 -> RRF -> Context]
# debug tab using the existing Task 4 corpus.
TRACE_BACKEND_CONNECTED: bool = True
TRACE_ENTRYPOINT: str = "semantic_search + lexical_search + rerank_rrf"


def _demo_sources() -> list[dict]:
    """Clearly-labelled dummy citation data for UI preview only."""
    return [
        {
            "title": "Quy tắc ứng xử PUBG (demo)",
            "publisher": "PUBG / KRAFTON",
            "url": "https://www.pubg.com/",
            "source_type": "official",
            "authority_level": "Official · Primary",
            "language": "vi",
            "snippet": "Dữ liệu minh họa giao diện — chưa phải kết quả retrieval thật.",
        },
        {
            "title": "Justice for Himass & TanVuu (demo)",
            "publisher": "Independent source",
            "url": "https://example.com/himass-tanvuu",
            "source_type": "independent",
            "authority_level": "Independent · Secondary",
            "language": "vi",
            "snippet": "Dữ liệu minh họa giao diện — chưa phải kết quả retrieval thật.",
        },
    ]


def ask_question(query: str, top_k: int = 5) -> dict:
    """Placeholder backend hook. TODO: connect RAG pipeline later.

    Args:
        query: Raw user question from the composer.
        top_k: Reserved for future retrieval depth (currently unused).

    Returns:
        Dict with keys ``answer``, ``sources``, ``retrieval_method``,
        ``retrieval_source`` and ``is_mock`` (always True for now).
    """
    _ = (top_k,)  # reserved for future retrieval depth
    cleaned = (query or "").strip()
    sources = _demo_sources()
    answer = (
        "Đây là **giao diện xem trước (UI Preview)** — backend RAG chưa được "
        "kết nối nên tôi chưa thể tra cứu nguồn thật.\n\n"
        f"Bạn vừa hỏi: _{cleaned}_.\n\n"
        "Khi pipeline hoàn tất, câu trả lời tại đây sẽ gồm:\n\n"
        "- Nội dung tổng hợp có citation `[1]`, `[2]`\n"
        "- Thẻ **Sources** với Official / Independent\n"
        "- Thông tin retrieval (Dense / BM25 / Hybrid)\n\n"
        "Các thẻ *Sources* bên dưới là **dữ liệu demo** chỉ để minh họa bố cục."
    )
    return {
        "answer": answer,
        "sources": sources,
        "retrieval_method": "hybrid (demo)",
        "retrieval_source": "none",
        "is_mock": True,
    }


def handle_query(query: str, top_k: int = 5) -> dict:
    """Alias kept for convenience; identical to :func:`ask_question`."""
    return ask_question(query, top_k=top_k)


def filter_sources(sources: list[dict], source_filter: str) -> list[dict]:
    """UI-only source filter. ``source_filter`` in all/official/independent."""
    normalized = (source_filter or "all").lower()
    if normalized in {"official"}:
        return [s for s in sources if s.get("source_type") == "official"]
    if normalized in {"independent"}:
        return [s for s in sources if s.get("source_type") == "independent"]
    return list(sources)


def get_retrieval_trace(query: str, top_k: int = 5) -> dict:
    """Run Tasks 5-7 against the existing Task 4 collection; never fake results."""
    from src.task4_chunking_indexing import get_collection
    from src.task5_semantic_search import semantic_search
    from src import task6_lexical_search as lexical
    from src.task7_reranking import rerank_rrf

    query = (query or "").strip()
    if not query:
        raise ValueError("Enter a retrieval query.")
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    collection = get_collection()
    stored = collection.get(include=["documents", "metadatas"])
    if not stored["ids"]:
        raise ValueError("rag_documents is empty.")
    lexical.CORPUS = [
        {"id": cid, "content": content, "metadata": metadata or {}}
        for cid, content, metadata in zip(
            stored["ids"], stored["documents"], stored["metadatas"])
    ]
    depth = min(max(10, top_k), len(stored["ids"]))
    dense = semantic_search(query, top_k=depth)
    bm25 = lexical.lexical_search(query, top_k=depth)
    fused = rerank_rrf([dense, bm25], top_k=top_k)
    dense_ranks = {item["id"]: rank for rank, item in enumerate(dense, 1)}
    bm25_ranks = {item["id"]: rank for rank, item in enumerate(bm25, 1)}

    def fields(item):
        metadata = item.get("metadata") or {}
        return {"chunk_id": item["id"], "content": item["content"],
                **{key: metadata[key] for key in
                   ("title", "source", "language", "authority_level")
                   if key in metadata}}

    def ranked(items):
        return [{**fields(item), "score": item["score"], "rank": rank,
                 "preview": item["content"]}
                for rank, item in enumerate(items, 1)]

    return {
        "query": query,
        "dense": {"metric": "cosine similarity", "results": ranked(dense)},
        "bm25": {"metric": "BM25 score", "results": ranked(bm25)},
        "rrf": {"metric": "RRF score (k=60)", "results": [
            {**fields(item), "rrf_score": item["score"], "final_rank": rank,
             "dense_rank": dense_ranks.get(item["id"]),
             "bm25_rank": bm25_ranks.get(item["id"])}
            for rank, item in enumerate(fused, 1)]},
        "selected_context": [{**fields(item), "final_rank": rank}
                             for rank, item in enumerate(fused, 1)],
        "is_mock": False,
        "backend_connected": True,
    }
