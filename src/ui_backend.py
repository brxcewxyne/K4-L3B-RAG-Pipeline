"""UI-only placeholder backend boundary for the PUBG RAG chatbot.

STRICT SCOPE: frontend shell only. Do NOT implement retrieval, embeddings,
BM25, RRF, LLM generation, or evaluation here.

The real backend (Tasks 5-10) will replace :func:`ask_question` later with
something equivalent to::

    from src.task10_generation import generate_with_citation

    def ask_question(query: str, top_k: int = 5) -> dict:
        return generate_with_citation(query, top_k=top_k)

Until then, :func:`handle_query` / :func:`ask_question` return clearly
labelled mock data so the UI states (empty / loading / answer / sources)
can be demonstrated without implying a live RAG pipeline.
"""

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

# Retrieval-flow trace boundary (future Tasks 5-7).
# get_retrieval_trace() renders the [Query -> Dense + BM25 -> RRF -> Context]
# debug tab. It returns DEMO data until the real retrieval pipeline exists.
TRACE_BACKEND_CONNECTED: bool = False
TRACE_ENTRYPOINT: str = (
    "src.task9_retrieval_pipeline:retrieve "
    "(+ src.task5_semantic_search, src.task6_lexical_search, "
    "src.task7_reranking)"
)


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


def _demo_trace(query: str, top_k: int) -> dict:
    """Clearly-labelled dummy retrieval trace for the flow tab (UI preview).

    Shape is the contract future Tasks 5-7 will populate — layers stay
    separate (dense ranks, bm25 ranks, rrf fusion, selected context).
    Scores across branches are NOT comparable; each panel labels its own
    metric (cosine similarity vs BM25 score vs RRF score).
    """
    _ = top_k
    cleaned = (query or "").strip() or "Stream sniping có vi phạm luật PUBG không?"
    dense = [
        {"chunk_id": "pubg_rules_of_conduct_vi::chunk-12",
         "title": "Quy tắc ứng xử PUBG (demo)",
         "source": "KRAFTON, Inc.", "language": "vi",
         "authority_level": "primary", "score": 0.82, "rank": 1,
         "preview": "Không sử dụng các chương trình hoặc thiết bị phần cứng trái phép…"},
        {"chunk_id": "pubg_asia_stars_himass_tanvuu_investigation_vi::chunk-3",
         "title": "Thông báo kết quả điều tra PUBG Asia Stars (demo)",
         "source": "PUBG / KRAFTON", "language": "vi",
         "authority_level": "primary", "score": 0.76, "rank": 2,
         "preview": "Hành vi sử dụng thông tin bên ngoài game (stream sniping)…"},
        {"chunk_id": "justice_for_himass_tanvuu_vi::chunk-1",
         "title": "Justice for Himass & TanVuu (demo)",
         "source": "Independent source", "language": "vi",
         "authority_level": "secondary", "score": 0.61, "rank": 3,
         "preview": "Yêu cầu xem xét lại mức độ xử lý, bảo đảm quy trình công bằng…"},
    ]
    bm25 = [
        {"chunk_id": "pubg_rules_of_conduct_vi::chunk-12",
         "title": "Quy tắc ứng xử PUBG (demo)",
         "source": "KRAFTON, Inc.", "score": 7.42, "rank": 1,
         "matched_terms": ["stream", "sniping", "gian lận"]},
        {"chunk_id": "pubg_report_cheating_bug_abuse_vi::chunk-0",
         "title": "How to report cheating/hacking (demo)",
         "source": "PUBG Support", "score": 5.18, "rank": 2,
         "matched_terms": ["report", "cheating"]},
        {"chunk_id": "pubg_ban_penalty_information_vi::chunk-2",
         "title": "I want to know more about bans (demo)",
         "source": "PUBG Support", "score": 3.96, "rank": 3,
         "matched_terms": ["ban"]},
    ]
    rrf = [
        {"chunk_id": "pubg_rules_of_conduct_vi::chunk-12",
         "title": "Quy tắc ứng xử PUBG (demo)",
         "source": "KRAFTON, Inc.", "rrf_score": round(1 / 61 + 1 / 61, 6),
         "dense_rank": 1, "bm25_rank": 1, "final_rank": 1},
        {"chunk_id": "pubg_asia_stars_himass_tanvuu_investigation_vi::chunk-3",
         "title": "Thông báo kết quả điều tra PUBG Asia Stars (demo)",
         "source": "PUBG / KRAFTON", "rrf_score": round(1 / 62, 6),
         "dense_rank": 2, "bm25_rank": None, "final_rank": 2},
        {"chunk_id": "pubg_report_cheating_bug_abuse_vi::chunk-0",
         "title": "How to report cheating/hacking (demo)",
         "source": "PUBG Support", "rrf_score": round(1 / 62, 6),
         "dense_rank": None, "bm25_rank": 2, "final_rank": 3},
    ]
    return {
        "query": cleaned,
        "dense": {"metric": "cosine similarity", "results": dense},
        "bm25": {"metric": "BM25 score", "results": bm25},
        "rrf": {"metric": "RRF score (k=60)",
                "note": "RRF gộp thứ hạng (rank), không so sánh điểm thô "
                        "giữa cosine và BM25.",
                "results": rrf},
        "selected_context": [
            {"chunk_id": r["chunk_id"], "title": r["title"],
             "final_rank": r["final_rank"]} for r in rrf
        ],
        "is_mock": True,
        "backend_connected": False,
    }


def get_retrieval_trace(query: str, top_k: int = 5) -> dict:
    """Return the retrieval trace for the flow tab (demo until Tasks 5-7).

    FUTURE HOOK: build this dict from ``retrieve()`` +
    per-branch ranks instead of :func:`_demo_trace`. Keep the same keys so
    the tab renders without redesign.
    """
    return _demo_trace(query, top_k)
