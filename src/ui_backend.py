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
