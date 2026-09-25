"""Reusable presentational helpers for the Streamlit UI shell.

These functions only build HTML strings / resolve assets — they never call
retrieval, embeddings, or LLMs. Keep them dependency-free (stdlib only) so
they are easy to unit-test and reuse after the backend is connected.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

EXAMPLE_PROMPTS: list[str] = [
    "Stream sniping có vi phạm luật PUBG không?",
    "PUBG quy định gì về sử dụng phần mềm gian lận?",
    "Nếu người khác cheat trên tài khoản của tôi thì sao?",
    "PUBG đã công bố gì về vụ Himass và TanVuu?",
]

SOURCE_FILTER_OPTIONS: list[tuple[str, str]] = [
    ("all", "Tất cả"),
    ("official", "Official"),
    ("independent", "Independent"),
]

_EMPTY_PORTRAIT_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='480' height='720'>"
    "<rect width='100%' height='100%' fill='#0b0e11'/>"
    "<circle cx='240' cy='270' r='86' fill='none' stroke='#2a3138' stroke-width='3'/>"
    "<path d='M120 620 C120 470 360 470 360 620' fill='none' stroke='#2a3138' stroke-width='3'/>"
    "</svg>"
)


def portrait_data_uri(image_path: str | Path | None) -> str | None:
    """Return a data URI for a local portrait, or None if missing."""
    if not image_path:
        return None
    path = Path(image_path)
    if not path.is_file():
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def empty_portrait_data_uri() -> str:
    encoded = base64.b64encode(_EMPTY_PORTRAIT_SVG.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def portrait_html(
    image_path: str | Path | None,
    *,
    name: str,
    subtitle: str = "PUBG Player",
    side: str = "left",
) -> str:
    """Build the side-portrait block (image blended into dark background)."""
    uri = portrait_data_uri(image_path) or empty_portrait_data_uri()
    safe_name = html.escape(name)
    safe_sub = html.escape(subtitle)
    safe_side = html.escape(side)
    missing_note = ""
    if portrait_data_uri(image_path) is None:
        missing_note = (
            "<div class='portrait-missing'>assets/ thiếu ảnh gốc — "
            "đang hiển thị khung chờ</div>"
        )
    return (
        f"<div class='portrait-col portrait-{safe_side}'>"
        f"<div class='portrait-frame'>"
        f"<img class='portrait-img' src='{uri}' alt='Chân dung {safe_name}' />"
        f"<div class='portrait-fade-inner portrait-fade-inner-{safe_side}'></div>"
        f"<div class='portrait-fade'></div>"
        f"</div>"
        f"<div class='portrait-label'>{safe_name}</div>"
        f"<div class='portrait-sub'>{safe_sub}</div>"
        f"{missing_note}"
        f"</div>"
    )


def source_badge(source_type: str) -> str:
    normalized = (source_type or "").lower()
    if normalized == "official":
        return "<span class='src-badge src-official'>Official · Primary</span>"
    return "<span class='src-badge src-independent'>Independent · Secondary</span>"


def source_card_html(index: int, source: dict) -> str:
    """Render one future RAG citation card (works with demo data too)."""
    title = html.escape(str(source.get("title", f"Nguồn {index}")))
    publisher = html.escape(str(source.get("publisher", "—")))
    url = str(source.get("url", "") or "")
    safe_url = html.escape(url)
    source_type = str(source.get("source_type", "independent"))
    authority = html.escape(str(source.get("authority_level", "")))
    language = html.escape(str(source.get("language", "")))
    snippet = html.escape(str(source.get("snippet", "")))
    meta_bits = " · ".join(b for b in [authority, language] if b)
    link_row = (
        f"<a class='src-url' href='{safe_url}' target='_blank' rel='noopener'>{safe_url}</a>"
        if url
        else "<span class='src-url src-no-url'>chưa có URL</span>"
    )
    return (
        f"<div class='src-card'>"
        f"<div class='src-index'>[{index}]</div>"
        f"<div class='src-body'>"
        f"<div class='src-title'>{title}</div>"
        f"<div class='src-pub'>{publisher}</div>"
        f"<div class='src-meta'>{source_badge(source_type)}"
        + (f"<span class='src-extra'>{meta_bits}</span>" if meta_bits else "")
        + f"</div>"
        + (f"<div class='src-snippet'>{snippet}</div>" if snippet else "")
        + f"<div class='src-link'>{link_row}</div>"
        f"</div>"
        f"</div>"
    )


def retrieval_info_html(retrieval_method: str = "", *, is_mock: bool = True) -> str:
    method = html.escape(retrieval_method or "hybrid")
    demo_tag = (
        "<span class='ret-pill ret-demo'>Dữ liệu demo — chưa retrieval thật</span>"
        if is_mock
        else "<span class='ret-pill ret-live'>RAG</span>"
    )
    return (
        f"<div class='retrieval-row'>"
        f"<span class='ret-label'>Retrieval</span>"
        f"<span class='ret-pill'>{method}</span>"
        f"{demo_tag}"
        f"</div>"
    )


def user_message_html(content: str) -> str:
    return (
        "<div class='msg msg-user'>"
        "<div class='msg-avatar msg-avatar-user'>B</div>"
        f"<div class='msg-bubble msg-bubble-user'>{html.escape(content)}</div>"
        "</div>"
    )


def assistant_message_html(    answer_markdown: str,
    sources: list[dict] | None = None,
    *,
    retrieval_method: str = "hybrid (demo)",
    is_mock: bool = True,
) -> str:
    """Assistant block: markdown answer rendered by Streamlit lives above this.

    This helper renders the *sources + retrieval* footer so message chrome
    stays consistent. (Kept separate from markdown to avoid nested HTML.)
    """
    _ = answer_markdown  # answer itself is rendered via st.markdown
    cards = "".join(
        source_card_html(s.get("citation_id", i + 1), s)
        for i, s in enumerate(sources or [])
    )
    sources_block = (
        f"<div class='src-list'><div class='src-heading'>Sources</div>{cards}</div>"
        if cards
        else "<div class='src-empty'>Chưa có nguồn trích dẫn.</div>"
    )
    return (
        "<div class='assistant-footer'>"
        f"{retrieval_info_html(retrieval_method, is_mock=is_mock)}"
        f"{sources_block}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Retrieval-flow tab (renders a trace dict, never computes
# Dense/BM25/RRF itself). Backend contract: src.ui_backend.get_retrieval_trace
# ---------------------------------------------------------------------------

def flow_banner_html(*, is_mock: bool = True) -> str:
    if not is_mock:
        return ("<div class='flow-banner'><span class='flow-banner-text'>"
                "LIVE · Dense + BM25 + RRF · rag_documents</span></div>")
    return (
        "<div class='flow-banner'>"
        "<span class='mock-tag'>UI PREVIEW</span>"
        "<span class='flow-banner-text'>Demo retrieval trace — "
        "backend <b>chưa kết nối</b> (Tasks 5–7 chưa triển khai). "
        "Mọi kết quả dưới đây là dữ liệu minh họa.</span>"
        "</div>"
    )


def flow_section_html(title: str, hint: str = "") -> str:
    safe = html.escape(title)
    sub = f"<div class='flow-hint'>{html.escape(hint)}</div>" if hint else ""
    return f"<div class='flow-section'><span>{safe}</span></div>{sub}"


def flow_diagram_html() -> str:
    return (
        "<div class='flow-diagram'>"
        "<div class='flow-node flow-node-query'>User Query</div>"
        "<div class='flow-arrow'>↓</div>"
        "<div class='flow-branches'>"
        "<div class='flow-branch'>"
        "<div class='flow-node'>Dense</div>"
        "<div class='flow-sub'>query embedding → ChromaDB → cosine</div>"
        "</div>"
        "<div class='flow-branch'>"
        "<div class='flow-node'>BM25</div>"
        "<div class='flow-sub'>query tokens → BM25 index → lexical score</div>"
        "</div>"
        "</div>"
        "<div class='flow-merge'>↘&nbsp;&nbsp;hai nhánh độc lập&nbsp;&nbsp;↙</div>"
        "<div class='flow-node flow-node-rrf'>RRF Fusion</div>"
        "<div class='flow-arrow'>↓</div>"
        "<div class='flow-node'>Final ranked chunks → LLM Context</div>"
        "</div>"
    )


def flow_query_html(query: str) -> str:
    return (
        "<div class='flow-query'>"
        "<span class='flow-query-label'>QUERY</span>"
        f"<span class='flow-query-text'>{html.escape(query or '—')}</span>"
        "</div>"
    )


def _flow_row(rank: int, title: str, chunk_id: str, score_html: str,
              meta_html: str, extra_html: str = "") -> str:
    return (
        "<div class='flow-row'>"
        f"<div class='flow-rank'>#{int(rank)}</div>"
        "<div class='flow-body'>"
        f"<div class='src-title'>{html.escape(title)}</div>"
        f"<div class='flow-chunk-id'>{html.escape(chunk_id)}</div>"
        f"<div class='flow-scores'>{score_html}</div>"
        f"<div class='flow-meta'>{meta_html}</div>"
        f"{extra_html}"
        "</div>"
        "</div>"
    )


def dense_panel_html(results: list[dict]) -> str:
    rows = []
    for item in results or []:
        score = item.get("score", 0.0)
        try:
            score_text = f"{float(score):.4f}"
        except (TypeError, ValueError):
            score_text = str(score)
        meta = " · ".join(str(item.get(k, "—")) for k in
                          ("source", "language", "authority_level"))
        preview = html.escape(str(item.get("preview", "")))
        extra = (f"<details class='flow-preview'><summary>Xem chunk</summary>"
                 f"<div>{preview}</div></details>" if preview else "")
        rows.append(_flow_row(
            item.get("rank", 0), str(item.get("title", "—")),
            str(item.get("chunk_id", "—")),
            f"<span class='flow-score-chip'>cosine similarity = "
            f"<b>{score_text}</b></span>",
            html.escape(meta), extra))
    return ("<div class='flow-panel'><div class='flow-panel-head'>DENSE "
            "— tương đồng ngữ nghĩa (cosine similarity)</div>"
            + "".join(rows) + "</div>" if rows else
            "<div class='flow-empty'>Chưa có kết quả Dense.</div>")


def bm25_panel_html(results: list[dict]) -> str:
    rows = []
    for item in results or []:
        score = item.get("score", 0.0)
        try:
            score_text = f"{float(score):.2f}"
        except (TypeError, ValueError):
            score_text = str(score)
        terms = ", ".join(str(t) for t in item.get("matched_terms", []) or [])
        extra = (f"<div class='flow-terms'>matched terms: "
                 f"{html.escape(terms)}</div>" if terms else "")
        rows.append(_flow_row(
            item.get("rank", 0), str(item.get("title", "—")),
            str(item.get("chunk_id", "—")),
            f"<span class='flow-score-chip'>BM25 score = <b>{score_text}</b>"
            f"</span>",
            html.escape(str(item.get("source", "—"))), extra))
    return ("<div class='flow-panel'><div class='flow-panel-head'>BM25 "
            "— liên quan từ vựng (lexical relevance)</div>"
            + "".join(rows) + "</div>" if rows else
            "<div class='flow-empty'>Chưa có kết quả BM25.</div>")


def rrf_panel_html(results: list[dict]) -> str:
    rows = []
    for item in results or []:
        score = item.get("rrf_score", 0.0)
        try:
            score_text = f"{float(score):.6f}"
        except (TypeError, ValueError):
            score_text = str(score)
        dense_rank = item.get("dense_rank", "—")
        bm25_rank = item.get("bm25_rank", "—")
        rows.append(_flow_row(
            item.get("final_rank", 0), str(item.get("title", "—")),
            str(item.get("chunk_id", "—")),
            f"<span class='flow-score-chip'>RRF score = <b>{score_text}</b>"
            f"</span>",
            html.escape(f"dense rank: {dense_rank} · "
                        f"bm25 rank: {bm25_rank} · "
                        f"{item.get('source', '—')}")))
    return ("<div class='flow-panel'><div class='flow-panel-head'>RRF FUSION "
            "— gộp thứ hạng (rank), không so sánh điểm thô</div>"
            + "".join(rows) + "</div>" if rows else
            "<div class='flow-empty'>Chưa có kết quả RRF.</div>")


def selected_context_html(items: list[dict]) -> str:
    rows = []
    for item in items or []:
        rows.append(
            "<div class='flow-row'>"
            f"<div class='flow-rank'>#{int(item.get('final_rank', 0))}</div>"
            "<div class='flow-body'>"
            f"<div class='src-title'>{html.escape(str(item.get('title', '—')))}</div>"
            f"<div class='flow-chunk-id'>{html.escape(str(item.get('chunk_id', '—')))}</div>"
            "</div>"
            "</div>")
    return ("<div class='flow-panel'><div class='flow-panel-head'>SELECTED "
            "CONTEXT — chunks sẽ gửi cho LLM (Top-K)</div>"
            + "".join(rows) + "</div>" if rows else
            "<div class='flow-empty'>Chưa có context được chọn.</div>")
