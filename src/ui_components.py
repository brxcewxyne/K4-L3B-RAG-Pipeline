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


def assistant_message_html(
    answer_markdown: str,
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
        source_card_html(i + 1, s) for i, s in enumerate(sources or [])
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
