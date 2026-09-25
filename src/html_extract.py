# -*- coding: utf-8 -*-
"""Shared stdlib-only HTML article extraction for Task 1 and Task 2.

No third-party parsers: uses :mod:`html.parser` so collection modules add
no new dependencies. Extracts ``{"title": ..., "blocks": [...]}`` where
blocks are Markdown-ish lines (``# `` headings, ``- `` list items,
plain paragraphs, ``|`` table rows) with navigation/header/footer chrome
removed. Original wording is preserved (no summarising/translation).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

_SKIP_TAGS = {"script", "style", "noscript", "nav", "header", "footer",
              "aside", "form", "button", "select", "iframe", "svg"}
_HEADING_TAGS = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### "}
_BLOCK_TAGS = {"p", "li", "blockquote", "pre", "tr", "h1", "h2", "h3",
               "h4", "h5", "h6", "div", "section", "article", "br", "hr"}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


class _ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._current: list[str] = []
        self.blocks: list[str] = []
        self._list_depth = 0
        self._in_li = 0
        self._prefix = ""

    # -- tag handling ----------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag in ("ul", "ol"):
            self._list_depth += 1
            self._flush()
        elif tag == "li":
            self._flush()
            self._in_li += 1
            if self._list_depth:
                self._prefix = "- "
        elif tag in _HEADING_TAGS:
            self._flush()
            self._prefix = _HEADING_TAGS[tag]
        elif tag in _BLOCK_TAGS:
            self._flush()
        elif tag == "td" or tag == "th":
            self._current.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag in ("ul", "ol"):
            self._flush()
            self._list_depth = max(0, self._list_depth - 1)
        elif tag == "li":
            self._flush()
            self._in_li = max(0, self._in_li - 1)
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
            return
        self._current.append(data)

    # -- block assembly --------------------------------------------------
    def _flush(self) -> None:
        text = _clean("".join(self._current))
        self._current = []
        prefix, self._prefix = self._prefix, ""
        if text:
            self.blocks.append(f"{prefix}{text}" if prefix else text)

    def close(self) -> None:  # noqa: D102
        self._flush()
        super().close()


def _text_len(fragment: str) -> int:
    return len(re.sub(r"<[^>]+>", " ", fragment))


def _scoped_html(html: str) -> str:
    """Prefer the longest <article> (body), else <main>, else de-chromed page.

    News/policy pages often contain several small <article> teasers, so the
    first match is not necessarily the content.
    """
    articles = re.findall(r"<article[\s>].*?</article\s*>", html,
                          flags=re.S | re.I)
    if articles:
        return max(articles, key=_text_len)
    lowered = html.lower()
    start = lowered.find("<main")
    end = lowered.find("</main>")
    if 0 <= start < end:
        return html[start:end + len("main") + 3]
    # Fallback: drop whole chrome subtrees with a tolerant regex.
    cleaned = re.sub(
        r"<(header|nav|footer|aside|script|style|noscript|form)[\s>].*?"
        r"</\1\s*>", " ", html, flags=re.S | re.I)
    return cleaned


def extract_article(html: str) -> dict:
    """Extract ``{"title": str, "blocks": [str]}`` from raw HTML."""
    head_title = re.search(r"<title[^>]*>(.*?)</title\s*>", html,
                           flags=re.S | re.I)
    parser = _ArticleParser()
    parser.feed(_scoped_html(html))
    parser.close()
    title = _clean(head_title.group(1)) if head_title else ""
    if not title:
        title = _clean("".join(parser._title_parts))
    if not title and parser.blocks:
        title = re.sub(r"^(#{1,4}\s+|-\s+)", "", parser.blocks[0])
    blocks: list[str] = []
    for block in parser.blocks:
        if len(block) < 2:
            continue
        if block not in blocks or len(block) > 120:
            blocks.append(block)
    # De-duplicate short repeated chrome remnants, keep order.
    seen: set[str] = set()
    unique: list[str] = []
    for block in blocks:
        key = block if len(block) <= 120 else None
        if key is None or key not in seen:
            unique.append(block)
            if key is not None:
                seen.add(key)
    return {"title": title, "blocks": unique}
