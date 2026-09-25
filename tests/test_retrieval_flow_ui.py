# -*- coding: utf-8 -*-
"""Retrieval-flow tab tests: trace contract (demo), render helpers, app tabs.

No real retrieval runs here — the tab must render clearly-labelled demo
data only. No network calls.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src import ui_components as comp
from src.ui_backend import TRACE_ENTRYPOINT, get_retrieval_trace

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_trace_contract_layers_are_separate():
    trace = get_retrieval_trace("stream sniping là gì?", top_k=5)
    assert trace["is_mock"] is True
    assert trace["backend_connected"] is False
    assert trace["query"] == "stream sniping là gì?"
    assert trace["dense"]["metric"] == "cosine similarity"
    assert trace["bm25"]["metric"] == "BM25 score"
    assert "RRF" in trace["rrf"]["metric"]
    assert trace["dense"]["results"] and trace["bm25"]["results"]
    assert trace["rrf"]["results"] and trace["selected_context"]
    rrf_first = trace["rrf"]["results"][0]
    assert {"chunk_id", "title", "rrf_score", "dense_rank",
            "bm25_rank", "final_rank"} <= set(rrf_first)


def test_panels_label_scores_explicitly():
    trace = get_retrieval_trace("x")
    dense_html = comp.dense_panel_html(trace["dense"]["results"])
    bm25_html = comp.bm25_panel_html(trace["bm25"]["results"])
    rrf_html = comp.rrf_panel_html(trace["rrf"]["results"])
    assert "cosine similarity" in dense_html
    assert "BM25 score" in bm25_html
    assert "RRF score" in rrf_html
    # raw BM25 and cosine numbers must not share one unlabeled "score" field
    assert "cosine similarity" not in bm25_html


def test_render_helpers_escape_user_content():
    evil = "<script>alert(1)</script>"
    html_out = comp.flow_query_html(evil) + comp.dense_panel_html(
        [{"rank": 1, "title": evil, "chunk_id": "c::chunk-0",
          "score": 0.5, "source": "s", "language": "vi",
          "authority_level": "primary", "preview": evil}])
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_app_renders_both_tabs_without_errors():
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["💬 Trò chuyện", "🔀 Luồng truy xuất"]
    body = " ".join(m.value for m in at.markdown)
    assert "cosine similarity" in body
    assert "BM25 score" in body
    assert "RRF" in body
    assert "chưa kết nối" in body
    assert TRACE_ENTRYPOINT  # integration point documented


def test_chat_tab_roundtrip_still_works():
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    at.text_area[0].set_value("Khi nào bị ban vĩnh viễn?")
    send = next(b for b in at.button if "Gửi" in str(b.label))
    send.click().run()
    assert not at.exception
    assert len(at.session_state.messages) == 2
    assert at.session_state.messages[1]["is_mock"] is True
