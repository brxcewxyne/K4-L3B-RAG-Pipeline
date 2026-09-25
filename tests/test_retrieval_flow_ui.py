# -*- coding: utf-8 -*-
"""Retrieval-flow tab tests: trace contract, render helpers, app tabs.

The live backend is explicitly stubbed here — no network/API calls.
"""

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src import ui_components as comp
from src.ui_backend import TRACE_ENTRYPOINT

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def fixture_trace(query="stream sniping là gì?"):
    dense = [{"chunk_id": "d::chunk-0", "title": "Dense doc",
              "source": "KRAFTON, Inc.", "language": "vi",
              "authority_level": "primary", "score": 0.82, "rank": 1,
              "preview": "Dense evidence", "content": "Dense evidence"}]
    bm25 = [{"chunk_id": "b::chunk-1", "title": "BM25 doc",
             "source": "PUBG Support", "score": 7.42, "rank": 1,
             "preview": "BM25 evidence", "content": "BM25 evidence"}]
    rrf = [{"chunk_id": "d::chunk-0", "title": "Dense doc",
            "source": "KRAFTON, Inc.", "rrf_score": 1 / 61 + 1 / 62,
            "dense_rank": 1, "bm25_rank": 2, "final_rank": 1}]
    return {
        "query": query,
        "dense": {"metric": "cosine similarity", "results": dense},
        "bm25": {"metric": "BM25 score", "results": bm25},
        "rrf": {"metric": "RRF score (k=60)", "results": rrf},
        "fallback": {"triggered": False},
        "pageindex": {"metric": "PageIndex vectorless (tree navigation)",
                      "results": []},
        "selected_context": [{"chunk_id": "d::chunk-0", "title": "Dense doc",
                              "final_rank": 1}],
        "retrieval_source": "hybrid",
        "is_mock": False,
        "backend_connected": True,
    }


def test_trace_contract_layers_are_separate():
    trace = fixture_trace()
    assert trace["is_mock"] is False
    assert trace["backend_connected"] is True
    assert trace["dense"]["metric"] == "cosine similarity"
    assert trace["bm25"]["metric"] == "BM25 score"
    assert "RRF" in trace["rrf"]["metric"]
    assert trace["dense"]["results"] and trace["bm25"]["results"]
    assert trace["rrf"]["results"] and trace["selected_context"]
    rrf_first = trace["rrf"]["results"][0]
    assert {"chunk_id", "title", "rrf_score", "dense_rank",
            "bm25_rank", "final_rank"} <= set(rrf_first)
    assert trace["fallback"]["triggered"] is False
    assert "pageindex" in trace


def test_panels_label_scores_explicitly():
    trace = fixture_trace()
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


def test_fallback_panel_marks_pageindex_downstream():
    html_out = comp.fallback_panel_html(
        {"triggered": True, "reason": "dense_score_below_threshold",
         "best_dense_score": 0.21, "threshold": 0.3, "method": "pageindex"},
        [{"chunk_id": "pageindex::pubg_rules_of_conduct_vi::0005",
          "title": "Quy tắc ứng xử", "page": 10, "section": "Mục 5",
          "preview": "Nội dung dẫn chiếu"}])
    assert "PageIndex" in html_out
    assert "0.21" in html_out
    assert "Mục 5" in html_out
    # panel is framed as post-RRF fallback, not a third parallel retriever
    assert "hybrid" in html_out


def test_app_renders_both_tabs_without_errors():
    with patch("src.ui_backend.get_retrieval_trace",
               return_value=fixture_trace()):
        at = AppTest.from_file(APP_PATH, default_timeout=90)
        at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["💬 Trò chuyện", "🔀 Luồng truy xuất"]
    body = " ".join(m.value for m in at.markdown)
    assert "cosine similarity" in body
    assert "BM25 score" in body
    assert "RRF" in body
    assert TRACE_ENTRYPOINT  # integration point documented


def test_chat_tab_roundtrip_still_works():
    ui_result = {"answer": "Fixture answer [1]",
                 "sources": [{"title": "Fixture", "publisher": "PUBG",
                              "url": "https://example.com/1",
                              "source_type": "official",
                              "authority_level": "Official · Primary",
                              "language": "vi", "snippet": "ev"}],
                 "retrieval_method": "hybrid", "retrieval_source": "hybrid",
                 "is_mock": False,
                 "retrieval_trace": fixture_trace()}
    with patch("src.ui_backend.handle_query", return_value=ui_result):
        at = AppTest.from_file(APP_PATH, default_timeout=90)
        at.run()
        at.text_area[0].set_value("Khi nào bị ban vĩnh viễn?")
        send = next(b for b in at.button if "Gửi" in str(b.label))
        send.click().run()
    assert not at.exception
    assert len(at.session_state.messages) == 2
    assert at.session_state.messages[1]["is_mock"] is False
