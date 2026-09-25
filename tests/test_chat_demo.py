"""Focused offline checks; API and retrieval are explicitly stubbed."""
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from src import task10_generation as generation
from src.ui_components import assistant_message_html


def test_chat_shares_trace_and_keeps_citation_numbers():
    chunks = [{"id": f"chunk-{i}", "content": "Fixture evidence",
               "score": 1 / (60 + i), "retrieval_method": "hybrid",
               "metadata": {"title": f"Fixture {i}", "source": "fixture",
                            "url": f"https://example.com/{i}",
                            "source_type": "official" if i == 1 else "independent"}}
              for i in (1, 2)]
    trace = {"query": "fixture query", "dense": {"results": []},
             "bm25": {"results": []}, "rrf": {"results": []},
             "selected_context": [], "is_mock": False, "backend_connected": True}
    with patch.object(generation, "retrieve_with_trace", return_value={
            "chunks": chunks, "retrieval_trace": trace}), \
         patch.object(generation, "call_llm", return_value="Fixture [2]") as llm:
        result = generation.generate_with_citation("fixture query")
        assert result["sources"][1]["citation_id"] == 2
        assert '[2]' in llm.call_args.args[1]
        from src.ui_backend import handle_query
        ui_result = handle_query("fixture query")
    html = assistant_message_html("Fixture [2]", sources=[ui_result["sources"][1]], is_mock=False)
    assert '[2]' in html and 'https://example.com/2' in html
    app_path = str(Path(__file__).resolve().parents[1] / "app.py")
    with patch("src.ui_backend.get_retrieval_trace", return_value=trace):
        app = AppTest.from_file(app_path, default_timeout=30).run()
    with patch("src.ui_backend.handle_query", return_value=ui_result), \
         patch("src.ui_backend.get_retrieval_trace") as flow:
        app.text_area[0].set_value("fixture query")
        app.button(key="FormSubmitter:composer-➤ Gửi").click().run()
        assert not app.exception and not app.error
        assert flow.call_count == 0
        assert app.session_state.flow_trace == trace
        assert app.session_state.messages[-1]["content"] == "Fixture [2]"
    with patch("src.ui_backend.handle_query", side_effect=generation.GenerationError("Provider unavailable")):
        app.text_area[0].set_value("fixture query")
        app.button(key="FormSubmitter:composer-➤ Gửi").click().run()
        assert not app.exception and app.error and not app.session_state.is_loading
    with patch.object(generation, "retrieve_with_trace", return_value={
            "chunks": [], "retrieval_trace": trace}), patch.object(generation, "call_llm") as llm:
        assert generation.generate_with_citation("fixture query")["answer"] == generation.INSUFFICIENT_EVIDENCE
        llm.assert_not_called()
    with patch.object(generation, "retrieve_with_trace", return_value={
            "chunks": chunks, "retrieval_trace": trace}), \
         patch.object(generation, "call_llm", return_value="Invalid citation [99]"):
        assert generation.generate_with_citation("fixture query")["sources"] == []
