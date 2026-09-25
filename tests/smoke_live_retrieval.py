"""Opt-in live demo smoke: .venv/Scripts/python.exe tests/smoke_live_retrieval.py."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest
from src.ui_backend import get_retrieval_trace

queries = [
    "Himass và TanVuu bị xử lý vì hành vi gì?",
    "Stream sniping có vi phạm luật PUBG không?",
]
for query in queries:
    trace = get_retrieval_trace(query)
    assert trace["query"] == query
    assert not trace["is_mock"] and trace["backend_connected"]
    for layer in ("dense", "bm25", "rrf"):
        assert trace[layer]["results"]
    assert all(item["content"] for item in trace["selected_context"])
    for item in trace["rrf"]["results"]:
        expected = sum(1 / (60 + item[key]) for key in ("dense_rank", "bm25_rank")
                       if item[key] is not None)
        assert abs(item["rrf_score"] - expected) < 1e-12
    print("PASS live trace:", ascii(query),
          {key: len(trace[key]["results"]) for key in ("dense", "bm25", "rrf")})

app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=90).run()
assert not app.exception and not app.error
assert len(app.tabs) == 2
trace = app.session_state.flow_trace
assert not trace["is_mock"] and trace["selected_context"]
body = " ".join(item.value for item in app.markdown)
assert "LIVE" in body and "RRF FUSION" in body
assert "Demo retrieval trace" not in body
app.text_input(key="flow_query_input").set_value(queries[0])
app.button(key="flow_run").click().run()
assert not app.exception and not app.error
assert app.session_state.flow_trace["query"] == queries[0]
print("PASS Streamlit: both tabs, live panels, query submit, selected context")

with patch("src.ui_backend.get_retrieval_trace", side_effect=RuntimeError("offline")):
    failed = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=90).run()
    assert not failed.exception and failed.error
print("PASS retrieval failure: UI error without crash")
