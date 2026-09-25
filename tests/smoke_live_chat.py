"""Opt-in: run only the four requested live chatbot demo queries."""
import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest
from src import ui_backend
from src.task10_generation import INSUFFICIENT_EVIDENCE, GenerationError
from src.task4_chunking_indexing import get_collection
from src.contracts import validate_generation_result
from src.ui_components import assistant_message_html

QUERIES = [
    "Stream sniping có vi phạm luật PUBG không?",
    "Himass và TanVuu bị xử lý vì hành vi gì?",
    "Nếu người khác cheat trên tài khoản của tôi thì tôi có bị ban không?",
    "Súng nào mạnh nhất trong PUBG hiện tại?",
]
app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
assert not app.exception and not app.error
reports = []
failures = []
sys.stdout.reconfigure(encoding="utf-8")
for index, query in enumerate(QUERIES):
    with patch("src.ui_backend.handle_query", wraps=ui_backend.handle_query) as chat, \
         patch("src.ui_backend.get_retrieval_trace", wraps=ui_backend.get_retrieval_trace) as flow:
        app.text_area[0].set_value(query)
        app.button(key="FormSubmitter:composer-➤ Gửi").click().run()
        assert not app.exception and not app.error
        assert chat.call_count == 1 and flow.call_count == 0
    message = app.session_state.messages[-1]
    trace = message["retrieval_trace"]
    assert not message["is_mock"] and trace == app.session_state.flow_trace
    assert trace["query"] == query
    assert all(trace[layer]["results"] for layer in ("dense", "bm25", "rrf"))
    assert len(trace["selected_context"]) <= 5
    answer, sources = message["content"], message["sources"]
    citations = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    if index < 3:
        if answer == INSUFFICIENT_EVIDENCE or not citations:
            failures.append(f"Query {index + 1}: no supported answer")
        assert citations <= {s["citation_id"] for s in sources}
        if not any(s.get("source_type") == "official" for s in sources):
            failures.append(f"Query {index + 1}: no official source")
    else:
        if answer != INSUFFICIENT_EVIDENCE or sources:
            failures.append("Query 4: refusal failed")
    for source in sources:
        stored = get_collection().get(ids=[source["id"]], include=["metadatas", "documents"])
        assert stored["metadatas"][0]["url"] == source["url"]
        assert stored["documents"][0] == source["content"]
    validate_generation_result({"answer": answer, "sources": sources,
                                "retrieval_source": "hybrid" if sources else "none"})
    if sources:
        rendered = assistant_message_html(answer, sources=[sources[-1]], is_mock=False)
        assert f'[{sources[-1]["citation_id"]}]' in rendered
    reports.append({"query": query, "answer": answer,
                    "sources": [{k: s.get(k) for k in
                                 ("citation_id", "id", "url", "source_type", "title")}
                                for s in sources],
                    "context": trace["selected_context"]})
    print(json.dumps(reports[-1], ensure_ascii=False), flush=True)
body = " ".join(m.value for m in app.markdown)
assert "UI PREVIEW" not in body and "Demo retrieval trace" not in body
with patch("src.ui_backend.handle_query", side_effect=GenerationError("Provider unavailable")):
    app.text_area[0].set_value(QUERIES[0])
    app.button(key="FormSubmitter:composer-➤ Gửi").click().run()
    assert not app.exception and app.error and not app.session_state.is_loading
assert not failures, failures
print("PASS: four live queries, actual source URLs, shared trace, stable citation numbers, clean provider error")
