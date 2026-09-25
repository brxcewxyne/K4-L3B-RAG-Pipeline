# -*- coding: utf-8 -*-
"""Task 8 (PageIndex fallback) tests — PageIndex API luôn được mock.

Không gọi network/API thật. Bao phủ: config, discovery, upload lifecycle,
registry, normalization, top_k, lỗi API, Task 9 fallback wiring.
"""

import pytest

import src.task8_pageindex_vectorless as task8
import src.task9_retrieval_pipeline as pipeline


def _node(node_id="0005", title="Mục 5", page=10, text="Nội dung dẫn chiếu"):
    return {"node_id": node_id, "title": title,
            "relevant_contents": [{"page_index": page,
                                   "relevant_content": text}]}


class FakeClient:
    """Fake PageIndexClient: ghi lại calls, không chạm network."""

    def __init__(self, nodes=None, ready=True, fail_on=None):
        self.nodes = nodes if nodes is not None else [_node()]
        self.ready = ready
        self.fail_on = fail_on or set()
        self.calls = {"submit_document": 0, "submit_query": 0,
                      "get_retrieval": 0}
        self.n_doc = 0

    def submit_document(self, file_path):
        self.calls["submit_document"] += 1
        if "submit_document" in self.fail_on:
            raise FakeAPIError("boom")
        self.n_doc += 1
        return {"doc_id": f"pi-doc-{self.n_doc}"}

    def is_retrieval_ready(self, doc_id):
        return self.ready

    def submit_query(self, doc_id, query, thinking=False):
        self.calls["submit_query"] += 1
        if "submit_query" in self.fail_on:
            raise FakeAPIError("boom")
        return {"retrieval_id": "r-1"}

    def get_retrieval(self, retrieval_id):
        self.calls["get_retrieval"] += 1
        return {"retrieval_id": retrieval_id, "status": "completed",
                "retrieved_nodes": self.nodes}


class FakeAPIError(Exception):
    pass


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Task 8 trỏ registry vào tmp; FakeAPIError thay PageIndexAPIError."""
    import pageindex
    monkeypatch.setattr(task8, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(task8, "_client", None)
    monkeypatch.setenv("PAGEINDEX_API_KEY", "test-key")
    monkeypatch.setattr(pageindex, "PageIndexAPIError", FakeAPIError,
                        raising=False)
    return tmp_path


def _use_fake(monkeypatch, fake):
    monkeypatch.setattr(task8, "_client", None)
    monkeypatch.setattr(task8, "get_client", lambda: fake)
    return fake


def test_missing_api_key_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(task8, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(task8, "_client", None)
    monkeypatch.delenv("PAGEINDEX_API_KEY", raising=False)
    with pytest.raises(task8.PageIndexError, match="PAGEINDEX_API_KEY"):
        task8.get_client()


def test_discover_finds_three_policy_pdfs():
    entries = task8.discover_documents()
    assert [e["key"] for e in entries] == [
        "pubg_rules_of_conduct_vi", "pubg_terms_of_service_vi",
        "pubg_privacy_policy_vi"]
    assert all(e["content_hash"] for e in entries)


def test_first_upload_writes_registry(monkeypatch, isolated):
    fake = _use_fake(monkeypatch, FakeClient())
    stored = task8.upload_documents(timeout_s=1, poll_s=0)
    assert set(stored) == {"pubg_rules_of_conduct_vi",
                           "pubg_terms_of_service_vi",
                           "pubg_privacy_policy_vi"}
    assert fake.calls["submit_document"] == 3
    for key, row in stored.items():
        assert row["pageindex_document_id"].startswith("pi-doc-")
        assert len(row["content_hash"]) == 40
        assert row["indexed_at"] and row["status"] == "ready"
    import json
    raw = json.loads((isolated / "registry.json").read_text())
    for row in raw["documents"].values():
        assert set(row) >= {"local_path", "source_url",
                            "pageindex_document_id", "content_hash",
                            "indexed_at", "status"}


def test_unchanged_documents_reuse_registry(monkeypatch, isolated):
    fake = _use_fake(monkeypatch, FakeClient())
    first = task8.upload_documents(timeout_s=1, poll_s=0)
    fake.calls["submit_document"] = 0
    second = task8.upload_documents(timeout_s=1, poll_s=0)
    assert fake.calls["submit_document"] == 0
    assert second == first


def test_changed_document_triggers_reupload(monkeypatch, tmp_path):
    reg = tmp_path / "registry.json"
    monkeypatch.setattr(task8, "REGISTRY_PATH", reg)
    monkeypatch.setattr(task8, "_client", None)
    monkeypatch.setenv("PAGEINDEX_API_KEY", "test-key")
    pdf = tmp_path / "policy.pdf"
    pdf.write_bytes(b"v1")
    monkeypatch.setattr(task8, "PAGEINDEX_DOCUMENTS", (
        {**{k: v for k, v in task8.PAGEINDEX_DOCUMENTS[0].items()
            if k != "local_path"},
         "key": "doc", "local_path": "policy.pdf"},))
    # local_path tương đối repo root -> monkeypatch REPO_ROOT thay cho đơn giản:
    monkeypatch.setattr(task8, "REPO_ROOT", tmp_path)
    fake = _use_fake(monkeypatch, FakeClient())
    task8.upload_documents(timeout_s=1, poll_s=0)
    assert fake.calls["submit_document"] == 1
    pdf.write_bytes(b"v2-changed")
    task8.upload_documents(timeout_s=1, poll_s=0)
    assert fake.calls["submit_document"] == 2


def test_partial_upload_failure_lists_document(monkeypatch, isolated):
    fake = _use_fake(monkeypatch, FakeClient(fail_on={"submit_document"}))
    with pytest.raises(task8.PageIndexError, match="pubg_rules_of_conduct_vi"):
        task8.upload_documents(timeout_s=1, poll_s=0)


def test_search_normalizes_page_section_and_top_k(monkeypatch, isolated):
    nodes = [_node("0001", "Mục A", 3, "text A"),
             _node("0002", "Mục B", 7, "text B"),
             {"node_id": "0003", "title": "Empty"}]
    fake = _use_fake(monkeypatch, FakeClient(nodes=nodes))
    task8.upload_documents(timeout_s=1, poll_s=0)
    out = task8.pageindex_search("hành vi bị cấm", top_k=2)
    assert len(out) == 2  # node rỗng bị loại, top_k được tôn trọng
    first = out[0]
    assert first["retrieval_method"] == "pageindex"
    assert first["metadata"]["page"] == 3
    assert first["metadata"]["section"] == "Mục A"
    assert first["metadata"]["doc_type"] == "legal"
    assert first["metadata"]["url"].startswith("https://")
    assert "text A" in first["content"]
    assert out[0]["score"] > out[1]["score"]  # thứ tự giảm dần
    from src.contracts import validate_search_results
    validate_search_results(out)


def test_search_rejects_empty_query(monkeypatch, isolated):
    _use_fake(monkeypatch, FakeClient())
    with pytest.raises(ValueError):
        task8.pageindex_search("   ")


def test_search_without_registry_index(monkeypatch, isolated):
    _use_fake(monkeypatch, FakeClient())
    with pytest.raises(task8.PageIndexError, match="pageindex_vectorless"):
        task8.pageindex_search("query")


def test_search_api_failure_raises_cleanly(monkeypatch, isolated):
    fake = _use_fake(monkeypatch, FakeClient(fail_on={"submit_query"}))
    task8.upload_documents(timeout_s=1, poll_s=0)
    with pytest.raises(task8.PageIndexError, match="search failed"):
        task8.pageindex_search("query")
    assert fake.calls["submit_query"] >= 1


class _SeededCollection:
    """1 fake id để depth>0 và mocked semantic_search được gọi."""

    def get(self, include=None):
        return {"ids": ["chunk-0"], "documents": ["seed"],
                "metadatas": [{}]}


def _dense(score):
    return [{"id": "chunk-0", "content": "dense evidence", "score": score,
             "metadata": {"source": "s.md", "title": "T", "doc_type": "legal",
                          "url": None, "chunk_index": 0},
             "retrieval_method": "dense"}]


def test_task9_calls_pageindex_when_dense_weak(monkeypatch):
    monkeypatch.setattr(pipeline, "get_collection", lambda: _SeededCollection())
    monkeypatch.setattr(pipeline, "semantic_search",
                        lambda query, top_k: _dense(0.2))
    monkeypatch.setattr(pipeline, "lexical_search",
                        lambda query, top_k: [])
    monkeypatch.setattr(pipeline, "rerank_rrf",
                        lambda lists, top_k: [])
    pi_item = {"id": "pageindex::doc::0001", "content": "pi evidence",
               "score": 1.0,
               "metadata": {"source": "KRAFTON, Inc.", "title": "T",
                            "doc_type": "legal", "url": "https://x",
                            "chunk_index": 0},
               "retrieval_method": "pageindex"}
    calls = []
    monkeypatch.setattr(
        pipeline, "pageindex_search",
        lambda query, top_k: calls.append((query, top_k)) or [pi_item])
    out = pipeline.retrieve_with_trace("q", top_k=2, score_threshold=0.5)
    assert calls == [("q", 2)]
    assert out["chunks"] == [pi_item]
    assert out["retrieval_source"] == "pageindex"
    assert out["retrieval_trace"]["fallback"]["triggered"] is True
    assert out["retrieval_trace"]["fallback"]["reason"] \
        == "dense_score_below_threshold"
    assert out["retrieval_trace"]["pageindex"]["results"]


def test_task9_skips_pageindex_when_dense_strong(monkeypatch):
    monkeypatch.setattr(pipeline, "get_collection", lambda: _SeededCollection())
    monkeypatch.setattr(pipeline, "semantic_search",
                        lambda query, top_k: _dense(0.9))
    monkeypatch.setattr(pipeline, "lexical_search",
                        lambda query, top_k: [])
    monkeypatch.setattr(pipeline, "rerank_rrf",
                        lambda lists, top_k: _dense(0.9))
    def _boom(query, top_k):
        raise AssertionError("pageindex must not run on strong hybrid")
    monkeypatch.setattr(pipeline, "pageindex_search", _boom)
    out = pipeline.retrieve_with_trace("q", top_k=2, score_threshold=0.5)
    assert out["retrieval_source"] == "hybrid"
    assert out["retrieval_trace"]["fallback"] == {"triggered": False}
    assert out["retrieval_trace"]["pageindex"]["results"] == []


def test_task9_pageindex_error_falls_through(monkeypatch):
    monkeypatch.setattr(pipeline, "get_collection", lambda: _SeededCollection())
    monkeypatch.setattr(pipeline, "semantic_search",
                        lambda query, top_k: _dense(0.1))
    monkeypatch.setattr(pipeline, "lexical_search",
                        lambda query, top_k: [])
    hybrid = _dense(0.1)
    monkeypatch.setattr(pipeline, "rerank_rrf", lambda lists, top_k: hybrid)
    def _fail(query, top_k):
        raise RuntimeError("provider down")
    monkeypatch.setattr(pipeline, "pageindex_search", _fail)
    out = pipeline.retrieve_with_trace("q", top_k=2, score_threshold=0.5)
    assert out["chunks"] == hybrid
    assert out["retrieval_source"] == "hybrid"
    assert out["retrieval_trace"]["fallback"]["pageindex_error"] is True


def test_normalize_live_provider_shape(monkeypatch, isolated):
    """Regression: observed live shape uses id/nested lists/section_title.

    No numeric page is provided (physical_index is a literal placeholder),
    so page must be omitted, never invented.
    """
    from src.contracts import validate_search_results

    entry = {**task8.PAGEINDEX_DOCUMENTS[0],
             "pageindex_document_id": "pi-live"}
    node = {"id": "0006",
            "title": "1) KHÔNG SỬ DỤNG CHƯƠNG TRÌNH TRÁI PHÉP",
            "metadata": ["pi-live", "pubg_rules_of_conduct_vi.pdf"],
            "relevant_contents": [[
                {"section_title": "1) KHÔNG SỬ DỤNG CHƯƠNG TRÌNH TRÁI PHÉP",
                 "physical_index": "<physical_index_2>",
                 "relevant_content": "KRAFTON cấm sử dụng trái phép"}]]}
    item = task8._normalize_node(entry, node, 1)
    assert item is not None
    assert item["id"] == "pageindex::pubg_rules_of_conduct_vi::0006"
    assert item["retrieval_method"] == "pageindex"
    assert "KRAFTON cấm sử dụng trái phép" in item["content"]
    assert item["metadata"]["section"].startswith("1) KHÔNG SỬ DỤNG")
    assert "page" not in item["metadata"]
    validate_search_results([item])


def test_poll_rejects_unexpected_shape(monkeypatch, isolated):
    class _Client:
        def get_retrieval(self, retrieval_id):
            return ["not", "a", "dict"]

    with pytest.raises(task8.PageIndexError, match="unexpected shape"):
        task8._poll_retrieval(_Client(), "r-1", timeout_s=1, poll_s=0)
