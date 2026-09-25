# -*- coding: utf-8 -*-
"""Focused Task 4 tests: chunking, external-embedding boundary (mocked),
Chroma indexing + idempotency. Never calls a real embedding API."""

import pytest

import src.task4_chunking_indexing as task4
from src.contracts import validate_document


def _doc(doc_id="doc-a", content="Hello world. " * 40,
         doc_type="news", **extra):
    meta = {"source": f"{doc_id}.md", "title": f"Title {doc_id}",
            "doc_type": doc_type, "url": "https://example.com/x"}
    meta.update(extra)
    return {"id": doc_id, "content": content, "metadata": meta}


def test_chunk_ids_stable_and_sequential():
    docs = [_doc("doc-a"), _doc("doc-b", content="Short text here.")]
    first = task4.chunk_documents(docs)
    second = task4.chunk_documents(docs)
    assert [c["id"] for c in first] == [c["id"] for c in second]
    assert len({c["id"] for c in first}) == len(first)
    for doc in docs:
        indices = [c["metadata"]["chunk_index"]
                   for c in first if c["id"].startswith(doc["id"])]
        assert indices == list(range(len(indices)))
    for chunk in first:
        validate_document(chunk, require_chunk=True)
        assert chunk["content"].strip()
        assert len(chunk["content"]) <= int(task4.CHUNK_SIZE * 1.1)


def test_chroma_metadata_serialization():
    clean = task4._chroma_metadata({
        "source": "a.md", "title": "T", "doc_type": "news",
        "url": None, "topic": ["cheating", "reporting"], "chunk_index": 0,
    })
    assert clean["topic"] == "cheating|reporting"
    assert "url" not in clean
    assert clean["chunk_index"] == 0


def test_embed_texts_uses_client_and_validates(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(task4, "_client", None)
    calls = []

    def fake_embed(self, texts):
        calls.append(list(texts))
        return [[float(i), 0.5] for i, _ in enumerate(texts)]

    monkeypatch.setattr(task4.EmbeddingClient, "embed", fake_embed)
    out = task4.embed_texts(["alpha", "beta"])
    assert out == [[0.0, 0.5], [1.0, 0.5]]
    assert calls == [["alpha", "beta"]]
    assert task4.embed_texts([]) == []
    with pytest.raises(ValueError):
        task4.embed_texts(["ok", "   "])


def test_post_json_retries_then_succeeds(monkeypatch):
    import requests

    attempts = []

    class Resp:
        status_code = 200
        headers = {}

        def json(self):
            return {"data": [{"index": 0, "embedding": [0.1]}]}

    def fake_post(*args, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            fail = Resp()
            fail.status_code = 503
            fail.headers = {}
            fail.text = "busy"
            return fail
        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    out = task4._post_json("https://x.test/e", {}, {}, 5)
    assert out["data"][0]["embedding"] == [0.1]
    assert len(attempts) == 3


def test_post_json_client_error_raises_without_retry(monkeypatch):
    import requests

    class Resp:
        status_code = 401
        headers = {}
        text = "bad key"

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    with pytest.raises(task4.EmbeddingError, match="401"):
        task4._post_json("https://x.test/e", {}, {}, 5)


def test_missing_api_key_raises_clear_error(monkeypatch):
    for var in ("OPENAI_API_KEY", "JINA_API_KEY", "GEMINI_API_KEY",
                "EMBEDDING_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(task4.EmbeddingError, match="OPENAI_API_KEY"):
        task4.EmbeddingClient(provider="openai", model="m")


def test_index_idempotent_with_mocked_embeddings(monkeypatch, tmp_path):
    monkeypatch.setattr(task4, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(task4, "EMBEDDING_PROVIDER", "openai")
    calls = []

    def fake_embed(self, texts):
        calls.append(len(texts))
        dim = 8
        return [[float(len(t) % 7)] * dim for t in texts]

    monkeypatch.setattr(task4.EmbeddingClient, "embed", fake_embed)
    chunks = task4.chunk_documents([_doc("doc-a"), _doc("doc-b")])

    task4.index_to_vectorstore([dict(c) for c in chunks])
    col = task4.get_collection()
    assert col.count() == len(chunks)
    first_calls = list(calls)

    task4.index_to_vectorstore([dict(c) for c in chunks])
    assert task4.get_collection().count() == len(chunks)
    assert calls == first_calls  # second run: 0 new embedding calls

    got = col.get(ids=[chunks[0]["id"]],
                  include=["documents", "metadatas"])
    assert got["documents"][0] == chunks[0]["content"]
    assert got["metadatas"][0]["embedding_provider"] == "openai"


def test_load_real_standardized_corpus():
    docs = task4.load_documents()
    assert len(docs) == 9
    by_id = {d["id"]: d for d in docs}
    assert by_id["justice_for_himass_tanvuu_vi"]["metadata"]["authority_level"] \
        == "secondary"
    assert by_id["justice_for_himass_tanvuu_vi"]["metadata"]["source_type"] \
        == "independent"
    en_ids = {d["id"] for d in docs
              if d["metadata"]["language"] == "en"}
    assert en_ids == {
        "pubg_report_cheating_bug_abuse_vi",
        "pubg_account_cheating_responsibility_vi",
        "pubg_report_cheat_seller_vi",
        "pubg_ban_penalty_information_vi",
    }
    for doc in docs:
        validate_document(doc)
