# -*- coding: utf-8 -*-
"""Threshold boundary tests for the Task 9 Dense fallback gate.

All retrieval/external calls are mocked — no network. The gate under test:
``best_dense is None or best_dense < score_threshold`` → PageIndex;
otherwise hybrid. RRF scores never influence the decision.
"""

from pathlib import Path

import pytest

import src.task9_retrieval_pipeline as pipeline

CAL = 0.5728  # calibrated value under test (mirrors artifact + code default)


def _dense(score):
    return [{"id": "chunk-0", "content": "dense evidence", "score": score,
             "metadata": {"source": "s.md", "title": "T", "doc_type": "legal",
                          "url": None, "chunk_index": 0},
             "retrieval_method": "dense"}]


def _hybrid(score):
    return [{"id": "chunk-0", "content": "dense evidence", "score": score,
             "metadata": {"source": "s.md", "title": "T", "doc_type": "legal",
                          "url": None, "chunk_index": 0},
             "retrieval_method": "hybrid"}]


class _SeededCollection:
    def get(self, include=None):
        return {"ids": ["chunk-0"], "documents": ["seed"], "metadatas": [{}]}


def _run(monkeypatch, dense_score=None, rrf_score=0.9):
    calls = {"pageindex": 0}
    monkeypatch.setattr(pipeline, "get_collection", lambda: _SeededCollection())
    dense = [] if dense_score is None else _dense(dense_score)
    monkeypatch.setattr(pipeline, "semantic_search",
                        lambda q, top_k: dense)
    monkeypatch.setattr(pipeline, "lexical_search", lambda q, top_k: [])
    monkeypatch.setattr(pipeline, "rerank_rrf",
                        lambda lists, top_k: _hybrid(rrf_score))

    def fake_pi(query, top_k):
        calls["pageindex"] += 1
        return [{"id": "pageindex::doc::0001", "content": "pi evidence",
                 "score": 1.0,
                 "metadata": {"source": "KRAFTON, Inc.", "title": "T",
                              "doc_type": "legal",
                              "url": "https://example.com/x",
                              "chunk_index": 0},
                 "retrieval_method": "pageindex"}]

    monkeypatch.setattr(pipeline, "pageindex_search", fake_pi)
    out = pipeline.retrieve_with_trace("q", top_k=2, score_threshold=CAL)
    return out, calls


def test_above_threshold_no_fallback(monkeypatch):
    out, calls = _run(monkeypatch, dense_score=CAL + 0.05)
    assert calls["pageindex"] == 0
    assert out["retrieval_source"] == "hybrid"
    assert out["retrieval_trace"]["fallback"] == {"triggered": False}


def test_below_threshold_triggers_pageindex(monkeypatch):
    out, calls = _run(monkeypatch, dense_score=CAL - 0.05)
    assert calls["pageindex"] == 1
    assert out["retrieval_source"] == "pageindex"
    assert out["retrieval_trace"]["fallback"]["triggered"] is True


def test_no_dense_results_triggers_fallback(monkeypatch):
    out, calls = _run(monkeypatch, dense_score=None)
    assert calls["pageindex"] == 1
    assert out["retrieval_source"] == "pageindex"
    assert (out["retrieval_trace"]["fallback"]["reason"]
            == "no_dense_results")


def test_rrf_score_does_not_affect_decision(monkeypatch):
    # High RRF + weak dense -> still falls back.
    out, calls = _run(monkeypatch, dense_score=0.1, rrf_score=0.99)
    assert calls["pageindex"] == 1
    assert out["retrieval_source"] == "pageindex"
    # Low RRF + strong dense -> stays hybrid.
    out, calls = _run(monkeypatch, dense_score=0.9, rrf_score=0.001)
    assert calls["pageindex"] == 0
    assert out["retrieval_source"] == "hybrid"


def test_exact_boundary_keeps_hybrid(monkeypatch):
    # Contract: gate is `best_dense < threshold`; equality keeps hybrid.
    out, calls = _run(monkeypatch, dense_score=CAL)
    assert calls["pageindex"] == 0
    assert out["retrieval_source"] == "hybrid"


def test_production_default_matches_calibration():
    import json
    artifact = json.loads(
        (Path(pipeline.__file__).resolve().parents[1]
         / "group_project" / "evaluation" / "threshold_calibration.json")
        .read_text(encoding="utf-8"))
    assert artifact["selected_threshold"] == pytest.approx(CAL)
    assert pipeline.SCORE_THRESHOLD == pytest.approx(CAL)
