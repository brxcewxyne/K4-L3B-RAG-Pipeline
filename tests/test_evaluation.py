# -*- coding: utf-8 -*-
"""Evaluation compliance tests — fully offline, no network/LLM calls."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "group_project" / "evaluation"


def _load_evaluate_module():
    spec = importlib.util.spec_from_file_location(
        "repo_evaluate", EVAL_DIR / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _golden():
    return json.loads((EVAL_DIR / "golden_dataset.json")
                      .read_text(encoding="utf-8"))


def test_golden_dataset_valid_and_grounded():
    data = _golden()
    assert isinstance(data, list) and len(data) >= 15
    ids = [c["id"] for c in data]
    assert len(set(ids)) == len(ids)
    corpus = "\n".join(p.read_text(encoding="utf-8")
                       for p in (REPO / "data" / "standardized").rglob("*.md"))
    corpus_tokens = set(re.findall(r"\w+", corpus.lower()))
    for case in data:
        assert {"question", "expected_answer", "expected_context"} <= set(case)
        assert all(str(case[k]).strip()
                   for k in ("question", "expected_answer", "expected_context"))
        overlap = (set(re.findall(r"\w+", case["expected_context"].lower()))
                   & corpus_tokens)
        assert len(overlap) >= 5, f"{case['id']} not grounded in corpus"


def test_metric_functions_deterministic():
    # Lexical helpers survive only as optional diagnostics / fallback.
    ev = _load_evaluate_module()
    assert ev.context_recall("apple banana cherry",
                             ["apple pie", "nothing"]) == pytest.approx(1 / 3)
    assert ev.context_precision("apple banana",
                                ["apple banana pie", "nothing here"]) == 0.5
    assert ev.faithfulness("apple pie", ["apple pie today"], False) == 1.0
    assert ev.faithfulness("anything", [], True) == 1.0
    assert ev.answer_relevance("apple pie", "apple cake") == pytest.approx(1 / 2)
    assert ev.context_recall("", ["x"]) == 0.0
    assert ev.context_precision("x", []) == 0.0


def test_config_a_is_dense_only(monkeypatch):
    ev = _load_evaluate_module()
    calls = []
    monkeypatch.setattr(ev, "semantic_search",
                        lambda q, top_k: calls.append((q, top_k)) or ["d"])
    assert ev.retrieve_a("q?") == ["d"]
    assert calls == [("q?", 5)]


def test_config_b_fuses_with_rrf_k60(monkeypatch):
    ev = _load_evaluate_module()
    monkeypatch.setattr(ev, "semantic_search", lambda q, top_k: ["d"])
    monkeypatch.setattr(ev, "lexical_search", lambda q, top_k: ["b"])
    seen = {}
    monkeypatch.setattr(ev, "rerank_rrf",
                        lambda lists, top_k, k: seen.update(
                            lists=lists, top_k=top_k, k=k) or ["h"])
    assert ev.retrieve_b("q?") == ["h"]
    assert seen == {"lists": [["d"], ["b"]], "top_k": 5, "k": 60}


def test_shared_generator_validates_citations(monkeypatch):
    ev = _load_evaluate_module()
    chunks = [{"content": "evidence text",
               "metadata": {"source": "s", "title": "t"}}]
    monkeypatch.setattr(ev, "call_llm", lambda sys, usr: "Trả lời [1]")
    answer, refused = ev.generate("q?", chunks)
    assert (answer, refused) == ("Trả lời [1]", False)
    monkeypatch.setattr(ev, "call_llm", lambda sys, usr: "Trả lời [9]")
    answer, refused = ev.generate("q?", chunks)
    assert refused and answer == ev.INSUFFICIENT_EVIDENCE
    answer, refused = ev.generate("q?", [])
    assert refused


def test_evaluation_results_schema():
    result = json.loads((EVAL_DIR / "evaluation_results.json")
                        .read_text(encoding="utf-8"))
    assert set(result["summary"]) == {"A", "B"}
    for name in ("A", "B"):
        assert result["summary"][name]["n"] >= 15
        for metric in ("faithfulness", "answer_relevance",
                       "context_recall", "context_precision"):
            value = result["summary"][name]["avg"][metric]
            assert 0.0 <= value <= 1.0
    assert len(result["cases"]) == 2 * result["summary"]["A"]["n"]
    assert result["meta"]["top_k"] == 5 and result["meta"]["rrf_k"] == 60
    assert result["meta"]["evaluator_model"]
    first = result["cases"][0]
    assert {"judge", "diagnostics", "judge_failed"} <= set(first)


def test_result_reports_completed():
    for path in (EVAL_DIR / "RESULT.md", REPO / "reports" / "RESULT.md"):
        report = path.read_text(encoding="utf-8")
        assert "TODO" not in report, f"placeholder left in {path}"
        lowered = report.lower()
        for heading in ("overall scores", "a/b comparison",
                        "worst performers", "recommendations"):
            assert heading in lowered, f"{heading} missing in {path}"
        assert "lexical proxy" not in lowered, f"stale proxy wording in {path}"


class _FakePost:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        payload = self.payloads[min(self.calls - 1, len(self.payloads) - 1)]
        return _FakeResponse(payload)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return {"choices": [{"message": {"content": self._payload}}]}


def _isolate_judge(monkeypatch, tmp_path):
    ev = _load_evaluate_module()
    monkeypatch.setattr(ev, "JUDGE_MODEL", "test-judge")
    monkeypatch.setattr(ev, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return ev


def test_judge_recall_parses_claim_verdicts(monkeypatch, tmp_path):
    import requests
    ev = _isolate_judge(monkeypatch, tmp_path)
    body = json.dumps({"claims": [
        {"claim": "c1", "supported": True, "evidence": "q1"},
        {"claim": "c2", "supported": False, "evidence": "none"},
        {"claim": "c3", "supported": True, "evidence": "q3"}]})
    monkeypatch.setattr(requests, "post", _FakePost([body]))
    score, evidence = ev.judge_context_recall(
        {"expected_answer": "a", "expected_context": "c"}, ["ctx"])
    assert score == pytest.approx(2 / 3)
    assert [c["claim"] for c in evidence["claims"]] == ["c1", "c2", "c3"]
    assert "supported" in evidence["claims"][0]


def test_judge_precision_preserves_rank(monkeypatch, tmp_path):
    import requests
    ev = _isolate_judge(monkeypatch, tmp_path)
    body = json.dumps({"chunks": [
        {"chunk_index": 1, "relevant": True, "reason": "direct hit"},
        {"chunk_index": 2, "relevant": False, "reason": "off-topic"}]})
    monkeypatch.setattr(requests, "post", _FakePost([body]))
    score, evidence = ev.judge_context_precision({"question": "q?"}, ["a", "b"])
    assert score == 0.5
    assert [c["chunk_index"] for c in evidence["chunks"]] == [1, 2]


def test_judge_relevance_clamped_and_scored(monkeypatch, tmp_path):
    import requests
    ev = _isolate_judge(monkeypatch, tmp_path)
    body = json.dumps({"score": 1.7, "reason": "perfect"})
    monkeypatch.setattr(requests, "post", _FakePost([body]))
    score, evidence = ev.judge_answer_relevance({"question": "q?"}, "a")
    assert score == 1.0
    assert evidence["reason"] == "perfect"


def test_judge_invalid_json_raises(monkeypatch, tmp_path):
    import requests
    ev = _isolate_judge(monkeypatch, tmp_path)
    monkeypatch.setattr(requests, "post", _FakePost(["not json{"]))
    with pytest.raises(ev.JudgeError):
        ev.judge_answer_relevance({"question": "q?"}, "a")


def test_judge_missing_key_raises(monkeypatch, tmp_path):
    ev = _load_evaluate_module()
    monkeypatch.setattr(ev, "JUDGE_MODEL", "test-judge")
    monkeypatch.setattr(ev, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(ev, "_judge_api_key", lambda: "")
    with pytest.raises(ev.JudgeError, match="OPENAI_API_KEY"):
        ev.judge_answer_relevance({"question": "q?"}, "a")


def test_judge_cache_avoids_duplicate_calls(monkeypatch, tmp_path):
    import requests
    ev = _isolate_judge(monkeypatch, tmp_path)
    fake = _FakePost([json.dumps({"score": 0.8, "reason": "ok"})])
    monkeypatch.setattr(requests, "post", fake)
    case = {"question": "q?"}
    assert ev.judge_answer_relevance(case, "a")[0] == 0.8
    assert ev.judge_answer_relevance(case, "a")[0] == 0.8
    assert fake.calls == 1
    assert (tmp_path / "cache.json").exists()


def test_evaluate_case_falls_back_cleanly(monkeypatch, tmp_path):
    ev = _isolate_judge(monkeypatch, tmp_path)
    monkeypatch.setattr(ev, "_judge_call",
                        lambda prompt: (_ for _ in ()).throw(
                            ev.JudgeError("down")))
    out = ev.evaluate_case(
        {"expected_answer": "apple pie", "expected_context": "apple",
         "question": "apple?"},
        [{"id": "c::chunk-0", "content": "apple pie today"}],
        "apple pie", False)
    assert out["judge_failed"] is True
    assert out["faithfulness"] == 1.0  # lexical fallback value
    assert out["judge"] == {"error": "down"}


def test_evaluate_case_refusal_rules_skip_judge(monkeypatch, tmp_path):
    ev = _isolate_judge(monkeypatch, tmp_path)
    calls = []
    payload = {"claims": [{"claim": "c", "supported": True,
                           "evidence": "e"}],
               "chunks": [{"chunk_index": 1, "relevant": True,
                           "reason": "r"}]}

    def fake_judge(prompt):
        calls.append(prompt)
        return payload

    monkeypatch.setattr(ev, "_judge_call", fake_judge)
    out = ev.evaluate_case(
        {"expected_answer": "a", "expected_context": "c", "question": "q?"},
        [{"id": "c::chunk-0", "content": "ctx"}],
        ev.INSUFFICIENT_EVIDENCE, True)
    assert out["faithfulness"] == 1.0
    assert out["answer_relevance"] == 0.0
    assert out["judge_failed"] is False
    assert len(calls) == 2  # recall + precision only
