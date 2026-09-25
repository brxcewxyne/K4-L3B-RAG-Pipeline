# -*- coding: utf-8 -*-
"""Repo evaluation: Config A (dense-only) vs Config B (hybrid + RRF).

- Retrieval reuses production modules (task5/task6/task7); same top_k=5,
  RRF k=60; generation reuses task10.format_context + task10.call_llm.
- Primary metrics are semantic LLM-judge scores (same judge model, prompt
  style and parameters for Config A and Config B). Judge output is
  structured JSON (verdicts + concise reason/evidence only, no
  chain-of-thought). Results are cached on disk to avoid duplicate API
  calls; judge/API failures fall back cleanly to lexical diagnostics.
- Legacy lexical overlap helpers are kept as optional diagnostics only
  (``diagnostics`` per row), never as primary scores.
- Live run calls real embedding + LLM APIs and writes
  group_project/evaluation/evaluation_results.json.

Run: ``python group_project/evaluation/evaluate.py``
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src import task6_lexical_search as lexical_module  # noqa: E402
from src.task5_semantic_search import semantic_search  # noqa: E402
from src.task6_lexical_search import lexical_search  # noqa: E402
from src.task7_reranking import rerank_rrf  # noqa: E402
from src.task10_generation import (  # noqa: E402
    INSUFFICIENT_EVIDENCE,
    GenerationError,
    call_llm,
    format_context,
)

EVAL_DIR = REPO / "group_project" / "evaluation"
CACHE_PATH = EVAL_DIR / ".judge_cache.json"
TOP_K = 5
RRF_K = 60

JUDGE_MODEL = ""
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 800
JUDGE_TIMEOUT_S = 60

CITATION_RE = re.compile(r"\[(\d+)\]")
LINK_RE = re.compile(r"https?://|www\.|\]\(")


class JudgeError(RuntimeError):
    """Judge/API failure (message never contains credentials)."""


def _judge_model() -> str:
    global JUDGE_MODEL
    if not JUDGE_MODEL:
        import os

        from dotenv import load_dotenv
        load_dotenv()
        from src.task10_generation import LLM_MODEL
        JUDGE_MODEL = (os.getenv("JUDGE_MODEL", "") or LLM_MODEL).strip()
    return JUDGE_MODEL


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False),
                          encoding="utf-8")


JUDGE_SYSTEM = (
    "You are a strict RAG evaluation judge. The corpus mixes Vietnamese "
    "and English; cross-lingual paraphrase counts as equivalent, never "
    "require exact wording. Respond with ONLY a JSON object matching the "
    "requested schema. No chain-of-thought, no extra text.")
 

def _judge_api_key() -> str:
    """Read the judge credential (isolated for tests; never logged)."""
    from dotenv import load_dotenv
    load_dotenv()
    return (os.getenv("OPENAI_API_KEY") or "").strip()


def _judge_call(user_prompt: str) -> dict:
    """One deterministic judge call (temperature 0, JSON mode), disk-cached."""
    import requests
    key = _judge_api_key()
    if not key:
        raise JudgeError("Missing OPENAI_API_KEY for the LLM judge.")
    cache = _load_cache()
    cache_key = hashlib.sha1(
        (_judge_model() + "\n" + user_prompt).encode("utf-8")).hexdigest()
    if cache_key in cache:
        cached = cache[cache_key]
        if isinstance(cached, dict):
            return cached
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},  # never logged
            json={"model": _judge_model(), "temperature": JUDGE_TEMPERATURE,
                  "response_format": {"type": "json_object"},
                  "max_completion_tokens": JUDGE_MAX_TOKENS,
                  "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                               {"role": "user", "content": user_prompt}]},
            timeout=JUDGE_TIMEOUT_S)
    except Exception as exc:
        raise JudgeError(f"Judge request failed: {type(exc).__name__}") from None
    if response.status_code != 200:
        raise JudgeError(f"Judge API HTTP {response.status_code}")
    try:
        payload = response.json()["choices"][0]["message"]["content"]
        data = json.loads(payload)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise JudgeError(f"Judge returned invalid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise JudgeError("Judge returned non-object JSON")
    cache[cache_key] = data
    _save_cache(cache)
    return data


def _short(text: object, limit: int = 300) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "…"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "yes", "1", "supported")


def judge_context_recall(case: dict, retrieved: list[str]) -> tuple[float, dict]:
    """Split reference into atomic claims; score supported/total."""
    reference = (f"Expected answer: {case['expected_answer']}\n"
                 f"Expected evidence: {case['expected_context']}")
    contexts = "\n".join(f"[{i}] {doc}" for i, doc in enumerate(retrieved, 1))
    data = _judge_call(
        "Split the REFERENCE into at most 8 atomic factual claims. For each "
        "claim give a verdict whether it is supported by the RETRIEVED "
        'CONTEXTS.\nSchema: {"claims": [{"claim": "<short claim>", '
        '"supported": true/false, "evidence": "<short quote or \'none\'>"}]}'
        f"\nREFERENCE:\n{reference}\nRETRIEVED CONTEXTS:\n{contexts}")
    claims = data.get("claims", [])
    if not isinstance(claims, list) or not claims:
        raise JudgeError("Judge returned no recall claims")
    verdicts = [bool(_as_bool(c.get("supported"))) for c in claims
                if isinstance(c, dict)]
    if not verdicts:
        raise JudgeError("Judge returned no recall verdicts")
    evidence = [{"claim": _short(c.get("claim")),
                 "supported": bool(_as_bool(c.get("supported"))),
                 "evidence": _short(c.get("evidence"))}
                for c in claims if isinstance(c, dict)]
    return sum(verdicts) / len(verdicts), {"claims": evidence}


def judge_context_precision(case: dict, retrieved: list[str]) -> tuple[float, dict]:
    """Judge each Top-5 chunk: does it materially help answer the question?"""
    chunks = "\n".join(f"[{i}] {doc}" for i, doc in enumerate(retrieved, 1))
    data = _judge_call(
        "For each numbered retrieved chunk, judge whether it materially "
        "helps answer the QUESTION (rank order is preserved in the output)."
        '\nSchema: {"chunks": [{"chunk_index": <1-based int>, '
        '"relevant": true/false, "reason": "<short reason>"}]}'
        f"\nQUESTION:\n{case['question']}\nRETRIEVED CHUNKS:\n{chunks}")
    items = data.get("chunks", [])
    if not isinstance(items, list) or not items or not retrieved:
        raise JudgeError("Judge returned no precision verdicts")
    verdicts = [bool(_as_bool(c.get("relevant"))) for c in items
                if isinstance(c, dict)]
    if not verdicts:
        raise JudgeError("Judge returned no precision verdicts")
    evidence = [{"chunk_index": c.get("chunk_index"),
                 "relevant": bool(_as_bool(c.get("relevant"))),
                 "reason": _short(c.get("reason"))}
                for c in items if isinstance(c, dict)]
    return sum(verdicts) / len(retrieved), {"chunks": evidence}


def judge_faithfulness(answer: str, retrieved: list[str]) -> tuple[float, dict]:
    """Split answer into factual claims; score supported/total."""
    contexts = "\n".join(f"[{i}] {doc}" for i, doc in enumerate(retrieved, 1))
    data = _judge_call(
        "Split the ANSWER into atomic factual claims (ignore citation "
        "markers like [1]). For each claim give a verdict whether it is "
        "supported by the RETRIEVED CONTEXTS."
        '\nSchema: {"claims": [{"claim": "<short claim>", '
        '"supported": true/false, "evidence": "<short quote or \'none\'>"}]}'
        f"\nANSWER:\n{answer}\nRETRIEVED CONTEXTS:\n{contexts}")
    claims = data.get("claims", [])
    if not isinstance(claims, list) or not claims:
        raise JudgeError("Judge returned no faithfulness claims")
    verdicts = [bool(_as_bool(c.get("supported"))) for c in claims
                if isinstance(c, dict)]
    if not verdicts:
        raise JudgeError("Judge returned no faithfulness verdicts")
    evidence = [{"claim": _short(c.get("claim")),
                 "supported": bool(_as_bool(c.get("supported"))),
                 "evidence": _short(c.get("evidence"))}
                for c in claims if isinstance(c, dict)]
    return sum(verdicts) / len(verdicts), {"claims": evidence}


def judge_answer_relevance(case: dict, answer: str) -> tuple[float, dict]:
    """Judge how directly and completely the answer addresses the question."""
    data = _judge_call(
        "Judge how directly and completely the ANSWER addresses the "
        "QUESTION. Output a single score from 0 (misses the question) to 1 "
        "(fully and directly answers it)."
        '\nSchema: {"score": <number 0-1>, "reason": "<short reason>"}'
        f"\nQUESTION:\n{case['question']}\nANSWER:\n{answer}")
    try:
        score = float(data.get("score"))
    except (TypeError, ValueError):
        raise JudgeError("Judge returned no relevance score") from None
    score = max(0.0, min(1.0, score))
    return score, {"score": score, "reason": _short(data.get("reason"))}


def tok(text: str) -> set[str]:
    """Lowercase unicode word tokens (diagnostic helper basis)."""
    return set(re.findall(r"\w+", (text or "").lower()))


def context_recall(expected_context: str, retrieved: list[str]) -> float:
    """DIAGNOSTIC lexical proxy (kept for fallback/debugging, not primary)."""
    exp = tok(expected_context)
    if not exp:
        return 0.0
    return len(exp & tok("\n".join(retrieved))) / len(exp)


def context_precision(expected_context: str, retrieved: list[str]) -> float:
    """DIAGNOSTIC lexical proxy (kept for fallback/debugging, not primary)."""
    if not retrieved:
        return 0.0
    exp = tok(expected_context)
    hits = sum(1 for doc in retrieved if len(tok(doc) & exp) >= 2)
    return hits / len(retrieved)


def faithfulness(answer: str, retrieved: list[str], refused: bool) -> float:
    """DIAGNOSTIC lexical proxy (kept for fallback/debugging, not primary)."""
    if refused:
        return 1.0
    words = tok(answer)
    if not words:
        return 0.0
    return len(words & tok("\n".join(retrieved))) / len(words)


def answer_relevance(answer: str, question: str) -> float:
    """DIAGNOSTIC lexical proxy (kept for fallback/debugging, not primary)."""
    words = tok(question)
    if not words:
        return 0.0
    return len(tok(answer) & words) / len(words)


def _bm25_corpus() -> list[dict]:
    if not lexical_module.CORPUS:
        from src.task4_chunking_indexing import get_collection
        stored = get_collection().get(include=["documents", "metadatas"])
        lexical_module.CORPUS = [
            {"id": cid, "content": content, "metadata": metadata or {}}
            for cid, content, metadata in
            zip(stored["ids"], stored["documents"], stored["metadatas"])]
    return lexical_module.CORPUS


def retrieve_a(question: str) -> list[dict]:
    """Config A: dense-only (task5 → top_k)."""
    return semantic_search(question, top_k=TOP_K)


def retrieve_b(question: str) -> list[dict]:
    """Config B: dense + BM25 → RRF (k=60) → top_k."""
    from src.task4_chunking_indexing import get_collection  # noqa: F401
    _bm25_corpus()
    depth = max(10, TOP_K)
    dense = semantic_search(question, top_k=depth)
    bm25 = lexical_search(question, top_k=depth)
    return rerank_rrf([dense, bm25], top_k=TOP_K, k=RRF_K)


SYSTEM_PROMPT = ("Bạn là trợ lý tra cứu chính sách PUBG. Trả lời ngắn gọn "
                 "bằng tiếng Việt. Chỉ sử dụng CONTEXT. Không tự bổ sung kiến "
                 "thức ngoài nguồn. Nếu không có bằng chứng trực tiếp, trả lời "
                 "nguyên văn: " + INSUFFICIENT_EVIDENCE + " Gắn từng nhận định "
                 "với nguồn bằng [1], [2], ... đúng số CONTEXT. Không tạo URL.")


def generate(question: str, chunks: list[dict]) -> tuple[str, bool]:
    """Shared generator for both configs. Returns (answer, refused)."""
    if not chunks:
        return INSUFFICIENT_EVIDENCE, True
    try:
        answer = call_llm(
            SYSTEM_PROMPT,
            f"CONTEXT:\n{format_context(chunks)}\n\nUSER:\n{question}")
    except GenerationError:
        return INSUFFICIENT_EVIDENCE, True
    citations = {int(v) for v in CITATION_RE.findall(answer)}
    if (INSUFFICIENT_EVIDENCE in answer or not citations
            or not citations <= set(range(1, len(chunks) + 1))
            or LINK_RE.search(answer)):
        return INSUFFICIENT_EVIDENCE, True
    return answer, False


def evaluate_case(case: dict, chunks: list[dict],
                  answer: str, refused: bool) -> dict:
    """Primary scores from the LLM judge; lexical values stay as diagnostics.

    A safe refusal makes no factual claims (faithfulness 1.0 by rule) and
    does not address the question (relevance 0.0 by rule), so no judge
    calls are spent on it. Any judge/API failure falls back cleanly to the
    lexical diagnostic for that metric and flags ``judge_failed``.
    """
    ctx = [c["content"] for c in chunks]
    diagnostics = {
        "faithfulness": round(faithfulness(answer, ctx, refused), 4),
        "answer_relevance": round(answer_relevance(answer, case["question"]), 4),
        "context_recall": round(context_recall(case["expected_context"], ctx), 4),
        "context_precision": round(context_precision(case["expected_context"], ctx), 4),
    }
    judge_failed = False
    judge_evidence: dict = {}
    try:
        recall, recall_ev = judge_context_recall(case, ctx)
        precision, precision_ev = judge_context_precision(case, ctx)
        if refused:
            faith, faith_ev = 1.0, {
                "rule": "safe refusal makes no factual claims"}
            relev, relev_ev = 0.0, {
                "rule": "refusal does not address the question"}
        else:
            faith, faith_ev = judge_faithfulness(answer, ctx)
            relev, relev_ev = judge_answer_relevance(case, answer)
        scores = {"faithfulness": faith, "answer_relevance": relev,
                  "context_recall": recall, "context_precision": precision}
        judge_evidence = {"context_recall": recall_ev,
                          "context_precision": precision_ev,
                          "faithfulness": faith_ev,
                          "answer_relevance": relev_ev}
    except JudgeError as exc:
        judge_failed = True
        scores = dict(diagnostics)
        judge_evidence = {"error": str(exc)[:200]}
    return {
        "faithfulness": round(scores["faithfulness"], 4),
        "answer_relevance": round(scores["answer_relevance"], 4),
        "context_recall": round(scores["context_recall"], 4),
        "context_precision": round(scores["context_precision"], 4),
        "judge_failed": judge_failed,
        "judge": judge_evidence,
        "diagnostics": diagnostics,
    }


def main() -> dict:
    golden = json.loads((EVAL_DIR / "golden_dataset.json")
                        .read_text(encoding="utf-8"))
    rows: list[dict] = []
    for case in golden:
        for name, retrieve in (("A", retrieve_a), ("B", retrieve_b)):
            chunks = retrieve(case["question"])
            answer, refused = generate(case["question"], chunks)
            rows.append({"id": case["id"], "config": name,
                         "question": case["question"], "answer": answer,
                         "refused": refused,
                         "chunk_ids": [c["id"] for c in chunks],
                         **evaluate_case(case, chunks, answer, refused)})
    summary = {}
    for name in ("A", "B"):
        sub = [r for r in rows if r["config"] == name]
        summary[name] = {
            "n": len(sub),
            "avg": {m: round(sum(r[m] for r in sub) / len(sub), 4)
                    for m in ("faithfulness", "answer_relevance",
                              "context_recall", "context_precision")},
        }
    result = {
        "meta": {
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "top_k": TOP_K, "rrf_k": RRF_K,
            "config_a": "dense-only (task5.semantic_search)",
            "config_b": "hybrid dense+BM25 with RRF (task5+task6->task7)",
            "generator": "task10.format_context + task10.call_llm (shared)",
            "evaluator_model": _judge_model(),
            "evaluator_params": {"temperature": JUDGE_TEMPERATURE,
                                 "response_format": "json_object",
                                 "max_completion_tokens": JUDGE_MAX_TOKENS},
            "metrics": ("LLM-judge claim/rank verdicts "
                        "(context_recall, context_precision, faithfulness, "
                        "answer_relevance); lexical overlap kept only in "
                        "per-row diagnostics"),
        },
        "summary": summary,
        "cases": rows,
    }
    (EVAL_DIR / "evaluation_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    header = f"{'id':<6}{'cfg':<4}{'faith':>7}{'relev':>7}{'recall':>7}{'prec':>7}  refused"
    print(header)
    for r in rows:
        print(f"{r['id']:<6}{r['config']:<4}{r['faithfulness']:>7.3f}"
              f"{r['answer_relevance']:>7.3f}{r['context_recall']:>7.3f}"
              f"{r['context_precision']:>7.3f}  {r['refused']}")
    print("A avg:", summary["A"]["avg"])
    print("B avg:", summary["B"]["avg"])
    return result


if __name__ == "__main__":
    main()
