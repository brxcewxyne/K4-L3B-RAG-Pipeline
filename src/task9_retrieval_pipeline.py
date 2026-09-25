"""Shared Task 5-7 orchestration for chat and retrieval flow.

Normal path: Dense + BM25 -> RRF (single fuse) -> dense-score gate.
Fallback path (Task 8): khi dense confidence yếu -> PageIndex vectorless
search; PageIndex lỗi -> rớt an toàn về hybrid (hoặc rỗng -> Task 10
safe refusal). PageIndex KHÔNG bao giờ là retriever mặc định.
"""

import os

from dotenv import load_dotenv

from src.task4_chunking_indexing import get_collection
from src.task5_semantic_search import semantic_search
import src.task6_lexical_search as lexical_module
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf
from src.task8_pageindex_vectorless import pageindex_search

load_dotenv()
# Calibrated 2026-09-25 on 16 in-domain + 16 out-of-domain Dense scores
# (see group_project/evaluation/threshold_calibration.json):
# max balanced accuracy, widest-margin midpoint of the optimal gap.
# Gate compares the ORIGINAL Dense top-1 cosine score with `>=`.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD") or "0.5728")


def _fields(item):
    metadata = item.get("metadata") or {}
    return {"chunk_id": item["id"], "content": item["content"],
            **{key: metadata[key] for key in
               ("title", "source", "language", "authority_level")
               if key in metadata}}


def _ranked(items):
    return [{**_fields(item), "score": item["score"], "rank": rank,
             "preview": item["content"]}
            for rank, item in enumerate(items, 1)]


def retrieve_with_trace(query: str, top_k: int = 5,
                        score_threshold: float = SCORE_THRESHOLD,
                        use_reranking: bool = True) -> dict:
    """Run Tasks 5-7 (+ Task 8 fallback) against Task 4 collection."""
    query = (query or "").strip()
    if not query:
        raise ValueError("Enter a retrieval query.")
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    top_k = min(top_k, 5)
    collection = get_collection()
    stored = collection.get(include=["documents", "metadatas"])
    lexical_module.CORPUS = [
        {"id": cid, "content": content, "metadata": metadata or {}}
        for cid, content, metadata in zip(
            stored["ids"], stored["documents"], stored["metadatas"])
    ]
    depth = min(max(10, top_k), len(stored["ids"]))
    dense = semantic_search(query, top_k=depth) if depth else []
    bm25 = lexical_search(query, top_k=depth) if depth else []
    fused = (rerank_rrf([dense, bm25], top_k=top_k)
             if use_reranking else dense[:top_k])
    dense_ranks = {item["id"]: rank for rank, item in enumerate(dense, 1)}
    bm25_ranks = {item["id"]: rank for rank, item in enumerate(bm25, 1)}

    # Confidence dùng dense cosine gốc — KHÔNG dùng RRF score.
    best_dense = dense[0]["score"] if dense else None
    fallback = {"triggered": False}
    pageindex_results: list[dict] = []
    if not dense:
        fallback = {"triggered": True, "reason": "no_dense_results",
                    "best_dense_score": None, "threshold": score_threshold,
                    "method": "pageindex"}
    elif best_dense < score_threshold:
        fallback = {"triggered": True,
                    "reason": "dense_score_below_threshold",
                    "best_dense_score": best_dense,
                    "threshold": score_threshold, "method": "pageindex"}
    if fallback["triggered"]:
        try:
            pageindex_results = pageindex_search(query, top_k=top_k)
            selected = list(pageindex_results)
            retrieval_source = "pageindex"
        except Exception:
            selected = list(fused)  # PageIndex lỗi -> rớt về hybrid
            retrieval_source = "hybrid"
            fallback["pageindex_error"] = True
    else:
        selected = list(fused)
        retrieval_source = "hybrid"

    trace = {
        "query": query,
        "dense": {"metric": "cosine similarity", "results": _ranked(dense)},
        "bm25": {"metric": "BM25 score", "results": _ranked(bm25)},
        "rrf": {"metric": "RRF score (k=60)", "results": [
            {**_fields(item), "rrf_score": item["score"], "final_rank": rank,
             "dense_rank": dense_ranks.get(item["id"]),
             "bm25_rank": bm25_ranks.get(item["id"])}
            for rank, item in enumerate(fused, 1)]},
        "fallback": fallback,
        "pageindex": {"metric": "PageIndex vectorless (tree navigation)",
                      "results": _ranked(pageindex_results)},
        "selected_context": [{**_fields(item), "final_rank": rank}
                             for rank, item in enumerate(selected, 1)],
        "retrieval_source": retrieval_source,
        "is_mock": False,
        "backend_connected": True,
    }
    return {"chunks": selected, "retrieval_source": retrieval_source,
            "retrieval_trace": trace}


def retrieve(query: str, top_k: int = 5,
             score_threshold: float = SCORE_THRESHOLD,
             use_reranking: bool = True) -> list[dict]:
    """Contract-compatible selected SearchResults (hybrid hoặc pageindex)."""
    return retrieve_with_trace(query, top_k, score_threshold, use_reranking)["chunks"]
