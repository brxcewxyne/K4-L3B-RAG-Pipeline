"""Shared Task 5-7 orchestration for chat and retrieval flow."""
import os
from dotenv import load_dotenv

load_dotenv()
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD") or "0.3")


def retrieve_with_trace(query: str, top_k: int = 5,
                        score_threshold: float = SCORE_THRESHOLD,
                        use_reranking: bool = True) -> dict:
    """Run Tasks 5-7 against the existing Task 4 collection; never fake results."""
    from src.task4_chunking_indexing import get_collection
    from src.task5_semantic_search import semantic_search
    from src import task6_lexical_search as lexical
    from src.task7_reranking import rerank_rrf

    query = (query or "").strip()
    if not query:
        raise ValueError("Enter a retrieval query.")
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    top_k = min(top_k, 5)
    collection = get_collection()
    stored = collection.get(include=["documents", "metadatas"])
    lexical.CORPUS = [
        {"id": cid, "content": content, "metadata": metadata or {}}
        for cid, content, metadata in zip(
            stored["ids"], stored["documents"], stored["metadatas"])
    ]
    depth = min(max(10, top_k), len(stored["ids"]))
    dense = semantic_search(query, top_k=depth) if depth else []
    bm25 = lexical.lexical_search(query, top_k=depth) if depth else []
    fused = (rerank_rrf([dense, bm25], top_k=top_k)
             if use_reranking else dense[:top_k])
    selected = fused if dense and dense[0]["score"] >= score_threshold else []
    dense_ranks = {item["id"]: rank for rank, item in enumerate(dense, 1)}
    bm25_ranks = {item["id"]: rank for rank, item in enumerate(bm25, 1)}

    def fields(item):
        metadata = item.get("metadata") or {}
        return {"chunk_id": item["id"], "content": item["content"],
                **{key: metadata[key] for key in
                   ("title", "source", "language", "authority_level")
                   if key in metadata}}

    def ranked(items):
        return [{**fields(item), "score": item["score"], "rank": rank,
                 "preview": item["content"]}
                for rank, item in enumerate(items, 1)]

    trace = {
        "query": query,
        "dense": {"metric": "cosine similarity", "results": ranked(dense)},
        "bm25": {"metric": "BM25 score", "results": ranked(bm25)},
        "rrf": {"metric": "RRF score (k=60)", "results": [
            {**fields(item), "rrf_score": item["score"], "final_rank": rank,
             "dense_rank": dense_ranks.get(item["id"]),
             "bm25_rank": bm25_ranks.get(item["id"])}
            for rank, item in enumerate(fused, 1)]},
        "selected_context": [{**fields(item), "final_rank": rank}
                             for rank, item in enumerate(selected, 1)],
        "is_mock": False,
        "backend_connected": True,
    }
    return {"chunks": selected, "retrieval_trace": trace}


def retrieve(query: str, top_k: int = 5,
             score_threshold: float = SCORE_THRESHOLD,
             use_reranking: bool = True) -> list[dict]:
    """Contract-compatible selected SearchResults; no PageIndex for the demo."""
    return retrieve_with_trace(query, top_k, score_threshold, use_reranking)["chunks"]
