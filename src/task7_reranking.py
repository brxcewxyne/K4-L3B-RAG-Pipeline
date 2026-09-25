"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.

-> Dùng Jina hoặc self host hoặc bất cứ công cụ nào bạn quen
"""


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult.

    Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.
    Score là RRF score, không cộng trực tiếp cosine/BM25 score.
    """
    if not ranked_lists:
        return []

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            # Giữ item đầu tiên gặp; metadata/content không bị mất
            if item_id not in items:
                items[item_id] = item

    ranked_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
    results = []
    for item_id in ranked_ids[:top_k]:
        fused = items[item_id].copy()
        fused["score"] = scores[item_id]
        fused["retrieval_method"] = "hybrid"
        results.append(fused)
    return results


if __name__ == "__main__":
    print("Implement rerank_rrf, then run contract tests.")
