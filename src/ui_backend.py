"""UI boundary for live retrieval and grounded generation."""
BACKEND_CONNECTED = True
BACKEND_ENTRYPOINT = "src.task10_generation:generate_with_citation"
TRACE_BACKEND_CONNECTED = True
TRACE_ENTRYPOINT = "src.task9_retrieval_pipeline:retrieve_with_trace"


def ask_question(query: str, top_k: int = 5) -> dict:
    from src.task10_generation import generate_with_citation
    result = generate_with_citation(query, top_k=top_k)
    # Preserve SearchResult contract while exposing fields used by source cards.
    result["sources"] = [
        {**source, **source["metadata"], "id": source["id"],
         "citation_id": source["citation_id"], "chunk_id": source["id"],
         "snippet": source["content"]}
        for source in result["sources"]
    ]
    return result


def handle_query(query: str, top_k: int = 5) -> dict:
    return ask_question(query, top_k)


def filter_sources(sources: list[dict], source_filter: str) -> list[dict]:
    """UI-only source filter. ``source_filter`` in all/official/independent."""
    normalized = (source_filter or "all").lower()
    if normalized in {"official"}:
        return [s for s in sources if s.get("source_type") == "official"]
    if normalized in {"independent"}:
        return [s for s in sources if s.get("source_type") == "independent"]
    return list(sources)


def get_retrieval_trace(query: str, top_k: int = 5) -> dict:
    from src.task9_retrieval_pipeline import retrieve_with_trace
    return retrieve_with_trace(query, top_k)["retrieval_trace"]
