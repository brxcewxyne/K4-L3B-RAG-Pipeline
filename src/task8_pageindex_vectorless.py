"""
Task 8 — PageIndex vectorless fallback.

PageIndex điều hướng cấu trúc tài liệu (cây mục lục) bằng reasoning thay vì
vector similarity: KHÔNG dùng ChromaDB, embeddings, cosine hay BM25 ở đây.

Luồng lifecycle:
    PDF chính sách dài  ->  PageIndex submit_document  ->  chờ indexing
    ->  lưu pageindex_document_id vào registry local  ->  tái sử dụng
    ->  pageindex_search() khi Task 9 rớt ngưỡng dense confidence.

Tài liệu ưu tiên: 3 PDF chính sách tiếng Việt dài, có cấu trúc
(rules of conduct / terms of service / privacy policy). PageIndex SDK
chỉ nhận PDF nên upload đúng file gốc ``data/landing/legal/*.pdf`` —
KHÔNG upload song song bản Markdown.

Chạy upload: ``python -m src.task8_pageindex_vectorless``
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

REPO_ROOT = Path(__file__).parent.parent

# Ba chính sách dài, có cấu trúc — ứng viên PageIndex tốt nhất của corpus.
# Metadata mirror đúng frontmatter Task 3 để kết quả fallback đồng nhất
# với chunk dense/BM25 khi đi qua Task 9/10.
PAGEINDEX_DOCUMENTS = (
    {
        "key": "pubg_rules_of_conduct_vi",
        "local_path": "data/landing/legal/pubg_rules_of_conduct_vi.pdf",
        "title": "Quy tắc ứng xử PUBG (Rules of Conduct) — Tiếng Việt",
        "source": "KRAFTON, Inc.",
        "publisher": "KRAFTON, Inc.",
        "url": "https://www.pubg.com/vi/clause/rules_of_conduct/label_steam/latest",
        "language": "vi",
        "doc_type": "legal",
        "source_type": "official_policy",
        "authority_level": "primary",
    },
    {
        "key": "pubg_terms_of_service_vi",
        "local_path": "data/landing/legal/pubg_terms_of_service_vi.pdf",
        "title": "Điều khoản Dịch vụ PUBG (Terms of Service) — Tiếng Việt",
        "source": "KRAFTON, Inc.",
        "publisher": "KRAFTON, Inc.",
        "url": "https://www.pubg.com/vi/clause/term_of_service/label_steam/latest",
        "language": "vi",
        "doc_type": "legal",
        "source_type": "official_policy",
        "authority_level": "primary",
    },
    {
        "key": "pubg_privacy_policy_vi",
        "local_path": "data/landing/legal/pubg_privacy_policy_vi.pdf",
        "title": "Chính sách Bảo mật PUBG (Privacy Policy) — Tiếng Việt",
        "source": "KRAFTON, Inc.",
        "publisher": "KRAFTON, Inc.",
        "url": "https://www.pubg.com/vi/clause/privacy_policy/label_steam/latest",
        "language": "vi",
        "doc_type": "legal",
        "source_type": "official_policy",
        "authority_level": "primary",
    },
)

REGISTRY_PATH = REPO_ROOT / "data" / "pageindex_registry.json"

READY_TIMEOUT_S = float(os.getenv("PAGEINDEX_READY_TIMEOUT_S", "900"))
READY_POLL_S = float(os.getenv("PAGEINDEX_READY_POLL_S", "15"))
SEARCH_TIMEOUT_S = float(os.getenv("PAGEINDEX_SEARCH_TIMEOUT_S", "180"))
SEARCH_POLL_S = float(os.getenv("PAGEINDEX_SEARCH_POLL_S", "3"))

_client = None


class PageIndexError(RuntimeError):
    """Lỗi Task 8 (config/thiếu key/API). Không bao giờ chứa API key."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def get_client():
    """Khởi tạo một lần PageIndexClient từ PAGEINDEX_API_KEY (không log key)."""
    global _client
    if _client is None:
        from pageindex import PageIndexClient

        api_key = (os.getenv("PAGEINDEX_API_KEY") or "").strip()
        if not api_key:
            raise PageIndexError(
                "Missing PAGEINDEX_API_KEY. Set it in .env "
                "(never commit .env)."
            )
        _client = PageIndexClient(api_key=api_key)
    return _client


def discover_documents() -> list[dict]:
    """Liệt kê tài liệu PageIndex và validate file tồn tại."""
    entries = []
    missing = []
    for item in PAGEINDEX_DOCUMENTS:
        path = REPO_ROOT / item["local_path"]
        if not path.is_file():
            missing.append(item["local_path"])
            continue
        entries.append({**item, "abs_path": str(path),
                        "content_hash": _file_sha1(path)})
    if missing:
        raise PageIndexError(
            "PageIndex source files missing: " + ", ".join(missing)
        )
    return entries


def load_registry() -> dict:
    """Đọc registry local {documents: {key: {...}}}; thiếu file -> rỗng."""
    if not REGISTRY_PATH.exists():
        return {"documents": {}}
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "documents" not in data:
        raise PageIndexError(f"Malformed registry: {REGISTRY_PATH}")
    return data


def save_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def wait_until_ready(doc_id: str, timeout_s: float = READY_TIMEOUT_S,
                     poll_s: float = READY_POLL_S) -> None:
    """Chờ PageIndex indexing xong (bounded); hết giờ -> PageIndexError."""
    client = get_client()
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            ready = client.is_retrieval_ready(doc_id)
        except Exception as exc:
            raise PageIndexError(
                f"PageIndex readiness check failed doc_id={doc_id}: "
                f"{type(exc).__name__}: {str(exc)[:200]}"
            ) from None
        if ready:
            return
        if time.monotonic() >= deadline:
            raise PageIndexError(
                f"PageIndex indexing not ready after {timeout_s}s "
                f"doc_id={doc_id}"
            )
        time.sleep(poll_s)


def upload_documents(timeout_s: float = READY_TIMEOUT_S,
                     poll_s: float = READY_POLL_S) -> dict:
    """Upload/index tài liệu thiếu hoặc đổi nội dung; tái dùng ID cũ.

    Trả về mapping registry ``documents``. Tài liệu lỗi được liệt kê rõ
    trong PageIndexError tổng hợp; tài liệu thành công vẫn được lưu.
    """
    from pageindex import PageIndexAPIError

    entries = discover_documents()
    registry = load_registry()
    stored = registry.setdefault("documents", {})
    failures: dict[str, str] = {}
    for entry in entries:
        key = entry["key"]
        prev = stored.get(key, {})
        if (prev.get("pageindex_document_id")
                and prev.get("content_hash") == entry["content_hash"]):
            continue  # unchanged -> reuse, 0 API call
        try:
            submitted = get_client().submit_document(entry["abs_path"])
            doc_id = submitted.get("doc_id")
            if not doc_id:
                raise PageIndexError(
                    f"PageIndex submit returned no doc_id for {key}: "
                    f"{str(submitted)[:200]}"
                )
            wait_until_ready(doc_id, timeout_s=timeout_s, poll_s=poll_s)
        except (PageIndexAPIError, PageIndexError, OSError) as exc:
            failures[key] = f"{type(exc).__name__}: {str(exc)[:300]}"
            continue
        stored[key] = {
            "local_path": entry["local_path"],
            "source_url": entry["url"],
            "pageindex_document_id": doc_id,
            "content_hash": entry["content_hash"],
            "indexed_at": _utcnow(),
            "status": "ready",
        }
        save_registry(registry)
    if failures:
        detail = "; ".join(f"{key} ({msg})" for key, msg in failures.items())
        raise PageIndexError(f"PageIndex upload failures: {detail}")
    print(f"PageIndex registry ready: {len(stored)} documents -> {REGISTRY_PATH}")
    return stored


def _poll_retrieval(client, retrieval_id: str,
                    timeout_s: float = SEARCH_TIMEOUT_S,
                    poll_s: float = SEARCH_POLL_S) -> dict:
    """Poll kết quả retrieval (bounded); failed/timeout -> PageIndexError."""
    deadline = time.monotonic() + timeout_s
    last_status = "unknown"
    while True:
        try:
            result = client.get_retrieval(retrieval_id)
        except Exception as exc:
            raise PageIndexError(
                f"PageIndex get_retrieval failed: {type(exc).__name__}: "
                f"{str(exc)[:200]}"
            ) from None
        last_status = str(result.get("status", "")).lower() if isinstance(
            result, dict) else ""
        if not isinstance(result, dict):
            raise PageIndexError(
                f"PageIndex retrieval unexpected shape "
                f"retrieval_id={retrieval_id} (got {type(result).__name__})"
            )
        if last_status == "completed":
            return result
        if last_status in ("failed", "error", "cancelled"):
            raise PageIndexError(
                f"PageIndex retrieval {last_status} "
                f"retrieval_id={retrieval_id}"
            )
        if time.monotonic() >= deadline:
            raise PageIndexError(
                f"PageIndex retrieval timeout after {timeout_s}s "
                f"retrieval_id={retrieval_id} (last status={last_status})"
            )
        time.sleep(poll_s)


def _rank_compat_score(rank: int) -> float:
    """Điểm tương thích theo rank (1, 1/2, 1/3, ...).

    PageIndex retrieval KHÔNG trả relevance score số — đây KHÔNG phải
    cosine similarity, KHÔNG phải BM25 score, KHÔNG phải RRF score; chỉ để
    thỏa contract số + thứ tự giảm dần của downstream.
    """
    return 1.0 / max(rank, 1)


def _node_score(node: dict):
    """Dùng score số thật của provider nếu có, else None (sẽ dùng rank)."""
    for field in ("score", "relevance", "relevance_score", "rank_score"):
        value = node.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _normalize_node(entry: dict, node: dict, rank: int) -> dict | None:
    """Chuẩn hóa một retrieved node -> SearchResult retrieval_method=pageindex.

    KHÔNG bịa page/section: chỉ giữ field provider thật sự trả về.
    """
    if not isinstance(node, dict):
        return None
    # Provider field drift (legacy endpoint): node id may be "node_id" or
    # "id"; relevant_contents may nest content blocks one list deeper.
    node_id = str(node.get("node_id") or node.get("id") or f"node-{rank}")
    title = str(node.get("title") or entry["title"]).strip() or entry["title"]
    raw_blocks = node.get("relevant_contents", []) or []
    flat_blocks: list = []
    for block in raw_blocks:
        if isinstance(block, list):
            flat_blocks.extend(block)
        else:
            flat_blocks.append(block)
    parts: list[str] = []
    pages: list[int] = []
    sections: list[str] = []
    for block in flat_blocks:
        if not isinstance(block, dict):
            continue
        text = str(block.get("relevant_content") or "").strip()
        if text:
            parts.append(text)
        page = block.get("page_index")
        if isinstance(page, bool):
            continue
        if isinstance(page, (int, float)):
            pages.append(int(page))
        # NOTE: provider "physical_index" is a literal placeholder string,
        # not a page number — never mapped to page (no invented metadata).
        section = str(block.get("section_title") or "").strip()
        if section:
            sections.append(section)
    if not parts:  # fallback các field text thô nếu provider đổi shape
        for field in ("text", "markdown", "content"):
            text = str(node.get(field) or "").strip()
            if text:
                parts.append(text)
                break
    content = "\n\n".join(parts).strip()
    if not content:
        return None
    score = _node_score(node)
    if score is None:
        score = _rank_compat_score(rank)
    metadata = {
        "source": entry["source"],
        "title": title,
        "doc_type": entry["doc_type"],
        "url": entry["url"],
        "publisher": entry["publisher"],
        "language": entry["language"],
        "source_type": entry["source_type"],
        "authority_level": entry["authority_level"],
        "source_file": entry["local_path"],
        "document_id": entry["key"],
        "pageindex_document_id": entry["pageindex_document_id"],
        "pageindex_node_id": node_id,
        "chunk_index": rank - 1,
    }
    if pages:
        metadata["page"] = min(pages)
    if sections:
        metadata["section"] = sections[0]
    elif node.get("title"):
        metadata["section"] = str(node["title"])
    return {
        "id": f"pageindex::{entry['key']}::{node_id}",
        "content": content,
        "score": score,
        "metadata": metadata,
        "retrieval_method": "pageindex",
    }


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless search trên các policy đã index; trả SearchResult chuẩn repo.

    Thứ tự = thứ tự registry (doc) rồi thứ tự provider (node); cắt top_k.
    Hết tài liệu ready -> PageIndexError. Provider lỗi -> PageIndexError
    (Task 9 bắt và rớt an toàn về hybrid/safe-refusal).
    """
    from pageindex import PageIndexAPIError

    from src.contracts import validate_search_results

    query = (query or "").strip()
    if not query:
        raise ValueError("pageindex_search: query must be non-empty")
    if top_k < 1:
        raise ValueError("pageindex_search: top_k must be positive")

    registry = load_registry()
    stored = registry.get("documents", {})
    entries = []
    for item in PAGEINDEX_DOCUMENTS:
        row = stored.get(item["key"], {})
        doc_id = row.get("pageindex_document_id")
        if not doc_id:
            continue
        entries.append({**item, "pageindex_document_id": doc_id})
    if not entries:
        raise PageIndexError(
            "No PageIndex documents indexed. Run "
            "python -m src.task8_pageindex_vectorless first."
        )

    client = get_client()
    merged: list[dict] = []
    skipped: list[str] = []
    for entry in entries:
        try:
            ready = client.is_retrieval_ready(entry["pageindex_document_id"])
        except PageIndexAPIError as exc:
            raise PageIndexError(
                f"PageIndex readiness failed {entry['key']}: {str(exc)[:200]}"
            ) from None
        if not ready:
            skipped.append(entry["key"])
            continue
        try:
            submitted = client.submit_query(
                entry["pageindex_document_id"], query)
            retrieval_id = submitted.get("retrieval_id")
            if not retrieval_id:
                raise PageIndexError(
                    f"PageIndex submit_query returned no retrieval_id "
                    f"for {entry['key']}"
                )
            result = _poll_retrieval(client, retrieval_id)
        except PageIndexAPIError as exc:
            raise PageIndexError(
                f"PageIndex search failed {entry['key']}: {str(exc)[:200]}"
            ) from None
        nodes = result.get("retrieved_nodes", []) or []
        for node in nodes:
            item = _normalize_node(entry, node, len(merged) + 1)
            if item is not None:
                merged.append(item)
    if skipped and not merged:
        raise PageIndexError(
            "No PageIndex documents ready (skipped: "
            + ", ".join(skipped) + ")"
        )
    # Sắp giảm dần theo score (ổn định) — giữ đúng thứ tự provider vì score
    # rank-derived đơn điệu theo rank.
    merged.sort(key=lambda item: item["score"], reverse=True)
    results = merged[:top_k]
    validate_search_results(results)
    return results


if __name__ == "__main__":
    upload_documents()
