"""
Task 4 — Chunking, embedding và indexing.

- Đọc toàn bộ Markdown trong data/standardized/ (output của Task 3).
- Chia văn bản bằng RecursiveCharacterTextSplitter (500 chars, overlap 50).
- Embed chunks qua EXTERNAL EMBEDDING API (không dùng model local):
  ``embed_texts()`` đứng sau một provider abstraction (``EmbeddingClient``)
  đọc cấu hình từ biến môi trường. Hỗ trợ provider OpenAI-compatible
  (OpenAI, Jina, gateway tương thích) và Gemini REST.
- Upsert vào ChromaDB persistent với cosine distance; ID ổn định +
  bỏ qua chunk đã index (cùng provider/model/nội dung) nên chạy lại
  không tốn API call trùng và không tạo dữ liệu trùng.

Mỗi document/chunk tuân thủ docs/MODULE_CONTRACTS.md.
Task 5 sẽ tái sử dụng ``embed_texts()`` và ``get_collection()`` nên cùng
embedding space được đảm bảo.

KHÔNG commit API key. Key đọc từ biến môi trường và không bao giờ bị log.

Chạy: ``python -m src.task4_chunking_indexing``
"""

import hashlib
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # repo .env (gitignored) supplies embedding API keys locally


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# --- Cấu hình embedding tập trung (provider ngoài, không model local) ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip() or None
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "").strip() or None
# Dimension thực tế do API trả về; resolve lúc runtime (không đoán trước).
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "0") or 0) or None

EMBEDDING_TIMEOUT_S = float(os.getenv("EMBEDDING_TIMEOUT_S", "60"))
EMBEDDING_MAX_ATTEMPTS = int(os.getenv("EMBEDDING_MAX_ATTEMPTS", "4"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

COLLECTION_NAME = "rag_documents"

_VALID_DOC_TYPES = {"legal", "news"}

_PROVIDER_DEFAULTS = {
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "model": "text-embedding-3-small",
        "key_vars": ("OPENAI_API_KEY", "EMBEDDING_API_KEY"),
    },
    "jina": {
        "api_base": "https://api.jina.ai/v1",
        "model": "jina-embeddings-v3",
        "key_vars": ("JINA_API_KEY", "EMBEDDING_API_KEY"),
    },
    "gemini": {
        "api_base": "https://generativelanguage.googleapis.com/v1beta",
        "model": "text-embedding-004",
        "key_vars": ("GEMINI_API_KEY", "EMBEDDING_API_KEY"),
    },
}

_client = None
_resolved_dim = EMBEDDING_DIM
_api_batches = 0


class EmbeddingError(RuntimeError):
    """Lỗi provider embedding (thông điệp chứa provider, không chứa key)."""


def _resolve_api_key(provider: str) -> str:
    key_vars = _PROVIDER_DEFAULTS[provider]["key_vars"]
    for var in key_vars:
        value = (os.getenv(var) or "").strip()
        if value:
            return value
    raise EmbeddingError(
        f"Missing API key for embedding provider={provider!r}. "
        f"Set one of: {', '.join(key_vars)} (never commit keys)."
    )


def _post_json(url: str, payload: dict, headers: dict,
               timeout_s: float) -> dict:
    """POST JSON với retry/backoff cho lỗi transient (429/5xx)."""
    import requests

    global _api_batches
    attempt = 0
    wait_s = 1.0
    last_error = "unknown error"
    while attempt < EMBEDDING_MAX_ATTEMPTS:
        attempt += 1
        try:
            response = requests.post(url, json=payload, headers=headers,
                                     timeout=timeout_s)
        except requests.RequestException as exc:
            last_error = f"connection error: {type(exc).__name__}"
        else:
            if response.status_code == 200:
                _api_batches += 1
                return response.json()
            if response.status_code in (429, 500, 502, 503, 504):
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_s = min(float(retry_after), 30.0)
                except (TypeError, ValueError):
                    pass
                last_error = f"HTTP {response.status_code} (transient)"
            else:
                detail = (response.text or "")[:300]
                raise EmbeddingError(
                    f"Embedding API error provider={EMBEDDING_PROVIDER!r} "
                    f"HTTP {response.status_code}: {detail}"
                )
        if attempt >= EMBEDDING_MAX_ATTEMPTS:
            break
        time.sleep(wait_s)
        wait_s = min(wait_s * 2, 30.0)
    raise EmbeddingError(
        f"Embedding API failed provider={EMBEDDING_PROVIDER!r} "
        f"after {attempt} attempts: {last_error}"
    )


class EmbeddingClient:
    """Boundary provider-agnostic cho external embedding API.

    Phần còn lại của Task 4 (và Task 5 sau này) chỉ gọi ``embed()``,
    không cần biết provider cụ thể.
    """

    def __init__(self, provider: str | None = None,
                 model: str | None = None,
                 api_base: str | None = None) -> None:
        name = (provider or EMBEDDING_PROVIDER).strip().lower()
        if name not in _PROVIDER_DEFAULTS:
            raise EmbeddingError(
                f"Unknown EMBEDDING_PROVIDER={name!r} "
                f"(expected one of {sorted(_PROVIDER_DEFAULTS)})"
            )
        defaults = _PROVIDER_DEFAULTS[name]
        self.provider = name
        self.model = (model or EMBEDDING_MODEL or defaults["model"]).strip()
        self.api_base = (api_base or EMBEDDING_API_BASE
                         or defaults["api_base"]).rstrip("/")
        self.api_key = _resolve_api_key(name)

    @property
    def config_stamp(self) -> dict:
        """Định danh cấu hình sinh vector (để Task 5 dùng đúng model)."""
        return {"embedding_provider": self.provider,
                "embedding_model": self.model}

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed một batch, giữ thứ tự input/output, trả list[list[float]]."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start:start + EMBEDDING_BATCH_SIZE]
            if self.provider == "gemini":
                vectors.extend(self._embed_gemini(batch))
            else:
                vectors.extend(self._embed_openai_compatible(batch))
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Embedding count mismatch provider={self.provider!r}: "
                f"{len(vectors)} != {len(texts)}"
            )
        dims = {len(row) for row in vectors}
        if len(dims) != 1:
            raise EmbeddingError(
                f"Inconsistent embedding dimensions: {sorted(dims)}"
            )
        global _resolved_dim
        _resolved_dim = next(iter(dims))
        return vectors

    def _embed_openai_compatible(self, batch: list[str]) -> list[list[float]]:
        data = _post_json(
            f"{self.api_base}/embeddings",
            {"model": self.model, "input": batch},
            {"Authorization": f"Bearer {self.api_key}",
             "Content-Type": "application/json"},
            EMBEDDING_TIMEOUT_S,
        )
        items = data.get("data", []) if isinstance(data, dict) else []
        ordered = sorted(items, key=lambda row: row.get("index", 0))
        result = []
        for row in ordered:
            values = row.get("embedding", [])
            result.append([float(x) for x in values])
        return result

    def _embed_gemini(self, batch: list[str]) -> list[list[float]]:
        data = _post_json(
            f"{self.api_base}/models/{self.model}:batchEmbedContents",
            {"requests": [
                {"model": f"models/{self.model}",
                 "content": {"parts": [{"text": text}]},
                 "taskType": "RETRIEVAL_DOCUMENT"}
                for text in batch
            ]},
            {"x-goog-api-key": self.api_key,
             "Content-Type": "application/json"},
            EMBEDDING_TIMEOUT_S,
        )
        items = data.get("embeddings", []) if isinstance(data, dict) else []
        return [[float(x) for x in row.get("values", [])] for row in items]


def get_embedding_client() -> EmbeddingClient:
    """Singleton client — không reload/khởi tạo lại mỗi lần gọi."""
    global _client
    if _client is None:
        _client = EmbeddingClient()
    return _client


def get_api_batch_count() -> int:
    """Số HTTP request embedding đã thực hiện trong tiến trình này."""
    return _api_batches


def _parse_frontmatter(path: Path) -> tuple[dict, str]:
    """Tách YAML frontmatter và Markdown body; raise ValueError rõ ràng."""
    import yaml

    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path.name}: missing YAML frontmatter")
    header, sep, body = text[len("---\n"):].partition("\n---\n")
    if not sep:
        raise ValueError(f"{path.name}: malformed YAML frontmatter")
    meta = yaml.safe_load(header) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"{path.name}: frontmatter must be a mapping")
    return meta, body


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document theo contract."""
    from src.contracts import validate_document

    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        meta, body = _parse_frontmatter(path)
        content = body.strip()
        if not content:
            print(f"Skip empty document: {path.name}")
            continue
        doc_type = str(meta.get("doc_type", "")).strip()
        if doc_type not in _VALID_DOC_TYPES:
            raise ValueError(
                f"{path.name}: invalid doc_type={doc_type!r} "
                f"(expected one of {sorted(_VALID_DOC_TYPES)})"
            )
        document = {
            "id": str(meta.get("id") or path.stem),
            "content": content,
            "metadata": {
                "source": str(meta.get("source") or path.name),
                "title": str(meta.get("title") or path.stem),
                "doc_type": doc_type,
                "url": meta.get("url"),
                "publisher": meta.get("publisher"),
                "language": meta.get("language"),
                "source_type": meta.get("source_type"),
                "authority_level": meta.get("authority_level"),
                "topic": meta.get("topic", []),
                "source_file": meta.get("source_file"),
                "document_id": str(meta.get("id") or path.stem),
            },
        }
        validate_document(document)
        documents.append(document)
    documents.sort(key=lambda item: item["id"])
    print(f"Loaded {len(documents)} documents from {STANDARDIZED_DIR}")
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id ổn định và chunk_index."""
    from src.contracts import validate_document

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for document in documents:
        for index, text in enumerate(splitter.split_text(document["content"])):
            if not text.strip():
                continue
            chunk = {
                # ID ổn định: cùng document + cùng cấu hình chunking = cùng ID.
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": {**document["metadata"], "chunk_index": index},
            }
            validate_document(chunk, require_chunk=True)
            chunks.append(chunk)
    ids = [chunk["id"] for chunk in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs generated"
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed một batch văn bản qua external API; cùng dimension, đúng số lượng.

    Task 5 tái sử dụng hàm này nên query và document dùng chung
    embedding space.
    """
    if not texts:
        return []
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("embed_texts: all inputs must be non-empty strings")
    return get_embedding_client().embed(texts)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk, giữ nguyên id/content/metadata."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError(
            f"embed_chunks count mismatch: {len(vectors)} != {len(chunks)}"
        )
    dims = {len(vector) for vector in vectors}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent embedding dimensions: {dims}")
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    client = get_embedding_client()
    print(f"Embedded {len(chunks)} chunks "
          f"(provider={client.provider}, model={client.model}, "
          f"dim={len(vectors[0])})")
    return chunks


def _content_sha1(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def _chroma_metadata(metadata: dict) -> dict:
    """Serialize metadata về kiểu primitive mà Chroma chấp nhận.

    - list (ví dụ ``topic``) -> chuỗi nối bằng "|" (deterministic).
    - None -> bỏ key (giữ nguyên trên chunk object, chỉ khác khi lưu Chroma).
    """
    clean: dict = {}
    for key in sorted(metadata):
        value = metadata[key]
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = "|".join(str(item) for item in value)
        if not isinstance(value, (str, int, float, bool)):
            value = str(value)
        clean[key] = value
    return clean


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _plan_indexing(chunks: list[dict], collection) -> tuple[list[dict], int]:
    """Chỉ embed/upsert chunk mới hoặc đổi nội dung/provider/model.

    Trả về (chunks_cần_xử_lý, số_chunk_bỏ_qua). Đọc state hiện có trong
    Chroma nên chạy lại trên corpus không đổi tốn 0 API call.
    """
    client = get_embedding_client()
    stamp = client.config_stamp
    ids = [chunk["id"] for chunk in chunks]
    stored: dict[str, dict] = {}
    for start in range(0, len(ids), 500):
        part = ids[start:start + 500]
        if not part:
            continue
        got = collection.get(ids=part, include=["metadatas"])
        for item_id, meta in zip(got.get("ids", []), got.get("metadatas", [])):
            stored[item_id] = meta or {}
    pending, skipped = [], 0
    for chunk in chunks:
        meta = stored.get(chunk["id"])
        if (meta and meta.get("embedding_provider") == stamp["embedding_provider"]
                and meta.get("embedding_model") == stamp["embedding_model"]
                and meta.get("content_sha1") == _content_sha1(chunk["content"])):
            skipped += 1
        else:
            pending.append(chunk)
    return pending, skipped


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB (idempotent — chạy lại không trùng)."""
    collection = get_collection()
    client = get_embedding_client()
    pending, skipped = _plan_indexing(chunks, collection)
    if pending:
        embedded = embed_chunks(pending)
        batch = 100
        for start in range(0, len(embedded), batch):
            part = embedded[start:start + batch]
            collection.upsert(
                ids=[chunk["id"] for chunk in part],
                documents=[chunk["content"] for chunk in part],
                embeddings=[chunk["embedding"] for chunk in part],
                metadatas=[{**_chroma_metadata(chunk["metadata"]),
                            "content_sha1": _content_sha1(chunk["content"]),
                            **client.config_stamp}
                           for chunk in part],
            )
    try:
        collection.modify(metadata={"hnsw:space": "cosine",
                                    **client.config_stamp,
                                    "embedding_dimension": str(_resolved_dim or "")})
    except Exception:
        pass
    print(f"Upserted {len(pending)} chunks into '{COLLECTION_NAME}' "
          f"(skipped already-indexed={skipped}, count={collection.count()})")


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    index_to_vectorstore(chunks)

    client = get_embedding_client()
    collection = get_collection()
    ids = [chunk["id"] for chunk in chunks]
    print("Documents loaded:", len(documents))
    print("Chunks created:", len(chunks))
    print("Unique chunk IDs:", len(set(ids)))
    print("Embedding provider:", client.provider)
    print("Embedding model:", client.model)
    print("Embedding dimension:", _resolved_dim)
    print("External embedding API batches:", get_api_batch_count())
    print("Chroma persistence path:", CHROMA_DIR)
    print("Chroma collection:", COLLECTION_NAME)
    print("Collection count:", collection.count())
    print("Duplicate chunk IDs:", len(ids) - len(set(ids)))
    print("Validation: PASS")


if __name__ == "__main__":
    run_pipeline()
