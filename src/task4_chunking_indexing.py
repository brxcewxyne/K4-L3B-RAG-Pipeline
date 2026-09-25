"""
Task 4 — Chunking, embedding và indexing.

- Đọc toàn bộ Markdown trong data/standardized/ (output của Task 3).
- Chia văn bản bằng RecursiveCharacterTextSplitter (500 chars, overlap 50).
- Embed chunks bằng đúng một provider/model: BAAI/bge-m3 (sentence-transformers,
  CPU, local — không gọi API trả phí). Task 5 tái sử dụng ``embed_texts()``
  nên cùng embedding space được đảm bảo.
- Upsert vào ChromaDB persistent với cosine distance; ID ổn định nên chạy
  lại không tạo dữ liệu trùng.

Mỗi document/chunk tuân thủ docs/MODULE_CONTRACTS.md.

Chạy: ``python -m src.task4_chunking_indexing``
"""

import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"

_VALID_DOC_TYPES = {"legal", "news"}

_model = None


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


def _get_model():
    """Load một lần duy nhất; tái sử dụng cho mọi lần embed (kể cả Task 5)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed một batch văn bản; trả về đúng số vector, cùng dimension."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, batch_size=32, normalize_embeddings=True,
                           show_progress_bar=False)
    result = [list(map(float, row)) for row in vectors]
    if len(result) != len(texts):
        raise ValueError(
            f"embed_texts count mismatch: {len(result)} != {len(texts)}"
        )
    dims = {len(row) for row in result}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent embedding dimensions: {dims}")
    return result


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
    print(f"Embedded {len(chunks)} chunks "
          f"(model={EMBEDDING_MODEL}, dim={len(vectors[0])})")
    return chunks


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


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB (idempotent — chạy lại không trùng)."""
    collection = get_collection()
    batch = 100
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        collection.upsert(
            ids=[chunk["id"] for chunk in part],
            documents=[chunk["content"] for chunk in part],
            embeddings=[chunk["embedding"] for chunk in part],
            metadatas=[_chroma_metadata(chunk["metadata"]) for chunk in part],
        )
    print(f"Upserted {len(chunks)} chunks into '{COLLECTION_NAME}' "
          f"(count={collection.count()})")


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)

    ids = [chunk["id"] for chunk in embedded_chunks]
    dims = {len(chunk["embedding"]) for chunk in embedded_chunks}
    collection = get_collection()
    print("Documents loaded:", len(documents))
    print("Chunks created:", len(chunks))
    print("Unique chunk IDs:", len(set(ids)))
    print("Embedding model:", EMBEDDING_MODEL)
    print("Embedding dimension:", next(iter(dims)))
    print("Chroma collection:", COLLECTION_NAME)
    print("Collection count:", collection.count())
    print("Duplicate chunk IDs:", len(ids) - len(set(ids)))
    print("Validation: PASS")


if __name__ == "__main__":
    run_pipeline()
