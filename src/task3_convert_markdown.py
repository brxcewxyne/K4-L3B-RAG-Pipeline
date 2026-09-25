"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

- PDF pháp lý (data/landing/legal/) được convert bằng MarkItDown
  (dependency đã khai báo trong pyproject.toml).
- JSON web/support/news (data/landing/web/, data/landing/news/ nếu có)
  dùng trực tiếp field ``content_markdown`` — không crawl lại HTML.
- Mỗi nguồn thô -> một file Markdown có YAML frontmatter, ghi vào
  data/standardized/legal/ và data/standardized/news/.

Chạy: ``python -m src.task3_convert_markdown``
"""

import json
import re
from pathlib import Path


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
MANIFEST_PATH = LANDING_DIR / "source_manifest.json"

LEGAL_SUFFIXES = {".pdf", ".doc", ".docx"}

# Thư mục JSON thô: "web" là vị trí corpus PUBG đã thu thập;
# "news" là quy ước skeleton gốc — đọc cả hai nếu tồn tại.
WEB_DIR_NAMES = ("web", "news")


def load_manifest() -> dict:
    """Map local_path (data/landing/...) -> manifest row để giữ provenance."""
    if not MANIFEST_PATH.exists():
        return {}
    rows = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {row["local_path"]: row for row in rows}


def stable_doc_id(stem: str) -> str:
    """ID tài liệu ổn định, suy từ tên file nguồn đã chuẩn hóa."""
    doc_id = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return doc_id or "document"


def yaml_value(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def build_frontmatter(meta: dict) -> str:
    lines = ["---"]
    order = ("id", "title", "source", "url", "publisher", "language",
             "doc_type", "source_type", "authority_level", "source_file",
             "date_crawled")
    for key in order:
        if key in meta and meta[key] is not None:
            lines.append(f"{key}: {yaml_value(meta[key])}")
    topics = meta.get("topic") or []
    if topics:
        lines.append("topic:")
        for item in topics:
            lines.append(f"  - {yaml_value(item)}")
    lines.append("---")
    return "\n".join(lines)


def write_markdown(path: Path, meta: dict, body: str) -> Path:
    body = re.sub(r"\n{3,}", "\n\n", body.strip())
    if not body:
        raise ValueError(f"Refusing to write empty standardized doc: {path.name}")
    text = f"{build_frontmatter(meta)}\n\n# {meta['title']}\n\n{body}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def convert_legal_docs(manifest: dict) -> list[Path]:
    """Convert PDF/DOCX trong landing/legal bằng MarkItDown."""
    from markitdown import MarkItDown

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()
    written: list[Path] = []
    for path in sorted(legal_dir.iterdir()):
        if not (path.is_file() and path.suffix.lower() in LEGAL_SUFFIXES):
            continue
        doc_id = stable_doc_id(path.stem)
        row = manifest.get(f"data/landing/legal/{path.name}", {})
        result = converter.convert(str(path))
        body = re.sub(r"\n{3,}", "\n\n", result.text_content.strip())
        # Dòng đầu của PDF thu thập lặp lại tiêu đề tài liệu; write_markdown
        # đã chèn "# title" nên loại bỏ dòng trùng để tránh heading kép.
        title = row.get("title", path.stem)
        first, _, rest = body.partition("\n")
        if first.strip() == title.strip():
            body = rest.strip()
        meta = {
            "id": doc_id,
            "title": title,
            "source": row.get("publisher", "KRAFTON, Inc."),
            "url": row.get("url"),
            "publisher": row.get("publisher", "KRAFTON, Inc."),
            "language": row.get("language", "vi"),
            "doc_type": "legal",
            "source_type": row.get("source_type", "official_policy"),
            "authority_level": row.get("authority_level", "primary"),
            "source_file": f"data/landing/legal/{path.name}",
            "date_crawled": row.get("date_crawled"),
        }
        written.append(write_markdown(output_dir / f"{path.stem}.md", meta, body))
    return written


def convert_news_articles() -> list[Path]:
    """Convert JSON web/news: metadata -> frontmatter, content_markdown -> body."""
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seen: set[str] = set()
    for dirname in WEB_DIR_NAMES:
        web_dir = LANDING_DIR / dirname
        if not web_dir.is_dir():
            continue
        for path in sorted(web_dir.glob("*.json")):
            if path.name in seen:
                continue
            seen.add(path.name)
            data = json.loads(path.read_text(encoding="utf-8"))
            doc_id = stable_doc_id(path.stem)
            meta = {
                "id": doc_id,
                "title": data["title"],
                "source": data.get("publisher", ""),
                "url": data.get("url"),
                "publisher": data.get("publisher", ""),
                # language trong JSON là authoritative (giữ "en" cho fallback).
                "language": data.get("language", "vi"),
                "doc_type": "news",
                "source_type": data.get("source_type", "official_support"),
                "authority_level": data.get("authority_level", "primary"),
                "source_file": f"data/landing/{dirname}/{path.name}",
                "date_crawled": data.get("date_crawled"),
                "topic": data.get("topic", []),
            }
            body = data.get("content_markdown", "")
            written.append(write_markdown(output_dir / f"{path.stem}.md", meta, body))
    return written


def validate_outputs(legal: list[Path], news: list[Path]) -> None:
    total = len(legal) + len(news)
    print(f"Raw sources: {total}")
    print(f"Standardized documents: {total}")
    print(f"Legal: {len(legal)}")
    print(f"News/support: {len(news)}")
    languages: dict[str, int] = {}
    ids: set[str] = set()
    for path in [*legal, *news]:
        text = path.read_text(encoding="utf-8")
        assert text.strip(), f"Empty standardized doc: {path}"
        assert text.startswith("---\n"), f"Missing frontmatter: {path}"
        assert "\nid: " in text.split("---", 2)[1], f"Missing id: {path}"
        match = re.search(r'^language: "(.*?)"', text, re.M)
        if match:
            languages[match.group(1)] = languages.get(match.group(1), 0) + 1
        match = re.search(r'^id: "(.*?)"', text, re.M)
        if match:
            assert match.group(1) not in ids, f"Duplicate id: {match.group(1)}"
            ids.add(match.group(1))
    print("Languages:")
    for lang in sorted(languages):
        print(f"  {lang}: {languages[lang]}")
    print("Validation: PASS")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    legal = convert_legal_docs(manifest)
    news = convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")
    validate_outputs(legal, news)


if __name__ == "__main__":
    convert_all()
