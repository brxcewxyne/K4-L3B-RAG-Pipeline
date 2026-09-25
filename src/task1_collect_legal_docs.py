# -*- coding: utf-8 -*-
"""Task 1 — Thu thập tài liệu chính sách/quy định (real downloader).

Ba chính sách PUBG/KRAFTON tiếng Việt là các *trang web chính thức*, nên
pipeline là: fetch HTML -> trích article (``src.html_extract``) -> dựng
lại PDF bằng fpdf2 (font Unicode của hệ thống). Không bịa nội dung, không
tóm tắt, không dịch — giữ nguyên văn bản nguồn.

Idempotency: file có sẵn vượt validation thì bỏ qua (không tải lại).
``force=True`` để tải lại có chủ đích.

Chạy: ``python -m src.task1_collect_legal_docs``
"""

from __future__ import annotations

import hashlib
import time
from datetime import date
from pathlib import Path

import requests

from src.html_extract import extract_article


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
TIMEOUT_S = 30
MAX_ATTEMPTS = 3
BACKOFF_S = 2.0
MIN_PDF_BYTES = 1024

# Font Unicode cho tiếng Việt: override qua TASK1_FONT_TTF, else dò hệ thống.
FONT_ENV_VAR = "TASK1_FONT_TTF"
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)
FONT_BOLD_CANDIDATES = (
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

LEGAL_DOCUMENTS = [
    {
        "id": "pubg_rules_of_conduct_vi",
        "title": "Quy tắc ứng xử PUBG (Rules of Conduct) — Tiếng Việt",
        "url": "https://www.pubg.com/vi/clause/rules_of_conduct/label_steam/latest",
        "filename": "pubg_rules_of_conduct_vi.pdf",
        "language": "vi",
        "publisher": "KRAFTON, Inc.",
        "source_type": "official_policy",
        "authority_level": "primary",
    },
    {
        "id": "pubg_terms_of_service_vi",
        "title": "Điều khoản Dịch vụ PUBG (Terms of Service) — Tiếng Việt",
        "url": "https://www.pubg.com/vi/clause/term_of_service/label_steam/latest",
        "filename": "pubg_terms_of_service_vi.pdf",
        "language": "vi",
        "publisher": "KRAFTON, Inc.",
        "source_type": "official_policy",
        "authority_level": "primary",
    },
    {
        "id": "pubg_privacy_policy_vi",
        "title": "Chính sách Bảo mật PUBG (Privacy Policy) — Tiếng Việt",
        "url": "https://www.pubg.com/vi/clause/privacy_policy/label_steam/latest",
        "filename": "pubg_privacy_policy_vi.pdf",
        "language": "vi",
        "publisher": "KRAFTON, Inc.",
        "source_type": "official_policy",
        "authority_level": "primary",
    },
]


class DownloadError(RuntimeError):
    """Lỗi tải/convert rõ nguồn (không fake success)."""


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def fetch_url(url: str, timeout_s: float = TIMEOUT_S,
              max_attempts: int = MAX_ATTEMPTS) -> bytes:
    """GET có retry/backoff; lỗi HTTP/network -> DownloadError rõ ràng."""
    import os

    last_error = "unknown error"
    session = requests.Session()
    session.headers.update({"User-Agent": os.getenv("TASK1_USER_AGENT",
                                                    USER_AGENT)})
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, timeout=timeout_s)
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 200 and response.content:
                return response.content
            last_error = (f"HTTP {response.status_code} "
                          f"({len(response.content)} bytes)")
        if attempt < max_attempts:
            time.sleep(BACKOFF_S * attempt)
    raise DownloadError(f"Failed to fetch {url} after {max_attempts} "
                        f"attempts: {last_error}")


def is_valid_pdf(data: bytes) -> bool:
    """PDF hợp lệ tối thiểu: signature %PDF + kích thước hợp lý."""
    return len(data) >= MIN_PDF_BYTES and data.startswith(b"%PDF")


def find_unicode_font() -> tuple[str, str]:
    """(regular, bold) TTF hỗ trợ tiếng Việt; thiếu -> DownloadError."""
    import os

    override = (os.getenv(FONT_ENV_VAR) or "").strip()
    if override:
        candidates = ([override], [override])
    else:
        candidates = (FONT_CANDIDATES, FONT_BOLD_CANDIDATES)
    regular = next((p for p in candidates[0] if Path(p).is_file()), None)
    bold = next((p for p in candidates[1] if Path(p).is_file()), regular)
    if regular is None:
        raise DownloadError(
            "No Unicode TTF font found for Vietnamese PDF output. Set "
            f"{FONT_ENV_VAR} to a TTF path (e.g. Arial/DejaVuSans).")
    return regular, bold


# Arial thiếu glyph số tròn ①-⑳ -> thay bằng (1)-(20) để không mất nội dung.
_CIRCLED_DIGITS = {chr(0x245F + n): f"({n})" for n in range(1, 21)}


def _ascii_compat(text: str) -> str:
    for circled, plain in _CIRCLED_DIGITS.items():
        text = text.replace(circled, plain)
    return text


def article_to_pdf_bytes(title: str, source_url: str, publisher: str,
                         date_str: str, blocks: list[str]) -> bytes:
    """Dựng PDF từ các block văn bản (giữ nguyên văn, không tóm tắt)."""
    from fpdf import FPDF

    regular, bold = find_unicode_font()
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=20)
    pdf.add_font("body", "", regular)
    pdf.add_font("body", "B", bold)
    pdf.set_title(title)
    pdf.set_author(publisher)
    pdf.add_page()
    pdf.set_font("body", "B", 18)
    pdf.multi_cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("body", "", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(0, 6, f"Nguồn chính thức: {source_url}",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 6, f"Nhà xuất bản: {publisher} | Ngày thu thập: {date_str}",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)
    for block in blocks:
        text = _ascii_compat(block.strip())
        if not text:
            continue
        if text.startswith("#### "):
            pdf.set_font("body", "B", 11)
            pdf.multi_cell(0, 7, text[5:].strip(), new_x="LMARGIN", new_y="NEXT")
        elif text.startswith("### "):
            pdf.set_font("body", "B", 12)
            pdf.multi_cell(0, 7.5, text[4:].strip(), new_x="LMARGIN", new_y="NEXT")
        elif text.startswith("## ") or text.startswith("# "):
            pdf.set_font("body", "B", 13)
            pdf.multi_cell(0, 8, text.lstrip("# ").strip(),
                           new_x="LMARGIN", new_y="NEXT")
        elif text.startswith("- "):
            pdf.set_font("body", "", 10.5)
            pdf.multi_cell(0, 6, "\u2022  " + text[2:].strip(),
                           new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("body", "", 10.5)
            pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1.5)
    out = pdf.output()
    return bytes(out) if isinstance(out, bytearray) else out


def sha1_of_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def download_documents(output_dir: Path | None = None,
                       force: bool = False) -> list[dict]:
    """Tải 3 PDF chính sách; bỏ qua file có sẵn còn hợp lệ (idempotent).

    Trả về rows tương thích manifest (title/local_path/url/publisher/
    language/source_type/authority_level/date_crawled + sha1/size_bytes/
    skipped). Thất bại nguồn nào -> DownloadError tổng hợp nêu tên.
    """
    target = Path(output_dir) if output_dir else DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    results: list[dict] = []
    failures: dict[str, str] = {}
    for source in LEGAL_DOCUMENTS:
        dest = target / source["filename"]
        try:
            if not force and dest.is_file() and is_valid_pdf(dest.read_bytes()):
                print(f"Skip (valid exists): {dest.name}")
                skipped = True
            else:
                html = fetch_url(source["url"]).decode("utf-8", errors="replace")
                article = extract_article(html)
                if not article["blocks"]:
                    raise DownloadError(f"No article content at {source['url']}")
                pdf_bytes = article_to_pdf_bytes(
                    source["title"], source["url"], source["publisher"],
                    date.today().isoformat(), article["blocks"])
                if not is_valid_pdf(pdf_bytes):
                    raise DownloadError(
                        f"Generated PDF failed validation: {source['id']}")
                tmp = dest.with_suffix(".pdf.part")
                tmp.write_bytes(pdf_bytes)
                tmp.replace(dest)  # atomic, không để lại file nửa vời
                print(f"Saved: {dest} ({len(pdf_bytes)} bytes)")
                skipped = False
            results.append({
                "id": source["id"],
                "title": source["title"],
                "local_path": f"data/landing/legal/{source['filename']}",
                "url": source["url"],
                "publisher": source["publisher"],
                "language": source["language"],
                "source_type": source["source_type"],
                "authority_level": source["authority_level"],
                "date_crawled": today,
                "sha1": sha1_of_file(dest),
                "size_bytes": dest.stat().st_size,
                "skipped": skipped,
            })
        except (DownloadError, OSError) as exc:
            failures[source["id"]] = str(exc)[:300]
    if failures:
        detail = "; ".join(f"{key} ({msg})" for key, msg in failures.items())
        raise DownloadError(f"Task 1 failures: {detail}")
    return results


if __name__ == "__main__":
    setup_directory()
    download_documents()
