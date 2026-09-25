# -*- coding: utf-8 -*-
"""Task 1 tests — network luôn được mock, không gọi HTTP thật."""

import pytest
import requests

import src.task1_collect_legal_docs as task1


FAKE_HTML = """<html><head><title>Chính sách thử nghiệm</title></head><body>
<article><h1>Điều 1. Phạm vi</h1><p>Nội dung chính sách bằng tiếng Việt
có dấu đầy đủ để kiểm tra font Unicode trong PDF.</p>
<ul><li>Điểm một</li><li>Điểm hai</li></ul></article>
</body></html>"""


class _FakeResponse:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(task1, "DATA_DIR", tmp_path / "legal")
    return tmp_path / "legal"


def test_setup_creates_directory(monkeypatch, tmp_path):
    target = _isolate(monkeypatch, tmp_path)
    task1.setup_directory()
    assert target.is_dir()


def test_registry_matches_manifest_contract():
    assert len(task1.LEGAL_DOCUMENTS) == 3
    for source in task1.LEGAL_DOCUMENTS:
        assert source["filename"].endswith(".pdf")
        assert source["language"] == "vi"
        assert source["source_type"] == "official_policy"
        assert source["authority_level"] == "primary"
        assert source["url"].startswith("https://www.pubg.com/vi/clause/")
    assert [s["filename"] for s in task1.LEGAL_DOCUMENTS] == [
        "pubg_rules_of_conduct_vi.pdf",
        "pubg_terms_of_service_vi.pdf",
        "pubg_privacy_policy_vi.pdf",
    ]


def test_successful_pdf_download(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(task1, "fetch_url",
                        lambda url, **k: FAKE_HTML.encode("utf-8"))
    results = task1.download_documents()
    assert len(results) == 3
    for row in results:
        dest = tmp_path / "legal" / row["local_path"].split("/")[-1]
        assert dest.is_file()
        assert task1.is_valid_pdf(dest.read_bytes())
        assert row["skipped"] is False
        assert len(row["sha1"]) == 40 and row["size_bytes"] > 1024
    # PDF tái trích xuất được tiếng Việt (font Unicode thật).
    from pypdf import PdfReader
    text = PdfReader(str(tmp_path / "legal"
                         / "pubg_rules_of_conduct_vi.pdf")).pages[0].extract_text()
    assert "Quy tắc ứng xử" in text and "Nguồn chính thức" in text


def test_invalid_http_response(monkeypatch):
    class _Session:
        def __init__(self):  # requests.Session() interface tối thiểu
            self.headers = {}

        def get(self, url, timeout=None):
            return _FakeResponse(500, b"error")

    monkeypatch.setattr(requests, "Session", lambda: _Session())
    with pytest.raises(task1.DownloadError, match="HTTP 500"):
        task1.fetch_url("https://example.com/x", max_attempts=1)


def test_html_error_page_is_not_valid_pdf():
    assert task1.is_valid_pdf(b"<html><body>Not Found</body></html>") is False
    assert task1.is_valid_pdf(b"") is False
    assert task1.is_valid_pdf(b"%PDF-1.4\n" + b"x" * 2000) is True


def test_existing_valid_file_is_reused(monkeypatch, tmp_path):
    target = _isolate(monkeypatch, tmp_path)
    target.mkdir(parents=True)
    (target / "pubg_rules_of_conduct_vi.pdf").write_bytes(
        b"%PDF-1.4\n" + b"y" * 5000)
    calls = []
    monkeypatch.setattr(
        task1, "fetch_url",
        lambda url, **k: calls.append(url) or FAKE_HTML.encode("utf-8"))
    results = task1.download_documents()
    by_id = {row["id"]: row for row in results}
    assert by_id["pubg_rules_of_conduct_vi"]["skipped"] is True
    # file hợp lệ không tải lại; 2 file còn thiếu vẫn được tải
    assert calls == [s["url"] for s in task1.LEGAL_DOCUMENTS
                     if s["id"] != "pubg_rules_of_conduct_vi"]


def test_failed_source_leaves_no_corrupt_file(monkeypatch, tmp_path):
    target = _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(task1, "fetch_url",
                        lambda url, **k: (_ for _ in ()).throw(
                            task1.DownloadError("HTTP 403")))
    with pytest.raises(task1.DownloadError, match="Task 1 failures"):
        task1.download_documents()
    leftovers = list(target.glob("*")) if target.exists() else []
    assert leftovers == []  # không file nửa vời, không .part


def test_empty_article_content_rejected(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(task1, "fetch_url",
                        lambda url, **k: b"<html><body></body></html>")
    with pytest.raises(task1.DownloadError, match="Task 1 failures"):
        task1.download_documents()
