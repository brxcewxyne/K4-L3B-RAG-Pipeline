# -*- coding: utf-8 -*-
"""Task 2 tests — HTTP luôn được mock, không gọi network thật."""

import json

import pytest
import requests

import src.task2_crawl_news as task2


VICTIM_HTML = """<html><head><title>Bản tin thử nghiệm</title></head><body>
<nav>Menu điều hướng rác</nav>
<article><h1>Tiêu đề bài viết</h1><p>Đoạn nội dung thứ nhất bằng tiếng Việt
với đầy đủ chi tiết về sự việc đang được tường thuật lại một cách khách quan.</p>
<p>Đoạn nội dung thứ hai cần được giữ lại đầy đủ, không tóm tắt, nhằm bảo đảm
tính toàn vẹn của nguồn tin gốc phục vụ cho hệ thống truy xuất.</p>
<p>Đoạn nội dung thứ ba bổ sung bối cảnh và các diễn biến liên quan tiếp theo.</p></article>
<footer>Chân trang rác cần loại bỏ</footer>
</body></html>"""

SUPPORT_HTML = """<html><head><title>Support article</title></head><body>
<article><h1>How to report</h1><p>Use the in-game report system to report
cheaters. Attach screenshots and videos as evidence for the support team
to review your case properly and fairly.</p><p>Provide the date of the
incident, the character name and the violation type so that every report
can be investigated thoroughly by the anti-cheat team.</p></article>
</body></html>"""


def _source(**overrides):
    base = dict(task2.NEWS_SOURCES[0])
    base.update(overrides)
    return base


def test_successful_article_crawl(monkeypatch):
    monkeypatch.setattr(task2, "fetch_html", lambda url, **k: VICTIM_HTML)
    article = task2.crawl_article(_source())
    assert {"title", "url", "publisher", "language", "source_type",
            "authority_level", "date_crawled", "topic",
            "content_markdown"} <= set(article)
    assert article["language"] == "vi"
    assert len(article["content_markdown"]) >= 200


def test_title_and_body_extraction(monkeypatch):
    monkeypatch.setattr(task2, "fetch_html", lambda url, **k: VICTIM_HTML)
    article = task2.crawl_article(_source())
    assert "Bản tin thử nghiệm" in article["title"]
    assert "Đoạn nội dung thứ nhất" in article["content_markdown"]
    assert "Đoạn nội dung thứ hai" in article["content_markdown"]
    assert "Menu điều hướng rác" not in article["content_markdown"]
    assert "Chân trang rác" not in article["content_markdown"]


def test_metadata_preservation(monkeypatch):
    monkeypatch.setattr(task2, "fetch_html", lambda url, **k: VICTIM_HTML)
    article = task2.crawl_article(_source())
    assert article["publisher"] == "PUBG / KRAFTON"
    assert article["source_type"] == "official_news"
    assert article["authority_level"] == "primary"
    assert article["topic"] == task2.NEWS_SOURCES[0]["topic"]
    assert article["url"] == task2.NEWS_SOURCES[0]["url"]


def test_english_fallback_metadata(monkeypatch):
    def fake_fetch(url, **kwargs):
        if "/hc/vi/" in url:
            raise task2.CrawlError("Failed to fetch x: HTTP 403")
        return SUPPORT_HTML

    monkeypatch.setattr(task2, "fetch_html", fake_fetch)
    source = _source(id="pubg_report_cheat_seller_vi",
                     url="https://support.pubg.com/hc/vi/articles/115004167334",
                     fallback_url="https://support.pubg.com/hc/en-us/articles/115004167334-x",
                     fallback_language="en")
    article = task2.crawl_article(source)
    assert article["url"] == source["fallback_url"]
    assert article["language"] == "en"  # không gán nhãn Việt cho nội dung Anh


def test_independent_secondary_classification(monkeypatch):
    monkeypatch.setattr(task2, "fetch_html", lambda url, **k: VICTIM_HTML)
    justice = next(s for s in task2.NEWS_SOURCES
                   if s["id"] == "justice_for_himass_tanvuu_vi")
    assert justice["source_type"] == "independent"
    assert justice["authority_level"] == "secondary"
    article = task2.crawl_article(justice)
    assert article["source_type"] == "independent"
    assert article["authority_level"] == "secondary"


def test_http_failure(monkeypatch):
    class _Session:
        def __init__(self):
            self.headers = {}

        def get(self, url, timeout=None):
            raise requests.ConnectionError("blocked")

    monkeypatch.setattr(requests, "Session", lambda: _Session())
    with pytest.raises(task2.CrawlError):
        task2.fetch_html("https://example.com/x", max_attempts=1)


def test_empty_content_rejected(monkeypatch):
    monkeypatch.setattr(task2, "fetch_html",
                        lambda url, **k: "<html><body><p>Hi</p></body></html>")
    with pytest.raises(task2.CrawlError, match="empty/invalid"):
        task2.crawl_article(_source())


def test_deterministic_output_filename():
    names = [s["output_filename"] for s in task2.NEWS_SOURCES]
    assert len(set(names)) == len(names) == 6
    assert all(name.endswith(".json") and "/" not in name for name in names)


def test_registry_covers_six_logical_sources():
    ids = [s["id"] for s in task2.NEWS_SOURCES]
    assert len(ids) == 6
    assert any(s["source_type"] == "independent" for s in task2.NEWS_SOURCES)
    assert sum(s["source_type"] == "official_support"
               for s in task2.NEWS_SOURCES) == 4
    assert sum(s["source_type"] == "official_news"
               for s in task2.NEWS_SOURCES) == 1


def test_crawl_all_writes_successes_and_reports_failures(monkeypatch, tmp_path):
    def fake_fetch(url, **kwargs):
        if "support.pubg.com" in url:
            raise task2.CrawlError("Failed: HTTP 403")
        return VICTIM_HTML

    monkeypatch.setattr(task2, "fetch_html", fake_fetch)
    result = task2.crawl_all(output_dir=tmp_path)
    assert len(result["saved"]) == 2  # pubg news + justice
    assert len(result["failed"]) == 4  # 4 support pages bị chặn
    written = sorted(p.name for p in tmp_path.glob("*.json"))
    assert written == sorted(result["saved"])
    for name in written:
        item = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        assert {"url", "title", "date_crawled", "content_markdown"} <= set(item)
        assert len(item["content_markdown"]) >= 200
    # nguồn lỗi không để lại file giả
    assert "pubg_report_cheat_seller_vi.json" not in written
