# -*- coding: utf-8 -*-
"""Task 2 — Crawl bài viết/thông báo (real crawler, stdlib HTML parsing).

Registry ``NEWS_SOURCES`` gồm 6 nguồn của corpus hiện tại. Mỗi nguồn thử
URL chính tắc trước; nguồn hỗ trợ bị chặn bot (403) thì dùng URL fallback
tiếng Anh đã document — và metadata ``language`` luôn phản ánh nội dung
thật sự tải về (không gán nhãn Việt cho nội dung Anh).

Nguồn độc lập (Justice) giữ ``source_type="independent"`` /
``authority_level="secondary"`` — không bao giờ gộp thành official.

Nguồn nào không truy cập được: báo lỗi rõ, KHÔNG ghi file giả, giữ
artifact cũ. Output JSON khớp schema repo
(title/url/publisher/language/source_type/authority_level/date_crawled/
topic/content_markdown) vào ``data/landing/news/``.

Chạy: ``python -m src.task2_crawl_news``
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import requests

from src.html_extract import extract_article


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
TIMEOUT_S = 30
MAX_ATTEMPTS = 3
BACKOFF_S = 2.0
MIN_CONTENT_CHARS = 200

NEWS_SOURCES = [
    {
        "id": "pubg_asia_stars_himass_tanvuu_investigation_vi",
        "url": "https://www.pubg.com/vi/news/11155",
        "title": "Thông Báo Kết Quả Điều Tra Và Biện Pháp Xử Lý Sự Cố Vận Hành PUBG Asia Stars",
        "publisher": "PUBG / KRAFTON",
        "language": "vi",
        "source_type": "official_news",
        "authority_level": "primary",
        "output_filename": "pubg_asia_stars_himass_tanvuu_investigation_vi.json",
        "fallback_url": "",
        "fallback_language": "",
        "topic": ["pubg_asia_stars", "himass_tanvuu", "stream_sniping",
                  "dieu_tra_vi_pham", "khoa_tai_khoan_vinh_vien",
                  "cam_thi_dau", "trach_nhiem_van_hanh"],
    },
    {
        "id": "justice_for_himass_tanvuu_vi",
        "url": "https://vnexpress.net/hon-4-1-trieu-chu-ky-yeu-cau-doi-lai-cong-bang-cho-himass-va-tanvuu-5124532.html",
        "title": "Hơn 4,1 triệu chữ ký yêu cầu đòi lại công bằng cho Himass và TanVuu",
        "publisher": "VnExpress",
        "language": "vi",
        "source_type": "independent",
        "authority_level": "secondary",
        "output_filename": "justice_for_himass_tanvuu_vi.json",
        "fallback_url": "",
        "fallback_language": "",
        "topic": ["justice_for_pubg_vn", "himass_tanvuu",
                  "kien_nghi_xem_xet_lai_an_phat", "cong_bang_minh_bach",
                  "phan_ung_cong_dong"],
    },
    {
        "id": "pubg_report_cheating_bug_abuse_vi",
        "url": "https://support.pubg.com/hc/vi/articles/360044488434",
        "title": "How to report cheating/hacking and bug abuse — PUBG Support",
        "publisher": "PUBG Support",
        "language": "vi",
        "source_type": "official_support",
        "authority_level": "primary",
        "output_filename": "pubg_report_cheating_bug_abuse_vi.json",
        "fallback_url": "https://support.pubg.com/hc/en-us/articles/360044488434-I-found-a-player-who-is-cheating-hacking-How-should-I-report-it",
        "fallback_language": "en",
        "topic": ["report_cheating", "report_hacking", "teamkiller",
                  "bug_abuse", "in_game_report", "anti_cheat"],
    },
    {
        "id": "pubg_account_cheating_responsibility_vi",
        "url": "https://support.pubg.com/hc/vi/articles/360002085694",
        "title": "If someone else uses cheats/hacks on my account, will I get banned? Can it be removed? — PUBG Support",
        "publisher": "PUBG Support",
        "language": "vi",
        "source_type": "official_support",
        "authority_level": "primary",
        "output_filename": "pubg_account_cheating_responsibility_vi.json",
        "fallback_url": "https://support.pubg.com/hc/en-us/articles/360002085694-If-someone-else-uses-cheats-hacks-on-my-account-will-I-get-banned-Can-it-be-removed",
        "fallback_language": "en",
        "topic": ["account_responsibility", "account_sharing", "cheating_ban",
                  "ban_irreversible", "account_security"],
    },
    {
        "id": "pubg_report_cheat_seller_vi",
        "url": "https://support.pubg.com/hc/vi/articles/115004167334",
        "title": "I want to report a site or user who is selling cheats/hacks — PUBG Support",
        "publisher": "PUBG Support",
        "language": "vi",
        "source_type": "official_support",
        "authority_level": "primary",
        "output_filename": "pubg_report_cheat_seller_vi.json",
        "fallback_url": "https://support.pubg.com/hc/en-us/articles/115004167334-I-want-to-report-a-site-or-user-who-is-selling-cheats-hacks",
        "fallback_language": "en",
        "topic": ["report_cheat_seller", "illegal_programs", "hack_vendors",
                  "fair_play", "anti_cheat"],
    },
    {
        "id": "pubg_ban_penalty_information_vi",
        "url": "https://support.pubg.com/hc/vi/articles/8463403858329",
        "title": "I want to know more about bans — PUBG Support",
        "publisher": "PUBG Support",
        "language": "vi",
        "source_type": "official_support",
        "authority_level": "primary",
        "output_filename": "pubg_ban_penalty_information_vi.json",
        "fallback_url": "https://support.pubg.com/hc/en-us/articles/8463403858329-I-want-to-know-more-about-bans",
        "fallback_language": "en",
        "topic": ["ban", "penalty_criteria", "misconduct", "permanent_ban",
                  "suspension_period", "rules_of_conduct"],
    },
]

# Tương thích skeleton cũ (danh sách URL công khai, thứ tự ổn định).
ARTICLE_URLS = [source["url"] for source in NEWS_SOURCES]


class CrawlError(RuntimeError):
    """Lỗi crawl rõ nguồn (không fake content)."""


def fetch_html(url: str, timeout_s: float = TIMEOUT_S,
               max_attempts: int = MAX_ATTEMPTS) -> str:
    """GET có retry/backoff; 403/block/timeout -> CrawlError rõ ràng."""
    import os

    last_error = "unknown error"
    session = requests.Session()
    session.headers.update({"User-Agent": os.getenv("TASK2_USER_AGENT",
                                                    USER_AGENT)})
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, timeout=timeout_s)
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 200 and response.text.strip():
                return response.text
            if response.status_code in (401, 403, 429):
                last_error = (f"HTTP {response.status_code} (blocked or "
                              f"rate-limited, no circumvention attempted)")
            else:
                last_error = f"HTTP {response.status_code}"
        if attempt < max_attempts:
            time.sleep(BACKOFF_S * attempt)
    raise CrawlError(f"Failed to fetch {url}: {last_error}")


def crawl_article(source: dict) -> dict:
    """Crawl một nguồn registry -> dict đúng schema JSON của repo."""
    attempts = [(source["url"], source["language"])]
    if source.get("fallback_url"):
        attempts.append((source["fallback_url"],
                         source.get("fallback_language", "en")))
    last_error = "no URL attempted"
    for url, language in attempts:
        try:
            html = fetch_html(url)
        except CrawlError as exc:
            last_error = str(exc)
            continue
        article = extract_article(html)
        body = "\n\n".join(article["blocks"]).strip()
        if len(body) < MIN_CONTENT_CHARS:
            last_error = (f"empty/invalid content from {url} "
                          f"({len(body)} chars)")
            continue
        return {
            "title": article["title"] or source["title"],
            "url": url,
            "publisher": source["publisher"],
            "language": language,
            "source_type": source["source_type"],
            "authority_level": source["authority_level"],
            "date_crawled": date.today().isoformat(),
            "topic": list(source.get("topic", [])),
            "content_markdown": body,
        }
    raise CrawlError(f"Crawl failed for {source['id']}: {last_error}")


def crawl_all(output_dir: Path | None = None) -> dict:
    """Crawl mọi nguồn; ghi một JSON mỗi nguồn thành công.

    Trả về ``{"saved": [...], "failed": {id: error}}`` — nguồn lỗi KHÔNG
    tạo file giả, artifact cũ được giữ nguyên.
    """
    target = Path(output_dir) if output_dir else DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    failed: dict[str, str] = {}
    for source in NEWS_SOURCES:
        try:
            article = crawl_article(source)
        except (CrawlError, OSError) as exc:
            failed[source["id"]] = str(exc)[:300]
            print(f"Failed: {source['id']} — {exc}")
            continue
        output = target / source["output_filename"]
        tmp = output.with_suffix(".json.part")
        tmp.write_text(json.dumps(article, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(output)
        saved.append(source["output_filename"])
        print(f"Saved: {output} ({len(article['content_markdown'])} chars, "
              f"lang={article['language']})")
    return {"saved": saved, "failed": failed}


if __name__ == "__main__":
    crawl_all()
