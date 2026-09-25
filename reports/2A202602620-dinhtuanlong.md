# Individual contribution report

---

## Thông tin

- Họ và tên: Đinh Tuấn Long
- Mã học viên: 2A202602620
- Nhóm: BLS
- Repository/branch: https://github.com/brxcewxyne/K4-L3B-RAG-Pipeline.git (`main`)

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Data — Task 1 legal collection | Viết downloader thật: registry 3 chính sách VI, fetch có retry/backoff, trích article, dựng lại PDF bằng fpdf2 (giữ nguyên văn, skip file đã hợp lệ, lỗi nêu tên nguồn, không fake) | `src/task1_collect_legal_docs.py` | Done |
| Data — Task 2 news crawling | Viết crawler thật cho 6 nguồn: thử URL tiếng Việt trước, fallback tiếng Anh đã document khi bị 403, metadata `language` ghi đúng thực tế, giữ `independent`/`secondary` cho nguồn Justice | `src/task2_crawl_news.py` | Done |
| Data — HTML extraction dùng chung | Parser `html.parser` chuẩn (không thêm dependency): chọn `<article>` dài nhất, loại chrome, giữ heading/bullet, đã live-validate trên pubg.com/VnExpress | `src/html_extract.py` | Done |
| Data — layout `news/` + provenance | Chuyển 6 JSON `landing/web/` → `landing/news/` theo đúng contract test; đồng bộ `source_manifest.json`, frontmatter `source_file`, metadata Chroma (count giữ 274, không re-embed) | data + manifest | Done |
| Index & Search — Task 3/4 | Chuẩn hoá Markdown (MarkItDown + frontmatter); chunking recursive 500/50 với ID ổn định; embedding ngoài `text-embedding-3-small`; ChromaDB `rag_documents` cosine, upsert idempotent | `src/task3_convert_markdown.py`, `src/task4_chunking_indexing.py` | Done |
| Index & Search — Task 5/6 | Dense retrieval tái dùng đúng `embed_texts` (cùng embedding space, score `1 − distance`); BM25 lexical riêng biệt, cùng schema `SearchResult` | `src/task5_semantic_search.py`, `src/task6_lexical_search.py` | Done |
| Generation & UI — Task 10 + Streamlit | Generation context-only qua OpenAI có kiểm tra citation và safe refusal; UI chat 3 cột + tab Retrieval Flow (Dense/BM25/RRF/panel fallback) | `src/task10_generation.py`, `app.py`, `src/ui_components.py` | Done |
| Tích hợp/kiểm thử chung | Sửa `format_context` thiếu `source` theo contract test; giữ toàn suite xanh | test liên quan | Done |

Chỉ kê khai công việc có thể đối chiếu bằng file, commit, pull request, test hoặc kết quả evaluation.

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Metadata ngôn ngữ/nguồn trung thực tuyệt đối: crawl thử bản tiếng Việt trước, chỉ dùng fallback tiếng Anh khi bị chặn, và `language` luôn phản ánh nội dung thật (không gán nhãn Việt cho nội dung Anh); nguồn Justice giữ `independent`/`secondary`.
   **Lý do/evidence:** Corpus Việt-là-chính nhưng 4 trang hỗ trợ chỉ lấy được bản Anh (HTTP 403 cả hai locale); test `test_english_fallback_metadata` và acceptance kiểm tra đúng phân loại này.
   **Trade-off:** Pipeline phức tạp hơn (hai URL + ngôn ngữ mỗi nguồn) và live-run Task 2 chỉ đạt 2–3/6 nguồn; bù lại provenance sạch cho RAG source-aware.

2. **Quyết định:** Mọi tầng ghi dữ liệu đều idempotent bằng ID ổn định (chunk `{doc}_chunk-{index}`, upsert Chroma, registry PageIndex theo content-hash, skip file PDF đã hợp lệ).
   **Lý do/evidence:** Chạy lại pipeline không tốn API call trùng, không trùng bản ghi (Chroma giữ nguyên count 274 qua nhiều lần chạy).
   **Trade-off:** Thêm logic skip/hash và kiểm thử cho từng tầng, đổi lại demo và chấm bài có thể chạy lại an toàn.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `pytest -q` toàn repo (87 passed); `tests/test_task1.py` + `tests/test_task2.py` (18 test, mock network); live-validate Task 1 (force vào thư mục tạm, so tiêu đề/dấu tiếng Việt với PDF gốc) và Task 2 (so metadata + word-overlap với artifact cũ); query kiểm thử retrieval/generation trên UI trước khi chốt.
- Kết quả trước/sau nếu có: 2 lỗi contract tồn tại trước phần việc của tôi (`landing/news` thiếu JSON, `format_context` thiếu `source`) → sau khi sửa và migrate layout, suite đạt 87 passed / 0 failed.
- Lỗi đã phát hiện và cách xử lý: (1) extractor chọn nhầm `<article>` teaser → chuyển sang chọn `<article>` dài nhất; (2) font Arial thiếu glyph số tròn ①②③ → map sang `(1)–(20)` để không mất nội dung; (3) cắt scope làm mất `<title>` → trích title từ HTML gốc trước khi scope.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: PDF tái dựng từ HTML đơn giản hoá layout (bảng penalty thành dòng text, phụ thuộc font Unicode của máy chạy — cần Arial/DejaVu), nên khác byte với PDF gốc dù nội dung tương đương.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: Bundled một font open-source vào repo và giữ cấu trúc bảng gốc trong PDF tái dựng để pipeline thu thập → chuẩn hoá hoàn toàn tái lập được byte-ổn định.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 26/09/2026
- Tên thành viên: Đinh Tuấn Long
