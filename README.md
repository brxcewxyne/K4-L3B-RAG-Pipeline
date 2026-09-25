# Day 8 — RAG Pipeline · Nhóm BLS

## Nhóm thực hiện

| Họ và tên | Mã học viên |
|---|---|
| Đinh Tuấn Long | 2A202602620 |
| Lê Duy Bảo | 2A202602749 |
| Trần Quốc Sáng | 2A202602712 |

## Mục tiêu

Nhóm BLS xây dựng chatbot RAG **PUBG Policy & Player Support Assistant**: trả lời câu hỏi dựa trên (grounded) bộ tài liệu PUBG do nhóm tự thu thập, có hybrid retrieval, citation, giao diện chat và báo cáo đánh giá.

Sản phẩm phải có: repository chạy được, tối thiểu 3 tài liệu chính sách và 5 bài viết/page tự thu thập, pipeline convert → chunk → index → dense + BM25 → RRF → fallback → generation có citation, chatbot Streamlit hiển thị câu trả lời và nguồn đã dùng, golden dataset tối thiểu 15 câu, đánh giá 4 metric và so sánh A/B, `group_project/evaluation/RESULT.md`, và báo cáo cá nhân mỗi thành viên trong `reports/` theo template `reports/INDIVIDUAL_REPORT.md`.

## Dữ liệu

Corpus tiếng Việt là chính (một số trang hỗ trợ PUBG chỉ tồn tại bản tiếng Anh nên dùng bản gốc tiếng Anh, metadata `language` ghi đúng thực tế):

- `data/landing/legal/`: **3 PDF** chính sách (Quy tắc ứng xử, Điều khoản Dịch vụ, Chính sách Bảo mật PUBG bản tiếng Việt).
- `data/landing/news/`: **6 JSON** (điều tra PUBG Asia Stars Himass/TanVuu, bài độc lập Justice for PUBG VN, 4 trang hỗ trợ: báo cáo gian lận/bug, trách nhiệm tài khoản, báo cáo bán cheat, thông tin ban/án phạt) + `source_manifest.json` giữ provenance (URL gốc, publisher, ngôn ngữ, official/primary vs independent/secondary).
- `data/standardized/`: **9 file Markdown** (3 `legal/` + 6 `news/`) kèm frontmatter, là đầu vào duy nhất của khâu index.

## Kiến trúc pipeline

```text
Raw Sources
    ↓
Collection (Task 1: tải PDF chính sách · Task 2: crawl JSON tin/bài hỗ trợ)
    ↓
Standardized Markdown (Task 3: MarkItDown + frontmatter provenance)
    ↓
Chunking (500 chars, overlap 50, ID ổn định)
    ↓
Embedding (OpenAI text-embedding-3-small) + ChromaDB (collection rag_documents, 274 chunks, cosine)
    ↓
Query
 ├─ Dense (ChromaDB cosine similarity)
 └─ BM25 (từ khóa)
      ↓
     RRF (k=60, gộp thứ hạng, chạy một lần)
      ↓
Dense confidence gate (điểm cosine gốc của Dense top-1, KHÔNG dùng điểm RRF)
   ├─ strong (>= SCORE_THRESHOLD) → Hybrid Top-K
   └─ weak → PageIndex fallback (vectorless: điều hướng cây tài liệu, không phải retriever thứ ba)
                    ↓
                 Context
                    ↓
              Generation (OpenAI, citation [1], [2]… + safe refusal khi thiếu bằng chứng)
                    ↓
           Answer + Citations
                    ↓
             Streamlit UI (tab Trò chuyện + tab Luồng truy xuất)
```

## Cài đặt (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
copy .env.example .env
```

Điền API key cần dùng trong `.env`; không commit file này. Yêu cầu Python 3.10–3.13 theo `pyproject.toml`. Không cần `playwright install chromium`: code collection trong repo dùng HTTP trực tiếp (`requests` + phân tích HTML bằng thư viện chuẩn), không dùng Crawl4AI/browser.

## Biến môi trường

Các biến thực tế trong `.env.example` (không commit giá trị thật):

| Biến | Dùng cho |
|---|---|
| `OPENAI_API_KEY` | Embedding, generation, LLM judge |
| `LLM_PROVIDER` / `LLM_MODEL` | Generation (mặc định `openai` / `gpt-4.1-mini`) |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` / `EMBEDDING_API_BASE` | Embedding ngoài (mặc định `openai` / `text-embedding-3-small`) |
| `JUDGE_MODEL` | Model chấm evaluation (mặc định theo `LLM_MODEL`) |
| `PAGEINDEX_API_KEY`, `PAGEINDEX_READY_TIMEOUT_S`, `PAGEINDEX_READY_POLL_S`, `PAGEINDEX_SEARCH_TIMEOUT_S`, `PAGEINDEX_SEARCH_POLL_S` | PageIndex fallback (tùy chọn) |
| `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `JINA_API_KEY` | Provider dự phòng/khác |
| `SCORE_THRESHOLD` | Ngưỡng fallback (trống = dùng giá trị đã hiệu chỉnh trong code: `0.5728`) |

## Thu thập và chuẩn hoá dữ liệu

```powershell
.venv\Scripts\python.exe -m src.task1_collect_legal_docs   # tải 3 PDF chính sách (bỏ qua file đã hợp lệ)
.venv\Scripts\python.exe -m src.task2_crawl_news           # crawl 6 JSON vào data/landing/news/
.venv\Scripts\python.exe -m src.task3_convert_markdown     # chuẩn hoá sang data/standardized/
```

Mỗi nguồn một file, metadata/provenance giữ nguyên (kể cả `language: en` cho trang hỗ trợ chỉ có bản tiếng Anh và `independent`/`secondary` cho nguồn Justice). Lưu ý thực tế: một số trang `support.pubg.com` có thể trả HTTP 403 khi crawl tự động; code báo lỗi rõ từng nguồn, không vượt kiểm soát truy cập và không ghi nội dung giả — artifact đã thu thập trong repo vẫn được giữ.

## Index và kiểm tra contract

```powershell
.venv\Scripts\python.exe -m src.task4_chunking_indexing   # chunk → embed → upsert ChromaDB (idempotent)
.venv\Scripts\python.exe -m pytest -q
```

## Chạy chatbot

```powershell
.venv\Scripts\python.exe -m streamlit run app.py --server.port=8501
# (viết gọn khi đã activate venv: streamlit run app.py)
```

- Tab **Trò chuyện**: hỏi đáp có citation, lọc nguồn Official/Independent.
- Tab **Luồng truy xuất**: quan sát kỹ thuật Query → Dense + BM25 → RRF → Context (và panel PageIndex fallback khi hybrid confidence yếu).

## Đánh giá

- Config A — Dense only: `task5.semantic_search → top_k=5`.
- Config B — Hybrid + RRF: `task5` + `task6` → `task7.rerank_rrf(top_k=5, k=60)`.
- Metric: Context Recall, Context Precision, Faithfulness, Answer Relevance — chấm bởi LLM judge `gpt-4.1-mini` (temperature 0, JSON), cùng judge cho cả hai config; golden 16 câu tại `group_project/evaluation/golden_dataset.json`.
- Chạy đánh giá: `python group_project/evaluation/evaluate.py` (kết quả vào `evaluation_results.json`).
- Hiệu chỉnh ngưỡng fallback: `python group_project/evaluation/calibrate_threshold.py` — 16 câu in-domain + 16 câu out-of-domain, chỉ dùng điểm cosine gốc của Dense (xem chi tiết ở `threshold_calibration.json`).

Kết quả hiện tại (verbatim từ `evaluation_results.json`):

| Metric              | Dense A | Hybrid B | Delta   |
| ------------------- | ------: | -------: | ------: |
| Faithfulness        |  0.9792 |   1.0000 | +0.0208 |
| Answer Relevance    |  0.6750 |   0.9000 | +0.2250 |
| Context Recall      |  0.9531 |   0.9688 | +0.0157 |
| Context Precision   |  0.4375 |   0.5500 | +0.1125 |
| Average             |  0.7612 |   0.8797 | +0.1185 |

Chi tiết xem `group_project/evaluation/RESULT.md` (bản đầy đủ) và `reports/RESULT.md`.

## Hiệu chỉnh ngưỡng fallback

- Placeholder cũ: `0.3` (chưa đo). Ngưỡng đã hiệu chỉnh: **`0.5728`** (midpoint của khoảng phân tách, max balanced accuracy).
- In-domain (16 câu golden): 0.6028 – 0.7973. Out-of-domain (16 câu probes): 0.2510 – 0.5428.
- Phân loại trên mẫu hiệu chỉnh: TP = 16, FN = 0, TN = 16, FP = 0.
- Confidence dùng điểm cosine gốc của Dense top-1, không bao giờ dùng điểm RRF; ngưỡng này đặc thù cho corpus/mẫu hiện tại, tách biệt hoàn toàn trên mẫu đo.

## Kiểm tra

```powershell
.venv\Scripts\python.exe -m pytest tests/test_contracts.py -q
.venv\Scripts\python.exe -m pytest tests/test_acceptance.py -q
.venv\Scripts\python.exe -m pytest -q
```

Trạng thái hiện tại: **87 passed, 0 failed**.

## Phân công đóng góp

| Milestone | Owner |
|---|---|
| Setup | All members |
| Data (Task 1–3) | Đinh Tuấn Long |
| Index & Search (chunking, embedding, ChromaDB, Dense, BM25) | Đinh Tuấn Long |
| Fusion & Fallback (RRF, PageIndex, orchestration) | Trần Quốc Sáng |
| Generation & UI (Task 10, citation, Streamlit, Retrieval Flow UI) | Đinh Tuấn Long |
| Evaluation (golden, 4 metric, A/B, RESULT) | Lê Duy Bảo |
| Demo & Handoff | All members |

## Quy tắc code quality

- Dense và BM25 cùng trả về `SearchResult` theo một schema (`docs/MODULE_CONTRACTS.md`).
- RRF chỉ gộp thứ hạng và chỉ chạy một lần.
- Fallback dùng cosine score gốc của dense retrieval.
- Threshold hiệu chỉnh trên query in-domain và out-of-domain, không có một con số đúng cho mọi corpus.

## Hạn chế đã biết

- Kiểm chứng PageIndex live qua API chưa thực hiện vì chưa cấu hình `PAGEINDEX_API_KEY`; code + unit test (mock) đã hoàn chỉnh, PageIndex là tùy chọn khi reproduce.
- Một số trang hỗ trợ PUBG có thể trả HTTP 403 khi crawl lại; corpus trong repo đã đầy đủ.
- Ngưỡng fallback `0.5728` hiệu chỉnh riêng cho corpus/mẫu hiện tại; đổi corpus, chunking hoặc embedding model thì cần hiệu chỉnh lại.
- Đánh giá semantic cần `OPENAI_API_KEY` (judge + generation).

## Tái tạo từ đầu (Windows)

1. Tạo/activate venv, cài project, copy `.env.example` → `.env` và điền key.
2. Thu thập: `task1_collect_legal_docs`, `task2_crawl_news`.
3. Chuẩn hoá: `task3_convert_markdown`.
4. Index: `task4_chunking_indexing` (upsert idempotent vào `chroma_db/`).
5. Kiểm tra: `pytest -q` (kỳ vọng 87 passed).
6. Chatbot: `streamlit run app.py`.
7. Đánh giá: `python group_project/evaluation/evaluate.py` (+ hiệu chỉnh ngưỡng: `calibrate_threshold.py`).

## Lộ trình 3 giờ (tham khảo lab)

| Mốc                  | Thời gian | Kết quả cần có                           |
| -------------------- | --------: | ---------------------------------------- |
| 0. Setup             |   10 phút | Môi trường và `.env` sẵn sàng            |
| 1. Data              |   25 phút | ≥3 legal, ≥5 news, Markdown đã chuẩn hoá |
| 2. Index & search    |   30 phút | ChromaDB, dense search và BM25 chạy được |
| 3. Fusion & fallback |   25 phút | RRF và fallback tuân thủ contract        |
| 4. Generation & UI   |   30 phút | Chatbot trả lời có citation              |
| 5. Evaluation        |   30 phút | 15+ Q&A, 4 metric, A/B comparison        |
| 6. Demo & handoff    |   30 phút | Test, report, demo và push repository    |

## Tài liệu

- [Module contracts](docs/MODULE_CONTRACTS.md): schema, interface và invariant mà code/test tuân theo.
- [Step-by-step guide](docs/STEP_BY_STEP.md): thứ tự triển khai và tiêu chí hoàn thành từng bước.
- [Grading rubric](docs/GRADING_RUBRIC.md): Rubric thang điểm.
- [Individual report](reports/INDIVIDUAL_REPORT.md): template báo cáo cá nhân (`reports/<student-id>-<short-name>.md`).
- [Suggested topics](docs/SUGGESTED_TOPICS.md): danh sách chủ đề tham khảo, không bắt buộc.
