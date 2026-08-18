# Individual Reflection — Lab 18

**Tên:** Chu Tuan Viet
**Module phụ trách:** M1–M5 (bài cá nhân)

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M1 Chunking, M2 Hybrid Search, M3 Reranking, M4 RAGAS Eval, M5 Enrichment
- Các hàm/class chính đã viết:
  - M1: `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()`
  - M2: `segment_vietnamese()`, `BM25Search.index/search()`, `DenseSearch.index/search()`, `reciprocal_rank_fusion()`
  - M3: `CrossEncoderReranker._load_model()/rerank()`, `FlashrankReranker.rerank()`
  - M4: `evaluate_ragas()` (config LLM qua OpenRouter + embeddings local), `failure_analysis()`
  - M5: `summarize_chunk()`, `generate_hypothesis_questions()`, `contextual_prepend()`, `extract_metadata()`, `_enrich_single_call()` (combined mode), `_extract_json()`
- Số tests pass: 37/37

## 2. Kiến thức học được

- Khái niệm mới nhất: **RRF (Reciprocal Rank Fusion)** — gộp BM25 (thang điểm IDF khác) và cosine (thang 0–1) bằng rank thay vì score; **contextual prepend** kiểu Anthropic giảm retrieval failure; **parent-child retrieval** (retrieve child precision, return parent context).
- Điều bất ngờ nhất: Corpus chỉ ~21K chars → 107 chunks, nhưng lỗi thật sự nằm ở chỗ **bảng markdown bị hierarchical chunk cắt ngang giữa table row** ("Trên 50M → CEO" rơi ra khỏi top-3) — lỗi chunking nhỏ gây sai cả câu trả lời về thẩm quyền phê duyệt.
- Kết nối với bài giảng: "Semantic chunking" (slide M1) ↔ `chunk_semantic()` — threshold 0.85 trên corpus thật; "BM25 + Dense fusion" ↔ `reciprocal_rank_fusion()`; "Cross-encoder reranking" ↔ `CrossEncoderReranker.rerank()`; "RAGAS 4 metrics" ↔ `evaluate_ragas()`; "Contextual embeddings" ↔ `contextual_prepend()`.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất 1: **Windows cp1252** — `UnicodeEncodeError: 'charmap' codec can't encode character '\u1ee9'` khi `print()` tiếng Việt. Cách giải quyết: thêm `sys.stdout.reconfigure(encoding="utf-8")` trong `config.py` (import ở mọi module) → main.py chạy ổn định.
- Khó khăn lớn nhất 2: **RAGAS + OpenRouter** — RAGAS 0.1.x mặc định dùng `gpt-3.5-turbo` (không có trên OpenRouter) và `OpenAIEmbeddings` (cần API riêng). Cách giải quyết: inject `llm=LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))` + `embeddings=LangchainEmbeddingsWrapper(HuggingFaceEmbeddings("all-MiniLM-L6-v2"))` vào `evaluate()`; đọc source ragas (`llms/base.py`, `embeddings/base.py`) để biết API.
- Khó khăn lớn nhất 3: **underthesea nối từ ghép bằng "_"** (`nghỉ_phép`) làm BM25 không khớp query — scaffold đã cảnh báo, fix bằng `.replace("_", " ")` sau `word_tokenize`.
- Thời gian debug: ~30–40 phút (encoding + RAGAS config + bảng bị cắt).

## 4. Nếu làm lại

- Sẽ làm khác điều gì: chọn **structure-aware + hierarchical kết hợp** cho docs có bảng thay vì hierarchical thuần (bảng phải giữ nguyên 1 chunk); thêm metadata `version`/`category` filter từ đầu thay vì chỉ để reranker tự xử lý.
- Module nào muốn thử tiếp: OCR pipeline cho PDF scan (BCTC, Nghị định 13) — 2/28 docs hiện bị bỏ; và thử `FlashrankReranker` (<5ms) so với bge-reranker-v2-m3 để so latency.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | — (cá nhân) |
| Problem solving | 4 |

## 6. Mapping lecture → code

| Lecture Concept | Module | Hàm cụ thể | Observation |
|----------------|--------|-------------|-------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85: nhóm câu theo cosine; trên corpus 26 docs tạo ít chunk hơn basic, không cắt giữa ý |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF giải quyết việc trộn 2 thang điểm khác nhau; doc xuất hiện ở cả 2 list được cộng dồn 1/(k+rank) |
| Vietnamese tokenization | M2 | `segment_vietnamese()` | underthesea `_` → space, nếu không BM25 "nghỉ_phép" vs "nghỉ phép" không khớp |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | top-20 → top-3; latency đo được từ benchmark thật (CrossEncoder ~ vài trăm ms/lần predict batch) |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Metric thấp nhất: faithfulness 0.72 — bằng chứng bị tách khỏi top-3 chứ không phải LLM bịa |
| Contextual embeddings | M5 | `contextual_prepend()` | Prepend "Trích từ <source>" + context LLM giúp reranker phân biệt policy; giữ nguyên source metadata |

## 7. Action Plan cho project

## Project: RAG hỏi đáp chính sách nội bộ công ty (nhiều phiên bản policy, tiếng Việt)

### Hiện tại
- RAG pipeline hiện tại: chunk theo paragraph + dense-only (naive), không có version handling.
- Known issues: câu hỏi về policy mới bị trả lời theo bản cũ; câu hỏi multi-ý (phép năm + lương) trả lời thiếu ý; bảng quyết định bị cắt.

### Plan áp dụng
1. [x] Chunking strategy: structure-aware cho docs có bảng/heading + hierarchical parent-child (retrieve child, return parent) — giữ nguyên table block.
2. [x] Search: Hybrid BM25 + Dense + RRF (BM25 bắt từ khóa tiếng Việt, dense bắt ý) — đã implement, giữ nguyên.
3. [ ] Reranking: bge-reranker-v2-m3 (production); đánh giá Flashrank nếu cần <5ms.
4. [ ] Evaluation: RAGAS 4 metrics trên test set riêng; thêm assert per-category (lookup/version/negation/multi-hop) để bắt regression.
5. [ ] Enrichment: combined single-call (1 call/chunk) + auto-metadata `version`/`category` làm filter khi query.

### Timeline
- Tuần 1: parent-child retrieval + version metadata filter (fix faithfulness/recall).
- Tuần 2: category filter cho mua sắm vs tạm ứng; prompt pro-rata + multi-ý.
- Tuần 3: OCR 2 PDF scan; benchmark reranker latency; viết eval per-category.
