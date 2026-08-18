# Group Report — Lab 18: Production RAG

**Nhóm:** Chu Tuan Viet (bài cá nhân)
**Ngày:** 2026-08-18

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Chu Tuan Viet | M1: Chunking | ✅ | 13/13 |
| Chu Tuan Viet | M2: Hybrid Search | ✅ | 5/5 |
| Chu Tuan Viet | M3: Reranking | ✅ | 5/5 |
| Chu Tuan Viet | M4: Evaluation | ✅ | 4/4 |
| Chu Tuan Viet | M5: Enrichment | ✅ | 10/10 |

**Tổng: 37/37 tests pass.**

## Kết quả RAGAS

Chạy `python main.py` trên 20 câu test, corpus 26 docs (107 hierarchical children), LLM qua OpenRouter (gpt-4o-mini), embedding bge-m3 (dense) + all-MiniLM-L6-v2 (eval).

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.8333 | 0.7242 | -0.1092 |
| Answer Relevancy | 0.2566 | 0.3451 | +0.0886 |
| Context Precision | 0.9250 | 0.9458 | +0.0208 |
| Context Recall | 0.9000 | 0.8250 | -0.0750 |

## Key Findings

1. **Biggest improvement:** Answer Relevancy +0.089 — hybrid search (BM25 + dense + RRF) + reranker (bge-reranker-v2-m3) + enrichment contextual prepend giúp câu trả lời bám sát câu hỏi; context precision cũng tăng nhẹ.
2. **Biggest challenge:** Faithfulness (-0.11) và context recall (-0.08) giảm — hierarchical child ngắn (196 chars) làm bằng chứng bị tách rời khỏi top-3 (bảng mua sắm bị cắt, ghi chú bảo hiểm ở chunk khác, version v2023 cũ vẫn lọt vào).
3. **Surprise finding:** 2 PDF scan (BCTC, Nghị định 13) không có text layer bị loại đúng theo boundary của repo — nhưng "phiên bản cũ v2023" vẫn được retrieve và gây nhiễu cho câu hỏi về v2024; metadata `version` chưa được dùng làm filter là lỗ hổng thực sự.

## Presentation Notes (5 phút)

1. RAGAS scores (naive vs production): precision +0.02, relevancy +0.09, nhưng faithfulness/recall giảm nhẹ — kể cả 2 hướng.
2. Biggest win — M3 reranking + M5 contextual prepend: context chính xác hơn → relevancy tăng; reranker benchmark (CrossEncoder): latency đo được trong lần chạy thật.
3. Case study — failure #3 (laptop 30 triệu): LLM trả lời "Kế toán trưởng" vì context [1] là chunk policy tạm ứng thay vì bảng thẩm quyền mua sắm; Error Tree: output sai → context sai policy → fix bằng metadata category filter.
4. Next optimization nếu có thêm 1 giờ: parent-child retrieval (return parent), metadata version/category filter, prompt pro-rata + multi-ý, OCR 2 PDF scan.
