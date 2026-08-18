# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Chu Tuan Viet (cá nhân)
**Thành viên:** Chu Tuan Viet → M1–M5

---

## RAGAS Scores

Chạy `python main.py` với 20 câu hỏi trong `test_set.json` (corpus: 26 docs có text layer, 107 hierarchical children).

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8333 | 0.7242 | -0.1092 |
| Answer Relevancy | 0.2566 | 0.3451 | +0.0886 |
| Context Precision | 0.9250 | 0.9458 | +0.0208 |
| Context Recall | 0.9000 | 0.8250 | -0.0750 |

Nhận xét nhanh:
- **Context Precision tăng** (+0.02) — reranker (bge-reranker-v2-m3) lọc được context nhiễu tốt hơn dense-only baseline.
- **Answer Relevancy tăng mạnh** (+0.09) — enrichment contextual prepend + hybrid search giúp câu trả lời bám câu hỏi hơn.
- **Faithfulness & Context Recall giảm** — context sau rerank chỉ còn top-3, trong khi baseline lấy top-3 dense; hierarchical child (~196 chars) ngắn hơn basic chunk nên một số bằng chứng (dòng cuối bảng, ghi chú bảo hiểm) bị rơi ra ngoài top-3. Đây là trade-off precision/recall của việc thu hẹp context.

## Bottom-5 Failures

### #1
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16-30 ngày cần phê duyệt của Giám đốc điều hành (CEO). Lưu ý: nghỉ trên 14 ngày không lương, nhân viên phải tự đóng phần bảo hiểm của mình.
- **Got:** Cần phê duyệt của Giám đốc điều hành (CEO). (thiếu ghi chú tự đóng bảo hiểm khi nghỉ > 14 ngày)
- **Worst metric:** faithfulness (0.375)
- **Error Tree:** Output đúng một phần → Context đúng? ✅ (context [0] từ `nghi_phep_khong_luong.md` có đúng dòng "16-30 ngày → CEO") → Câu trả lời thiếu chi tiết ghi chú bảo hiểm, mà ghi chú đó nằm ở **chunk khác** (ngoài top-3) → LLM không có bằng chứng nên không thể trả lời đủ.
- **Root cause:** M1 hierarchical child quá ngắn tách ghi chú bảo hiểm ra chunk riêng; M2/M3 top-3 không mang đủ bằng chứng phụ.
- **Suggested fix:** Dùng parent-child retrieval: retrieve child → trả **parent** làm context (chunk lớn hơn, giữ nguyên ghi chú). Kiểm tra lại bằng test: query "nghỉ phép không lương > 14 ngày bảo hiểm" phải trả về chunk chứa "tự đóng".

### #2
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn 15 ngày. Quá hạn 5 ngày → phí 2%/tháng trên 15.000.000 = 300.000 VNĐ/tháng (pro-rata ~50.000 VNĐ cho 5 ngày).
- **Got:** "2% của 15 triệu = 300.000 VNĐ. Do đó nhân viên bị phạt 300.000 VNĐ." — đúng con số 300k/tháng nhưng kết luận "bị phạt 300.000" bỏ qua pro-rata, mâu thuẫn với GT.
- **Worst metric:** faithfulness (0.494)
- **Error Tree:** Output sai một phần → Context đúng? ✅ (cả 3 context đều từ `tam_ung.md`, có "2%/tháng") → Prompt gốc "Trả lời CHỈ dựa trên context" không ép LLM tính pro-rata theo số ngày quá hạn → LLM chốt 300k trọn tháng.
- **Root cause:** M4 generation: thiếu instruction "tính theo số ngày thực quá hạn"; context cũng thiếu ví dụ pro-rata.
- **Suggested fix:** Bổ sung prompt: "Nếu policy tính phí theo tháng, nêu cả mức/tháng và số tiền pro-rata theo số ngày quá hạn." Verify bằng eval lại 1 câu này.

### #3
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30M trong khoảng 5-50M → Giám đốc phòng ban (Director) phê duyệt + xác nhận cấu hình CNTT + 3 báo giá.
- **Got:** "cần sự phê duyệt của Kế toán trưởng vì số tiền này lớn hơn 5.000.000 VNĐ" — **sai** người phê duyệt.
- **Worst metric:** answer_relevancy (0.500)
- **Error Tree:** Output sai → Context sai? ⚠️ Context [1] là chunk `tam_ung.md` ("Tạm ứng ≥ 5M: cần thêm...") thay vì bảng thẩm quyền mua sắm → LLM nhầm policy tạm ứng với policy mua sắm. Bảng "5-50M → Director" nằm trong `mua_sam.md` nhưng không vào top-3.
- **Root cause:** M2/M3 retrieval nhầm policy: "30 triệu + phê duyệt" match cả tạm ứng lẫn mua sắm; reranker không phân biệt được category; metadata `category` chưa được dùng làm filter.
- **Suggested fix:** Thêm metadata filter theo source/category (mua_sam vs tam_ung) hoặc đưa category vào contextual prepend để reranker phân biệt. Test: query mua sắm không được trả về chunk tạm ứng trong top-3.

### #4
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** "Cần Tổng Giám đốc (CEO) phê duyệt." — đúng.
- **Worst metric:** faithfulness (0.541)
- **Error Tree:** Output đúng → Context đủ? ⚠️ Context [0] là chunk chứa **đầu bảng** thẩm quyền (Dưới 5M → Manager; 5-50M → ...) nhưng dòng "Trên 50M → CEO" bị **chặt sang chunk kế tiếp** → RAGAS không thấy bằng chứng ">50M → CEO" trong context nên chấm faithfulness thấp.
- **Root cause:** M1 hierarchical split **cắt ngang bảng markdown** — child 256 chars chặt giữa table row.
- **Suggested fix:** Chunking structure-aware cho tài liệu có bảng: không cắt giữa table block (giữ nguyên cả bảng trong 1 chunk), hoặc cắt theo dòng header. Test: "55 triệu" phải trả về chunk chứa cả dòng "Trên 50.000.000 VNĐ → CEO".

### #5
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** v2024: 15 + 3 (9÷3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** "18 ngày phép năm (15 + 3). Số ngày phép này có lương." — đúng phép năm, **thiếu** lương Senior.
- **Worst metric:** answer_relevancy (0.562)
- **Error Tree:** Output sai một phần → Context đúng một phần: context [0] v2024 có ví dụ "9 năm → 18 ngày" ✅, nhưng context [1] là `nghi_phep_nam_v2023.md` (**version cũ 12 ngày**) và không có chunk lương nào trong top-3 → câu trả lời thiếu mảng lương.
- **Root cause:** M2 retrieval thiếu chunk lương (câu hỏi 2 ý: phép năm + lương, chỉ 1 ý được cover); version cũ v2023 vẫn lọt vào context gây nhiễu.
- **Suggested fix:** M5 auto-metadata gắn `version` vào chunk, M2 dùng version filter khi query; mở rộng top_k trước rerank (20 → 30) để câu hỏi multi-hop có đủ 2 ý trong ứng viên. Test: query "phép năm và lương" phải có cả chunk v2024 lẫn chunk lương trong top-3.

## Case Study (cho presentation)

**Question chọn phân tích:** "Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?"

**Error Tree walkthrough:**
1. Output đúng? → ❌ Trả lời "Kế toán trưởng" — sai, phải là Giám đốc phòng ban.
2. Context đúng? → ⚠️ Context [1] là chunk `tam_ung.md` (policy tạm ứng ≥5M) — đúng từ khóa "30 triệu + phê duyệt" nhưng **sai policy**. Bảng thẩm quyền mua sắm nằm ở `mua_sam.md` không vào top-3.
3. Query rewrite OK? → Không có query rewrite; retrieval dựa raw query.
4. Fix ở bước: **M2/M3** — thêm metadata `category` filter (mua_sam vs tam_ung) + contextual prepend nhấn mạnh policy; nếu không, reranker đọc cả 2 chunk thì vẫn cần prompt nhắc "chỉ dùng policy mua sắm".

**Nếu có thêm 1 giờ, sẽ optimize:**
- Retrieval parent-child: retrieve child → return parent context (fix #1, #4).
- Metadata version filter (fix #5) + category filter (fix #3).
- Prompt: instruction tính pro-rata + "nếu hỏi nhiều ý, trả lời đủ từng ý" (fix #2).
- OCR BCTC.pdf + Nghị định 13 để tăng corpus (2 PDF scan hiện bị bỏ qua).
