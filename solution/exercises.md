# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Với Gemini 3 Flash Preview, cả bốn mức temperature đều cho phản hồi mạch lạc và cùng xoay quanh lịch sử, hồ nước, phố cổ hoặc cà phê trứng của Hà Nội. Ở 0.0, câu trả lời có bố cục đều và thiên về liệt kê; ở 1.2 và 1.8, cách diễn đạt phong phú hơn và thêm nhiều chi tiết. Trong mẫu thử này tôi chưa thấy mức nào mất mạch lạc rõ rệt, nhưng 1.8 đưa ra các khẳng định nhiều hơn nên cần kiểm chứng thông tin kỹ hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Với trợ lý soạn thảo hợp đồng pháp lý, tôi chọn temperature khoảng 0.1–0.2 để câu chữ ổn định, nhất quán và hạn chế thêm diễn giải không được yêu cầu. Với trợ lý viết slogan quảng cáo, tôi chọn khoảng 0.8–1.2 để tạo ra nhiều phương án, hình ảnh và cách chơi chữ đa dạng. Hai sản phẩm ưu tiên khác nhau: pháp lý ưu tiên độ chính xác, còn quảng cáo cần sự mới lạ.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Workload là 20.000 × 2 × 500 = 20.000.000 output token mỗi ngày. Theo bảng giá hiện tại trong template cho Gemini 3 Flash Preview và Gemini 3.1 Flash-Lite, chi phí ước tính lần lượt là 20.000 × 0,003 = 60 USD/ngày và 20.000 × 0,0015 = 30 USD/ngày (đây là ước tính theo giá bảng, còn thí nghiệm đang dùng Free Tier). Model lớn xứng đáng khi cần suy luận hoặc xử lý nội dung khó; model nhỏ phù hợp cho phân loại, tóm tắt, FAQ hay khối lượng lớn câu hỏi đơn giản.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Persona nhà thơ trả lời bằng các hình ảnh như đứa trẻ, bông hoa, chim non và mầm cây; câu trả lời có nhịp điệu, không dùng cấu trúc kỹ thuật. Persona kỹ sư senior trả lời dài hơn, định nghĩa rõ ML, phân loại supervised/unsupervised/reinforcement learning, mô tả quy trình triển khai và có ví dụ Python với scikit-learn. Kết quả cho thấy system prompt điều khiển được giọng văn, độ dài, mức chi tiết, từ vựng chuyên môn, cách tổ chức và loại ví dụ mà model đưa ra.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Đoạn văn có 203 từ. Với `count_tokens(..., model="gemini-3-flash-preview")`, tiktoken không có encoding cho Gemini nên hàm dùng fallback và cho 234 token; ước lượng Part 1 là 203 / 0,75 ≈ 271 token, cao hơn khoảng 15,7%. Để so sánh bằng tiktoken thật, encoding gpt-4o cho 484 token, trong khi ước lượng thô vẫn là 271 token, tức dự toán thiếu khoảng 44,1%. Tiếng Việt có dấu và cách tách token không trùng với số từ, nên không nên dùng quy tắc từ/0,75 làm ngân sách cuối cùng.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Chatbot văn bản hưởng lợi nhiều từ streaming vì người dùng thấy câu trả lời xuất hiện ngay thay vì chờ toàn bộ phản hồi. Trợ lý giọng nói cũng hưởng lợi rất lớn: có thể bắt đầu đọc khi đã có các cụm từ đầu tiên, giảm cảm giác trễ trong hội thoại. Pipeline dịch tài liệu chạy ngầm ban đêm ít cần streaming nhất vì kết quả chỉ được dùng sau khi hoàn tất; ở đây độ tin cậy, khả năng retry và lưu kết quả quan trọng hơn phản hồi tức thì.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Nếu hàng nghìn client cùng retry bằng delay cố định, chúng dễ gửi lại request cùng lúc và tạo thêm một đợt quá tải mới. Exponential backoff làm khoảng cách retry tăng dần, giảm áp lực lên server và cho hệ thống thời gian phục hồi. Tuy vậy, nhiều client vẫn có thể đồng bộ ở các mốc 0,1; 0,2; 0,4 giây; jitter thêm một lượng ngẫu nhiên vào delay để phá sự đồng bộ này và tránh “retry storm”.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt tôi dùng là: “Bạn là trợ lý học tập thân thiện. Hãy trả lời ngắn gọn, rõ ràng bằng tiếng Việt; giải thích từng bước khi câu hỏi có nội dung kỹ thuật.” Nếu xóa cụm “ngắn gọn, rõ ràng bằng tiếng Việt”, câu trả lời có thể dài hơn hoặc không còn giữ ngôn ngữ mong muốn. Nếu xóa cụm “giải thích từng bước khi câu hỏi có nội dung kỹ thuật”, trợ lý có thể chỉ nêu kết luận mà không trình bày quy trình học hoặc cách làm.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Ví dụ, ở lượt đầu người dùng nói “Tôi đang xây ứng dụng dự báo giá nhà bằng dữ liệu căn hộ ở Hà Nội”, rồi sau bốn lượt chuyển sang hỏi “Nên xử lý cột diện tích lúc nãy thế nào?”. Khi lượt đầu đã bị cắt khỏi history, trợ lý có thể không biết “cột diện tích” thuộc bộ dữ liệu và bài toán nào. Một cách khắc phục là tóm tắt các lượt cũ thành một memory ngắn, giữ các yêu cầu và dữ kiện quan trọng, rồi gửi summary đó cùng bốn lượt gần nhất.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
