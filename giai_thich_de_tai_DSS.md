# Giải thích đề tài và chương trình (dành cho cả nhóm)

Đọc file này trước khi nhận việc. Mục đích: ai trong nhóm cũng hiểu **dự án làm gì, vì sao làm, các phần ghép với nhau thế nào, và thầy cô sẽ chấm điều gì**.

## 1. Tóm tắt trong một đoạn

Chúng ta xây một **hệ thống hỗ trợ quyết định (DSS)** cho người dùng điện thoại muốn ngủ tốt hơn. Người dùng nhập thói quen dùng điện thoại buổi tối. Hệ thống dùng **học máy** để ước lượng nguy cơ bị "nợ ngủ", rồi đưa ra **khuyến nghị cụ thể** (giảm bao nhiêu phút, bật bộ lọc ánh sáng xanh, đổi loại ứng dụng...) và cho thử **"nếu tôi làm vậy thì nguy cơ giảm còn bao nhiêu"**. Sản phẩm cuối là một ứng dụng web chạy bằng Python (Streamlit), kèm báo cáo và slide.

**Tên đề tài:** Ứng dụng học máy trong hệ thống hỗ trợ quyết định quản lý thời gian sử dụng điện thoại nhằm giảm thiếu hụt giấc ngủ.

## 2. Vấn đề mà đề tài giải quyết

- Dùng điện thoại sát giờ ngủ (mạng xã hội, video, game, nhắn tin) làm chậm giờ ngủ và giảm chất lượng giấc ngủ. Ngủ thiếu kéo dài gọi là **nợ ngủ (sleep debt)**, dẫn tới mệt mỏi, khó tập trung.
- Người dùng thường **biết** mình dùng máy khuya nhưng **không biết** nên đổi cái gì và đổi bao nhiêu thì có tác dụng. Lời khuyên chung chung ("đừng dùng điện thoại trước khi ngủ") ít hiệu quả vì không cá nhân hóa và không có con số.
- Đề tài biến dữ liệu thành lời khuyên **cá nhân hóa và định lượng được**.

## 3. Vì sao đây là DSS chứ không chỉ là một mô hình học máy

Môn học chấm **phần hỗ trợ ra quyết định**, không chỉ độ chính xác của mô hình. Đối chiếu các thành phần:

| Thành phần DSS | Trong dự án của chúng ta |
|---|---|
| Người ra quyết định | Người dùng điện thoại (sinh viên, nhân viên văn phòng) muốn cải thiện giấc ngủ |
| Quyết định cần đưa ra | Giảm bao nhiêu phút dùng máy trước khi ngủ, bật bộ lọc ánh sáng xanh không, hạ độ sáng, đổi loại ứng dụng, giảm caffeine |
| Cơ sở dữ liệu | Dataset Kaggle *Sleep Debt & Screen Time: Late Night Phone Habits* (8.500 dòng, 18 cột) |
| Cơ sở mô hình | Mô hình học máy dự đoán nguy cơ nợ ngủ (lưu thành `model.joblib`) |
| Giao diện người dùng | Ứng dụng Streamlit |
| Lớp quyết định | Phân mức nguy cơ, luật khuyến nghị, mô phỏng what-if, xếp hạng phương án |

Nếu thầy cô hỏi "đâu là phần hỗ trợ ra quyết định?" thì câu trả lời là **lớp quyết định** ở dòng cuối: nó biến con số dự đoán thành hành động người dùng có thể chọn.

## 4. Chương trình hoạt động thế nào

```
Người dùng nhập thói quen
        │
        ▼
  Tiền xử lý (preprocess)       ← mã hóa, chuẩn hóa dữ liệu cho mô hình hiểu
        │
        ▼
  Mô hình học máy (model)       ← trả về xác suất "nợ ngủ đáng kể", ví dụ 0,72
        │
        ▼
  Lớp quyết định (decision)
   ├─ Phân mức nguy cơ: Thấp / Trung bình / Cao
   ├─ Khuyến nghị: nên đổi gì (kèm lý do)
   ├─ What-if: đổi một thói quen, tính lại nguy cơ
   └─ Xếp hạng: top 3 phương án giảm nguy cơ nhiều nhất
        │
        ▼
  Giao diện Streamlit hiển thị kết quả
```

**Ví dụ minh họa (số giả định, không phải kết quả thật; số thật phụ thuộc rất mạnh vào chronotype và nghề của người dùng):**

1. Một người dùng nhập: dùng điện thoại trước khi ngủ 120 phút, ứng dụng chính là video ngắn, độ sáng cao, chưa bật bộ lọc ánh sáng xanh.
2. Mô hình ước lượng nguy cơ nợ ngủ khoảng 72%, hệ thống xếp mức **Cao**.
3. Hệ thống khuyến nghị: giảm còn 60 phút, bật bộ lọc ánh sáng xanh, hạ độ sáng, kèm lý do từ phân tích dữ liệu.
4. Người dùng kéo thanh trượt giảm 60 phút và bật bộ lọc; hệ thống tính lại: nguy cơ còn khoảng 45%, tức giảm khoảng 27 điểm phần trăm.
5. Hệ thống xếp hạng 3 phương án hiệu quả nhất và chỉ ra phương án **ít phải thay đổi nhất** mà vẫn xuống mức Trung bình.

## 5. Dữ liệu và điều cần nhớ nhất

Các cột trong dataset chia thành bốn nhóm:

| Nhóm | Ví dụ cột | Dùng làm gì |
|---|---|---|
| **Biến quyết định** (người dùng đổi được) | `bedtime_phone_minutes`, `primary_bedtime_app`, `screen_brightness_pct`, `blue_light_filter_active`, `caffeine_post_5pm_mg`, `physical_activity_min` | Đầu vào của mô hình và là thứ what-if thay đổi |
| **Biến ngữ cảnh** (không đổi được) | `age`, `gender`, `occupation_type`, `chronotype` | Đầu vào của mô hình |
| **Cột kết quả giấc ngủ** | `total_sleep_hours`, `sleep_latency_min`, `deep_sleep_pct`, `rem_sleep_pct`, `morning_alarm_snoozes`, `next_day_fatigue_score` | **KHÔNG đưa vào đầu vào**; chỉ để phân tích |
| **Mã định danh** | `user_id` | Bỏ, không đưa vào mô hình |
| **Nhãn (đáp án)** | `sleep_debt_category` (4 mức: Optimal Recovery, Mild Deficit, Moderate Debt, Severe Sleep Debt) | Thứ mô hình học để dự đoán |

**Vì sao không dùng cột kết quả giấc ngủ?** Ví dụ đời thường: muốn dự đoán một học sinh có đậu hay không, mà lại đưa luôn "điểm tổng kết" vào đầu vào thì dự đoán đúng 100% nhưng vô nghĩa. Ở đây nhãn nợ ngủ được tính từ chính các chỉ số giấc ngủ, nên nếu đưa chúng vào, mô hình "gian lận". Hiện tượng này gọi là **rò rỉ dữ liệu (data leakage)** và là điểm thầy cô rất thích hỏi. Hơn nữa, người dùng thật chưa biết giấc ngủ đêm nay của mình, họ chỉ biết thói quen dùng máy.

### Những gì khám phá dữ liệu (EDA) cho thấy

| Phát hiện | Con số |
|---|---|
| Nhãn nhị phân khá cân bằng | 60,1% nợ ngủ đáng kể, 39,9% không |
| **Phút dùng máy** là đòn bẩy mạnh nhất trong nhóm biến đổi được | Tỉ lệ nợ ngủ đáng kể: khoảng 36% (0-30 phút), 51%, 73%, 91%, 98% (trên 120 phút) |
| **Chronotype** rất mạnh nhưng không đổi được | Night Owl khoảng 96%, Intermediate 56%, Morning Lark 27% |
| **Nghề** khá mạnh, không đổi được | Healthcare / Shift Worker khoảng 90%, các nghề khác 55-58% |
| Lọc ánh sáng xanh, loại ứng dụng, độ sáng, caffeine, vận động, tuổi, giới tính | Ảnh hưởng rất nhỏ hoặc gần như không |

Ý nghĩa cho hệ thống: khuyến nghị quan trọng nhất luôn là giảm phút dùng máy. Với người Night Owl hoặc làm ca, nguy cơ nền rất cao (Night Owl dùng dưới 30 phút vẫn khoảng 87%), nên what-if phải nói thật rằng giảm phút dùng máy giúp ích nhưng chưa đủ để xuống mức thấp.

## 6. Mô hình học máy làm gì

- **Bài toán:** phân loại. Từ thói quen, dự đoán nhóm "nợ ngủ đáng kể" (Moderate Debt, Severe Sleep Debt) hay không. Cách gộp đã chốt sau khi xem dữ liệu: nợ ngủ đáng kể chiếm 60,1%, nhóm còn lại 39,9%, khá cân bằng.
- **Ba mô hình sẽ thử:** Logistic Regression (đơn giản, dễ giải thích), Random Forest, XGBoost (thường mạnh hơn nhưng khó giải thích hơn). Trước đó có một mô hình **baseline** (luôn đoán nhóm phổ biến nhất) để biết mô hình thật học được gì.
- **Đánh giá bằng:** Accuracy (đúng bao nhiêu phần trăm), Recall (bắt được bao nhiêu người có nguy cơ thật; ở đây rất quan trọng vì bỏ sót người có nguy cơ tệ hơn cảnh báo nhầm), F1, ROC-AUC, confusion matrix.
- **Giải thích:** feature importance (và SHAP nếu kịp) cho biết thói quen nào ảnh hưởng nhiều nhất. Kết quả này là nguồn cho các luật khuyến nghị.

## 7. Mục tiêu hướng đến

**Mục tiêu chung:** xây DSS cá nhân hóa giúp người dùng chọn cách điều chỉnh thời gian dùng điện thoại để giảm nguy cơ thiếu ngủ.

**Mục tiêu cụ thể:**
1. Khám phá mối liên hệ giữa thói quen dùng điện thoại buổi tối và nợ ngủ.
2. Xây và so sánh mô hình học máy dự đoán nguy cơ, **chỉ dùng đặc trưng hành vi**.
3. Xác định yếu tố nào ảnh hưởng nhiều và **thay đổi được**.
4. Xây lớp quyết định: khuyến nghị, what-if, xếp hạng phương án.
5. Triển khai bản demo bằng ứng dụng web.

**Sản phẩm cuối cùng:**
- Mã nguồn trên GitHub (chạy được từ máy sạch).
- Mô hình đã huấn luyện và bảng so sánh mô hình.
- Ứng dụng Streamlit demo được.
- Báo cáo và slide.
- Bản demo 3-5 phút và bộ câu hỏi phản biện.

**Thầy cô sẽ nhìn vào:**
- Có đủ thành phần DSS và chỉ ra được phần hỗ trợ quyết định không.
- Dữ liệu và mô hình được xử lý đúng không (đặc biệt không rò rỉ dữ liệu).
- Demo chạy trực tiếp mượt không.
- Từng thành viên có hiểu toàn bộ hệ thống không.

## 8. Giới hạn cần nói thật

- Dataset có dấu hiệu là **dữ liệu mô phỏng** (rất sạch, chênh lệch giữa các nhóm rất lớn), nên kết luận chỉ minh họa phương pháp, không khẳng định cho thực tế.
- What-if chỉ cho biết "theo mô hình thì nguy cơ thay đổi thế nào", **không chứng minh nhân quả**.
- Dữ liệu cắt ngang (mỗi người một lần ghi nhận), không theo dõi cùng một người theo thời gian.
- Hệ thống chỉ mang tính tham khảo, không thay thế tư vấn y tế.
- Hai yếu tố mạnh nhất (chronotype, nghề) không đổi được, nên với một số nhóm người dùng hệ thống không thể đưa nguy cơ xuống mức thấp chỉ bằng thay đổi thói quen dùng điện thoại.
- Nhóm giữ `gender` trong đầu vào; biến này gần như không ảnh hưởng và nhóm Non-Binary chỉ 248 người, nên cần kiểm tra hiệu năng theo nhóm giới tính và nói rõ cỡ mẫu nhỏ.

## 9. Thuật ngữ cần biết

| Thuật ngữ | Nghĩa ngắn gọn |
|---|---|
| DSS | Hệ thống hỗ trợ ra quyết định: dữ liệu + mô hình + giao diện + logic quyết định |
| Nợ ngủ | Thiếu ngủ tích lũy so với nhu cầu |
| Nhãn (target) | Đáp án mà mô hình học để dự đoán |
| Đặc trưng (feature) | Thông tin đầu vào của mô hình (phút dùng máy, loại app...) |
| Rò rỉ dữ liệu | Đưa thông tin "kết quả" vào đầu vào, làm mô hình đúng giả tạo |
| Baseline | Mô hình đơn giản nhất để làm mốc so sánh |
| Recall | Trong những người có nguy cơ thật, mô hình bắt được bao nhiêu phần trăm |
| Feature importance | Mức độ quan trọng của từng đặc trưng với dự đoán |
| What-if | Thử thay đổi một thói quen và xem nguy cơ thay đổi thế nào |
| Pipeline | Chuỗi bước (tiền xử lý rồi mô hình) gói thành một khối để dùng lại |

## 10. Cấu trúc repo

```
dss/ (thư mục sleep-dss)
├── data/raw/        dữ liệu gốc từ Kaggle
├── data/clean/      dữ liệu sau xử lý
├── notebooks/       EDA, thử nghiệm mô hình
├── src/
│   ├── preprocess.py   xử lý dữ liệu, định nghĩa danh sách cột
│   ├── train.py        huấn luyện, đánh giá, lưu mô hình
│   └── decision.py     lớp quyết định
├── models/          model.joblib
├── app/app.py       giao diện Streamlit
├── reports/         báo cáo, slide, hình
└── requirements.txt các thư viện (đã ghim phiên bản)
```

## 11. Những câu thầy cô hay hỏi (mọi người nên trả lời được)

1. Đâu là phần hỗ trợ ra quyết định, ai là người ra quyết định?
2. Vì sao không dùng số giờ ngủ làm đầu vào cho mô hình?
3. Vì sao chọn mô hình này mà không chọn mô hình kia?
4. What-if có chứng minh nhân quả không?
5. Dữ liệu có đáng tin không, hạn chế là gì?
6. Hệ thống khác gì một mô hình dự đoán thông thường?
7. Vì sao dùng giới tính làm đầu vào, có gây thiên lệch không?
8. Vì sao khuyến nghị giảm phút dùng máy mà người Night Owl vẫn có nguy cơ cao?
