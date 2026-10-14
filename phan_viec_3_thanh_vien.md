# Phân việc dự án DSS cho Hiếu, Chính, Tấn

Cả nhóm đọc `giai_thich_de_tai_DSS.md` trước. Cột **Hạn** để trống, nhóm tự điền ngày. Cột **Xong khi** là tiêu chí nghiệm thu: đạt đúng điều kiện đó mới coi là xong và mới push.

> **Cách tôi xếp người:** Hiếu phụ trách mô hình và đánh giá (thiên thống kê), Chính phụ trách lớp quyết định và ứng dụng (phần lập trình nặng và ghép nối), Tấn phụ trách kiểm thử, báo cáo, demo. Nếu thế mạnh của mỗi người khác, hoán đổi cả khối việc chứ đừng xé nhỏ.
> **Dương (chủ repo)** làm dữ liệu, tiền xử lý và điều phối (mục 5).

## 1. Hợp đồng chung (phải giữ đúng để ghép được)

**Danh sách đặc trưng** nằm duy nhất trong `src/preprocess.py` (Dương viết). Các người khác `import` từ đó, không tự gõ lại tên cột:

| Tên | Ý nghĩa |
|---|---|
| `FEATURES` | Toàn bộ cột đầu vào |
| `DECISION_VARS` | Biến quyết định (người dùng đổi được) |
| `CONTEXT_VARS` | Biến ngữ cảnh (không đổi được) |
| `TARGET` | Tên cột nhãn nhị phân `high_sleep_debt` (1 = Moderate Debt hoặc Severe Sleep Debt) |

`src/preprocess.py` **đã hoàn thành**. Nội dung hiện tại:

| Nhóm | Cột |
|---|---|
| `DECISION_VARS` (6) | `bedtime_phone_minutes`, `screen_brightness_pct`, `caffeine_post_5pm_mg`, `physical_activity_min`, `blue_light_filter_active`, `primary_bedtime_app` |
| `CONTEXT_VARS` (4) | `age`, `gender`, `occupation_type`, `chronotype` |
| `LEAKAGE_COLS` (loại khỏi đầu vào) | `sleep_latency_min`, `total_sleep_hours`, `deep_sleep_pct`, `rem_sleep_pct`, `morning_alarm_snoozes`, `next_day_fatigue_score` |

Hàm và hằng số có sẵn để dùng: `prepare_data()` (ra `X_train, X_test, y_train, y_test`, chia 80/20 có phân tầng), `build_preprocessor()` (đặt đầu Pipeline), `profile_to_frame(profile)` (dict thành DataFrame một dòng, có kiểm tra), `CATEGORY_VALUES`, `NUMERIC_RANGES` (dựng form và thanh trượt), `get_xy(..., multiclass=, include_leakage=)`.

**Mô hình** (`models/model.joblib`, Hiếu giao): là **một Pipeline hoàn chỉnh** gồm mã hóa và mô hình. Nhận DataFrame có đúng các cột thô trong `FEATURES`, gọi `predict_proba` trả về xác suất lớp "nợ ngủ đáng kể".

**Hàm của `src/decision.py`** (Chính viết, Tấn kiểm thử, app gọi). Đủ 6 hàm, đúng tên và chữ ký:

| Hàm | Nhận | Trả về |
|---|---|---|
| `load_model()` | không | Pipeline đã nạp |
| `predict_risk(profile)` | `dict` thói quen | `float` xác suất 0-1 |
| `risk_level(p)` | `float` | `"Thấp"`, `"Trung bình"` hoặc `"Cao"` |
| `recommend(profile)` | `dict` | `list[dict]`, mỗi phần tử có `hanh_dong` và `ly_do` |
| `what_if(profile, changes)` | `dict`, `dict` thay đổi | `dict` có `truoc`, `sau`, `chenh_lech` |
| `rank_plans(profile, top_k=3)` | `dict`, số phương án | `list[dict]` xếp theo mức giảm nguy cơ |

**Mô hình tạm:** trong lúc chờ Hiếu giao, Chính dùng mô hình giả cùng định dạng để làm tiếp, khi có `model.joblib` thật thì thay vào, không sửa app.

**Nhánh và file sở hữu:**

| Người | Nhánh Git | File/thư mục sở hữu |
|---|---|---|
| Hiếu | `model` | `notebooks/02_model.ipynb`, `src/train.py`, `models/` |
| Chính | `decision`, `app` | `src/decision.py`, `app/` |
| Tấn | `report`, `test` | `notebooks/01b_eda_charts.ipynb`, `tests/`, `reports/` |

Mỗi người chỉ sửa file của mình. Cần sửa file người khác thì nhắn họ hoặc tạo Pull Request. Mọi thay đổi vào `main` đi qua Pull Request, Dương review.

---

## 2. Việc của Hiếu: mô hình và đánh giá

Nhận từ Dương: `src/preprocess.py` và `data/clean/`. Giao cho Chính: `models/model.joblib`. Giao cho Tấn: bảng và hình cho báo cáo.

| Thứ tự | Việc | Xong khi | Hạn |
|---|---|---|---|
| 1 | Đọc `giai_thich_de_tai_DSS.md`, chạy `notebooks/01_eda.ipynb` của Dương. Tạo `notebooks/02_model.ipynb`, nạp dữ liệu bằng `prepare_data()` | `X_train` có 6.800 dòng, `X_test` 1.700 dòng, tỉ lệ nhãn 1 khoảng 60,1% ở cả hai | |
| 2 | **Baseline:** mô hình đoán lớp phổ biến nhất, rồi Logistic Regression. Lưu `models/model.joblib` bản đầu và **push sớm** để Chính tích hợp | Chính nạp được file, `predict_proba` trả về số hợp lệ | |
| 3 | Huấn luyện Random Forest và XGBoost; cross-validation 5-fold, tinh chỉnh tham số chỉ trên tập huấn luyện | Có điểm CV trung bình và độ lệch cho cả 3 mô hình | |
| 4 | Đánh giá trên tập kiểm tra: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix. Xuất `models/metrics.csv` (cột: mô hình, accuracy, precision, recall, f1, roc_auc) | File có đủ 3 mô hình và baseline | |
| 5 | **Thí nghiệm rò rỉ dữ liệu:** huấn luyện thêm một bản có đưa các cột kết quả giấc ngủ vào (`prepare_data(include_leakage=True)` và `build_preprocessor(include_leakage=True)`), so điểm với bản đúng | Có bảng so sánh rõ hai bản, chỉ ra bản rò rỉ cao bất thường | |
| 6 | Feature importance (SHAP nếu kịp), kiểm tra calibration. Xuất `models/feature_importance.csv` (cột: đặc trưng, mức quan trọng) và hình vào `reports/figures/` | Biết top 5 đặc trưng; Chính dùng file này viết luật khuyến nghị | |
| 7 | Chọn mô hình cuối, ghi lý do (hiệu năng và khả năng giải thích). Lưu `models/model.joblib` bản chính thức, gói đúng một Pipeline | Notebook Run All không lỗi; Chính nạp được, nguy cơ hiển thị khớp | |
| 7a | **Kiểm tra theo nhóm:** tính Recall và ROC-AUC riêng cho từng nhóm giới tính (nhóm Non-Binary chỉ 248 người, ghi chú cỡ mẫu nhỏ) và từng chronotype | Có bảng hiệu năng theo nhóm để trả lời câu hỏi về thiên lệch | |
| 8 | Viết `src/train.py` chạy lại toàn bộ huấn luyện bằng một lệnh | `python src/train.py` ra lại đúng `model.joblib` và `metrics.csv` | |
| 9 | Viết chương Mô hình trong báo cáo (đã gồm bảng so sánh, rò rỉ, importance) và soạn câu trả lời "vì sao chọn mô hình này" | Tấn nhận được văn bản và hình | |
| 10 | Push nhánh `model`, tạo Pull Request | Dương duyệt và gộp | |

---

## 3. Việc của Chính: lớp quyết định và ứng dụng

Nhận từ Hiếu: `model.joblib` (bản tạm rồi bản chính), `feature_importance.csv`. Nhận từ Dương: danh sách đặc trưng. Giao cho Tấn: `decision.py` để kiểm thử; app để dùng thử.

| Thứ tự | Việc | Xong khi | Hạn |
|---|---|---|---|
| 1 | Đọc `giai_thich_de_tai_DSS.md`. Dựng khung Streamlit trong `app/app.py` bằng mô hình giả, form nhập dựng từ `CATEGORY_VALUES` và `NUMERIC_RANGES` (có cả ô chọn giới tính) | `streamlit run app/app.py` mở được, form đủ 10 trường | |
| 2 | Viết `src/decision.py` đủ 6 hàm theo hợp đồng, gắn vào mô hình giả. Push sớm khung hàm để Tấn viết kiểm thử | Tấn gọi được cả 6 hàm với một hồ sơ mẫu | |
| 3 | `risk_level`: chia Thấp, Trung bình, Cao theo ngưỡng (ban đầu 30% và 60%, chỉnh sau khi xem phân phối xác suất thật) | Hồ sơ nhẹ ra Thấp, hồ sơ nặng ra Cao | |
| 4 | `recommend`: viết luật từ EDA và `feature_importance.csv` của Hiếu; mỗi luật có `hanh_dong` và `ly_do`. Theo EDA, giảm phút dùng máy là khuyến nghị chính; lọc ánh sáng xanh, đổi app, hạ độ sáng, giảm caffeine, vận động chỉ là khuyến nghị phụ, ghi rõ tác dụng nhỏ | Hồ sơ nặng ra ít nhất 2 khuyến nghị, hồ sơ nhẹ không bị khuyên vô lý, khuyến nghị phụ không bị phóng đại | |
| 5 | `what_if`: đổi biến quyết định, tính lại nguy cơ, trả về trước/sau/chênh lệch. Chỉ cho đổi biến trong `DECISION_VARS`, không cho đổi biến ngữ cảnh | Đổi biến ngữ cảnh thì báo lỗi rõ ràng | |
| 6 | `rank_plans`: thử lưới thay đổi khả thi (giảm 15/30/45/60 phút, bật bộ lọc, hạ độ sáng, đổi app), lọc theo ràng buộc, trả top 3 và phương án ít nỗ lực nhất xuống dưới ngưỡng Trung bình | Top 3 xếp đúng thứ tự giảm nguy cơ | |
| 7 | Thay mô hình giả bằng `model.joblib` thật của Hiếu | App chạy ra nguy cơ từ mô hình thật, không sửa gì khác ngoài đường dẫn | |
| 8 | Hoàn thiện app: trang nhập, trang kết quả (nguy cơ %, mức, yếu tố chính), trang khuyến nghị và what-if (thanh trượt, biểu đồ trước/sau, bảng top phương án), trang thông tin mô hình (bảng so sánh, hạn chế, lưu ý không thay thế tư vấn y tế) | Đi hết luồng từ nhập đến xếp hạng không lỗi | |
| 9 | Sửa lỗi theo danh sách của Tấn, kiểm tra app chạy sau khi `git clone` và `pip install -r requirements.txt` | Chạy được trên máy sạch | |
| 10 | Viết chương Lớp quyết định và Ứng dụng (kèm ảnh chụp app) | Tấn nhận được văn bản và hình | |
| 11 | Push nhánh `decision` và `app`, tạo Pull Request | Dương duyệt và gộp | |

Điểm cần cẩn thận: `what_if` và `rank_plans` đều gọi mô hình nhiều lần, nên dựng DataFrame một dòng cho mỗi lần thử, đừng sửa DataFrame gốc.

Điểm cần nói thật trên giao diện: người Night Owl hoặc làm ca (Healthcare / Shift Worker) có nguy cơ nền rất cao (Night Owl dùng dưới 30 phút vẫn khoảng 87% trong dữ liệu). Khi `rank_plans` không tìm được phương án nào xuống dưới ngưỡng Trung bình, app phải hiển thị thông báo trung thực (ví dụ "giảm phút dùng máy giúp giảm nguy cơ nhưng chưa đủ xuống mức thấp") thay vì để trống hoặc hứa quá.

---

## 4. Việc của Tấn: kiểm thử, báo cáo, demo

Nhận từ Dương: dữ liệu và EDA. Nhận từ Hiếu và Chính: văn bản chương của từng người, hình. Giao: báo cáo, slide, kịch bản demo.

| Thứ tự | Việc | Xong khi | Hạn |
|---|---|---|---|
| 1 | Đọc `giai_thich_de_tai_DSS.md`. Vẽ biểu đồ EDA trong `notebooks/01b_eda_charts.ipynb` (không sửa notebook của Dương): nhãn theo phút dùng máy, theo loại app, theo bộ lọc ánh sáng xanh, theo chronotype. Lưu hình vào `reports/figures/` | 4-6 hình rõ, có tiêu đề và đơn vị | |
| 2 | Ghi chú các dấu hiệu dữ liệu giống mô phỏng (phân phối quá đều, quan hệ quá rõ) cho phần hạn chế | Một đoạn văn bản kèm bằng chứng từ biểu đồ | |
| 3 | Vẽ **sơ đồ kiến trúc DSS** (người dùng, dữ liệu, mô hình, lớp quyết định, giao diện), lưu `reports/figures/kien_truc_dss.png` | Có đủ 5 thành phần như bảng trong mục 3 của file giải thích | |
| 4 | Lập bộ 8-10 hồ sơ mẫu trong `tests/profiles.json` (người dùng nặng, trung bình, nhẹ; có Night Owl, Morning Lark, Healthcare / Shift Worker; trường hợp biên như 0 và 180 phút dùng máy, độ sáng 100) | Mỗi hồ sơ có đủ 10 cột trong `FEATURES`, **gồm cả `gender`**, giá trị phân loại lấy từ `CATEGORY_VALUES` | |
| 5 | Viết `tests/kiem_thu_quyet_dinh.py`: chạy từng hồ sơ qua 6 hàm của `decision.py` và kiểm tra: nguy cơ nằm trong 0-1; **giảm phút dùng máy không làm nguy cơ tăng bất thường**; hồ sơ nặng có khuyến nghị; `what_if` không nhận biến ngữ cảnh; top phương án xếp đúng thứ tự. Ghi kết quả vào `docs/ket_qua_kiem_thu.csv` (cột: hồ sơ, kiểm tra, kết quả, ghi chú) | Mọi dòng là `True` hoặc có lý do rõ; đưa danh sách lỗi cho Chính | |
| 6 | Dùng thử app như người dùng thật: ghi chỗ khó hiểu, nhãn sai, chữ lỗi, gửi Chính | Danh sách lỗi có ảnh chụp và cách tái hiện | |
| 7 | Tổng hợp báo cáo (Word hoặc Docs): tự viết giới thiệu, cơ sở lý thuyết (DSS, học máy phân loại, đánh giá mô hình), hạn chế, kết luận, hướng phát triển; ghép chương Dữ liệu của Dương, Mô hình của Hiếu, Lớp quyết định của Chính | Báo cáo đủ các chương theo mục lục, thống nhất cách trình bày và đánh số hình, bảng | |
| 8 | Làm slide khoảng 10-12 trang, có trang sơ đồ kiến trúc, trang rò rỉ dữ liệu, trang demo, trang hạn chế | Số liệu trong slide khớp báo cáo | |
| 9 | Soạn `reports/qa_phan_bien.md` gồm 15-20 câu hỏi kèm trả lời (bắt đầu từ mục 11 file giải thích) | Mỗi câu có người chịu trách nhiệm trả lời | |
| 10 | Viết kịch bản demo 3-5 phút (một người dùng nặng, một người dùng nhẹ, một what-if), chuẩn bị ảnh chụp và video quay màn hình dự phòng | Cả nhóm chạy theo kịch bản được trong thời gian quy định | |
| 11 | Điều phối buổi tập: bấm giờ, hỏi chéo mỗi người về phần của người khác | Mỗi người giải thích được luồng từ dữ liệu đến khuyến nghị | |
| 12 | Push nhánh `report` và `test`, tạo Pull Request | Dương duyệt và gộp | |

---

## 5. Việc của Dương (để mọi người biết mình chờ gì)

| Việc | Giao cho ai | Xong khi |
|---|---|---|
| **Đã xong:** EDA (`notebooks/01_eda.ipynb`), chốt nhãn 2 lớp, danh sách cột, nhóm quyết định giữ `gender` | Cả nhóm | Đã chốt |
| **Đã xong:** `src/preprocess.py` với `FEATURES`, `DECISION_VARS`, `CONTEXT_VARS`, `TARGET`, `prepare_data`, `build_preprocessor`, `profile_to_frame` | Hiếu, Chính | Chạy `python src/preprocess.py` ra 8.500 dòng |
| **Đã xong:** lưu `data/clean/sleep_clean.csv` | Hiếu | Hiếu nạp được |
| Review và gộp Pull Request, giữ `main` luôn chạy được | Cả nhóm | `main` chạy được sau mỗi lần gộp |
| Chương Dữ liệu và Tiền xử lý trong báo cáo | Tấn | Tấn nhận được văn bản |
| Chạy thử trên máy sạch: clone, cài, `streamlit run app/app.py` | Cả nhóm | Chạy không lỗi, sửa lỗi môi trường |

## 6. Điểm đồng bộ (họp ngắn khoảng 15 phút)

| Khi nào | Chốt gì |
|---|---|
| Sau khi Dương xong EDA | Nhãn 2 hay 4 lớp, danh sách cột đầu vào và biến quyết định |
| Sau khi Dương xong `preprocess.py` | Chữ ký hàm, hợp đồng ở mục 1, ai đang chờ ai |
| Sau khi Hiếu chọn mô hình cuối | Giao `model.joblib`, Chính chốt luật khuyến nghị theo importance |
| Sau khi Chính xong app | Cả nhóm dùng thử, Tấn gom danh sách lỗi |
| Trước buổi bảo vệ | Chốt kịch bản demo và ai nói phần nào |

## 7. Phân công báo cáo và thuyết trình

| Phần | Người viết | Người trình bày (gợi ý) |
|---|---|---|
| Giới thiệu, cơ sở lý thuyết, kiến trúc DSS | Tấn | Tấn |
| Dữ liệu và tiền xử lý | Dương | Dương |
| Mô hình và đánh giá | Hiếu | Hiếu |
| Lớp quyết định và demo ứng dụng | Chính | Chính (thao tác demo) |
| Hạn chế, kết luận, hướng phát triển | Tấn | Tấn |

## 8. Nếu có người trễ

- Chính là điểm nghẽn lớn nhất (ba hàm quyết định và cả ứng dụng). Nếu quá tải, Dương nhận trang thông tin mô hình và dữ liệu của app, Tấn nhận phần soạn hồ sơ mẫu và chữ trong app.
- Nếu Hiếu trễ, dùng baseline Logistic Regression làm mô hình chính (đủ cho bản tối thiểu), cải tiến sau.
- Khi cần cắt giảm, bỏ theo thứ tự: SHAP, calibration, nhãn 4 lớp, xếp hạng phương án tự động. Luôn giữ: mô hình, luật khuyến nghị, what-if, app chạy được, báo cáo và slide.
