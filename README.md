# Sleep DSS: hệ thống hỗ trợ quyết định quản lý thời gian dùng điện thoại để giảm thiếu ngủ

Đồ án cuối kì môn **Hệ trợ giúp quyết định (DSS)**. Đề tài: *Ứng dụng học máy trong hệ thống hỗ trợ quyết định quản lý thời gian sử dụng điện thoại nhằm giảm thiếu hụt giấc ngủ*.

Người dùng nhập thói quen dùng điện thoại buổi tối. Hệ thống dùng học máy ước lượng nguy cơ nợ ngủ, đưa ra khuyến nghị cụ thể và cho thử "nếu tôi thay đổi thói quen thì nguy cơ giảm còn bao nhiêu" (what-if).

> Trạng thái: đã xong dữ liệu, tiền xử lý và khám phá dữ liệu (EDA). Mô hình, lớp quyết định và ứng dụng Streamlit đang phát triển.

## Các thành phần của DSS

| Thành phần | Trong dự án |
|---|---|
| Người ra quyết định | Người dùng điện thoại muốn cải thiện giấc ngủ |
| Cơ sở dữ liệu | `data/raw/bedtime_screentime_sleep_debt.csv` (8.500 dòng, 18 cột) |
| Cơ sở mô hình | Mô hình học máy phân loại, lưu ở `models/model.joblib` |
| Lớp quyết định | `src/decision.py`: mức nguy cơ, khuyến nghị, what-if, xếp hạng phương án |
| Giao diện | `app/app.py` (Streamlit) |

## Dữ liệu

Nguồn: Talwar, S., *Sleep Debt & Screen Time: Late Night Phone Habits*, Kaggle. Kiểm tra giấy phép sử dụng trên trang Kaggle của dataset.

- Nhãn: `sleep_debt_category` (4 mức). Bản chính gộp thành nhị phân `high_sleep_debt`: 1 nếu Moderate Debt hoặc Severe Sleep Debt (60,1%), 0 nếu còn lại (39,9%).
- **Biến quyết định** (người dùng đổi được): `bedtime_phone_minutes`, `primary_bedtime_app`, `screen_brightness_pct`, `blue_light_filter_active`, `caffeine_post_5pm_mg`, `physical_activity_min`.
- **Biến ngữ cảnh** (không đổi được): `age`, `gender`, `occupation_type`, `chronotype`.
- **Cột bị loại khỏi đầu vào** vì là kết quả giấc ngủ (gây rò rỉ dữ liệu): `sleep_latency_min`, `total_sleep_hours`, `deep_sleep_pct`, `rem_sleep_pct`, `morning_alarm_snoozes`, `next_day_fatigue_score`. `user_id` chỉ là mã định danh.

Phát hiện chính từ EDA: phút dùng điện thoại trước khi ngủ là đòn bẩy mạnh nhất trong nhóm biến đổi được; chronotype và nghề rất mạnh nhưng không đổi được; các biến còn lại ảnh hưởng rất nhỏ. Dữ liệu có dấu hiệu mô phỏng nên kết quả chỉ mang tính minh họa phương pháp. Chi tiết trong `notebooks/01_eda.ipynb`.

## Cài đặt

Yêu cầu: Git và Python. Dự án được phát triển trên Python 3.14.4; cả nhóm dùng cùng phiên bản để mô hình lưu bằng `joblib` đọc được trên mọi máy.

```powershell
git clone https://github.com/daiduongtatran/dss.git
cd dss
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Nếu PowerShell chặn script khi kích hoạt môi trường ảo: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, rồi kích hoạt lại. Có thể bỏ qua bước kích hoạt và gọi thẳng `.venv\Scripts\python.exe`.
- Mac/Linux: kích hoạt bằng `source .venv/bin/activate`.
- Đầu dòng lệnh hiện `(.venv)` là môi trường đã bật.

## Cách chạy

**Làm sạch dữ liệu** (tạo `data/clean/sleep_clean.csv`, phải ra 8.500 dòng):

```powershell
python src/preprocess.py
```

**Notebook EDA:** mở `notebooks/01_eda.ipynb` trong VS Code, chọn kernel `.venv`, bấm *Run All*.

**Ứng dụng demo** (sẽ có khi hoàn thành):

```powershell
streamlit run app/app.py
```

## Dùng nhanh trong code

```python
from sklearn.pipeline import Pipeline
from src.preprocess import prepare_data, build_preprocessor, profile_to_frame

X_train, X_test, y_train, y_test = prepare_data()           # chia 80/20, có phân tầng
pipe = Pipeline([("prep", build_preprocessor()), ("model", clf)])
pipe.fit(X_train, y_train)

# Dự đoán cho một hồ sơ (dict đủ 10 trường trong FEATURES)
risk = pipe.predict_proba(profile_to_frame(profile))[0, 1]
```

Tên cột và danh sách đặc trưng định nghĩa **duy nhất** trong `src/preprocess.py` (`FEATURES`, `DECISION_VARS`, `CONTEXT_VARS`, `TARGET`). Các file khác chỉ import từ đó, không tự gõ lại tên cột.

## Cấu trúc thư mục

```
dss/
├── data/
│   ├── raw/            dữ liệu gốc từ Kaggle
│   └── clean/          dữ liệu sau làm sạch (sleep_clean.csv)
├── notebooks/          01_eda.ipynb, 01b_eda_charts.ipynb, 02_model.ipynb
├── src/
│   ├── preprocess.py   làm sạch, định nghĩa cột, bộ tiền xử lý, profile_to_frame
│   ├── train.py        huấn luyện, đánh giá, lưu mô hình
│   └── decision.py     mức nguy cơ, khuyến nghị, what-if, xếp hạng phương án
├── models/             model.joblib, metrics.csv, feature_importance.csv
├── app/                app.py (Streamlit)
├── tests/              hồ sơ mẫu và kiểm thử khuyến nghị
├── reports/            báo cáo, slide, hình (figures/)
├── requirements.txt    thư viện, đã ghim phiên bản
└── README.md
```

## Quy ước làm việc nhóm

- Mỗi người một nhánh; không commit thẳng vào `main`. Làm xong tạo Pull Request, người điều phối review rồi gộp.
- Trước khi làm: `git pull origin main`. Commit nhỏ, thông điệp rõ nghĩa.
- Chỉ sửa file thuộc phần của mình; cần sửa file người khác thì nhắn họ hoặc tạo Pull Request.
- Không commit thư mục `.venv`.

## Thành viên

| Thành viên | Phụ trách |
|---|---|
| Dương | Dữ liệu, tiền xử lý, EDA, điều phối repo |
| Hiếu | Mô hình học máy và đánh giá |
| Chính | Lớp quyết định và ứng dụng Streamlit |
| Tấn | Kiểm thử, báo cáo, slide, demo |

## Hạn chế

- Dữ liệu có dấu hiệu là dữ liệu mô phỏng; dữ liệu cắt ngang (mỗi người một lần ghi nhận).
- What-if cho biết thay đổi theo mô hình trên dữ liệu quan sát, không chứng minh quan hệ nhân quả.
- Hệ thống chỉ mang tính tham khảo, không thay thế tư vấn y tế.
