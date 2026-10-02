# Sleep DSS

Hệ thống hỗ trợ quyết định quản lý thời gian dùng điện thoại nhằm giảm thiếu hụt giấc ngủ (môn Hệ trợ giúp quyết định).

## Dữ liệu
Talwar, S. *Sleep Debt & Screen Time: Late Night Phone Habits*, Kaggle.
Đặt file CSV vào `data/raw/`.

## Cài đặt
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chạy ứng dụng
```
streamlit run app/app.py
```

## Cấu trúc
- `data/raw`, `data/clean`: dữ liệu gốc và đã làm sạch
- `notebooks/`: EDA và thử nghiệm mô hình
- `src/`: preprocess, train, decision (khuyến nghị / what-if)
- `models/`: mô hình đã huấn luyện
- `app/`: giao diện Streamlit
- `reports/`: báo cáo, slide