"""
app/app.py - Giao diện Streamlit cho Hệ thống hỗ trợ quyết định (DSS)
Người phụ trách: Chính

Chạy ứng dụng từ thư mục gốc (dss/):
    streamlit run app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Đảm bảo import được package src khi chạy streamlit
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import (
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    load_model,
    predict_risk,
    rank_plans,
    recommend,
    risk_level,
    what_if,
)
from src.preprocess import CATEGORY_VALUES, DECISION_VARS, FEATURES, NUMERIC_RANGES

# ---------------------------------------------------------------------------
# Cấu hình trang
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sleep Debt DSS - Hỗ trợ quyết định giảm thiếu ngủ",
    page_icon="🌙",
    layout="wide",
)

# Ẩn nút Deploy mặc định của Streamlit
st.markdown(
    """
    <style>
    .stAppDeployButton {display:none;}
    </style>
    """,
    unsafe_allow_html=True
)

# Nhãn tiếng Việt thân thiện cho các trường
FIELD_LABELS = {
    "bedtime_phone_minutes": "Số phút dùng điện thoại trước khi ngủ (phút)",
    "screen_brightness_pct": "Độ sáng màn hình (%)",
    "caffeine_post_5pm_mg": "Lượng caffeine tiêu thụ sau 17h (mg)",
    "physical_activity_min": "Thời gian vận động thể chất trong ngày (phút)",
    "blue_light_filter_active": "Bộ lọc ánh sáng xanh (Night Shift / Eye Comfort)",
    "primary_bedtime_app": "Ứng dụng dùng chính trước khi ngủ",
    "age": "Tuổi",
    "gender": "Giới tính",
    "occupation_type": "Nghề nghiệp / Đặc thù công việc",
    "chronotype": "Nhịp sinh học (Chronotype)",
}

PRESET_PROFILES = {
    "Mặc định (Người dùng nặng - Sinh viên)": {
        "bedtime_phone_minutes": 120,
        "screen_brightness_pct": 85,
        "caffeine_post_5pm_mg": 120,
        "physical_activity_min": 15,
        "blue_light_filter_active": 0,
        "primary_bedtime_app": "TikTok / Reels",
        "age": 21,
        "gender": "Male",
        "occupation_type": "Student",
        "chronotype": "Intermediate",
    },
    "Người dùng nhẹ (Thói quen lành mạnh)": {
        "bedtime_phone_minutes": 20,
        "screen_brightness_pct": 30,
        "caffeine_post_5pm_mg": 0,
        "physical_activity_min": 45,
        "blue_light_filter_active": 1,
        "primary_bedtime_app": "News / Reading",
        "age": 28,
        "gender": "Female",
        "occupation_type": "Corporate 9-to-5",
        "chronotype": "Morning Lark",
    },
    "Cú đêm / Làm ca (Nguy cơ nền rất cao)": {
        "bedtime_phone_minutes": 95,
        "screen_brightness_pct": 75,
        "caffeine_post_5pm_mg": 160,
        "physical_activity_min": 20,
        "blue_light_filter_active": 0,
        "primary_bedtime_app": "YouTube",
        "age": 30,
        "gender": "Female",
        "occupation_type": "Healthcare / Shift Worker",
        "chronotype": "Night Owl",
    },
}


def _badge_color(level: str) -> str:
    if level == "Thấp":
        return "green"
    if level == "Trung bình":
        return "orange"
    return "red"


# ---------------------------------------------------------------------------
# Tiêu đề & Thanh bên (Sidebar nhập hồ sơ 10 trường)
# ---------------------------------------------------------------------------
st.title("🌙 Hệ thống Hỗ trợ Quyết định (DSS) Quản lý Thời gian Dùng Điện thoại")
st.caption(
    "Đề tài: Ứng dụng học máy trong hệ thống hỗ trợ quyết định quản lý thời gian sử dụng "
    "điện thoại nhằm giảm thiếu hụt giấc ngủ."
)

with st.sidebar:
    st.header("📋 1. Nhập hồ sơ người dùng")
    preset_name = st.selectbox(
        "Chọn nhanh hồ sơ mẫu (để thử nghiệm):",
        options=list(PRESET_PROFILES.keys()),
    )
    preset = PRESET_PROFILES[preset_name]

    st.subheader("A. Thói quen buổi tối (Có thể thay đổi)")
    bedtime_phone_minutes = st.slider(
        FIELD_LABELS["bedtime_phone_minutes"],
        min_value=int(NUMERIC_RANGES["bedtime_phone_minutes"][0]),
        max_value=int(NUMERIC_RANGES["bedtime_phone_minutes"][1]),
        value=int(preset["bedtime_phone_minutes"]),
        step=5,
    )
    primary_bedtime_app = st.selectbox(
        FIELD_LABELS["primary_bedtime_app"],
        options=CATEGORY_VALUES["primary_bedtime_app"],
        index=CATEGORY_VALUES["primary_bedtime_app"].index(preset["primary_bedtime_app"]),
    )
    blue_light_filter_active = st.selectbox(
        FIELD_LABELS["blue_light_filter_active"],
        options=[0, 1],
        format_func=lambda x: "Đang bật (1)" if x == 1 else "Đang tắt (0)",
        index=int(preset["blue_light_filter_active"]),
    )
    screen_brightness_pct = st.slider(
        FIELD_LABELS["screen_brightness_pct"],
        min_value=int(NUMERIC_RANGES["screen_brightness_pct"][0]),
        max_value=int(NUMERIC_RANGES["screen_brightness_pct"][1]),
        value=int(preset["screen_brightness_pct"]),
        step=5,
    )
    caffeine_post_5pm_mg = st.slider(
        FIELD_LABELS["caffeine_post_5pm_mg"],
        min_value=int(NUMERIC_RANGES["caffeine_post_5pm_mg"][0]),
        max_value=int(NUMERIC_RANGES["caffeine_post_5pm_mg"][1]),
        value=int(preset["caffeine_post_5pm_mg"]),
        step=10,
    )
    physical_activity_min = st.slider(
        FIELD_LABELS["physical_activity_min"],
        min_value=int(NUMERIC_RANGES["physical_activity_min"][0]),
        max_value=int(NUMERIC_RANGES["physical_activity_min"][1]),
        value=int(preset["physical_activity_min"]),
        step=5,
    )

    st.subheader("B. Thông tin ngữ cảnh (Cố định)")
    age = st.slider(
        FIELD_LABELS["age"],
        min_value=int(NUMERIC_RANGES["age"][0]),
        max_value=int(NUMERIC_RANGES["age"][1]),
        value=int(preset["age"]),
        step=1,
    )
    gender = st.selectbox(
        FIELD_LABELS["gender"],
        options=CATEGORY_VALUES["gender"],
        index=CATEGORY_VALUES["gender"].index(preset["gender"]),
    )
    occupation_type = st.selectbox(
        FIELD_LABELS["occupation_type"],
        options=CATEGORY_VALUES["occupation_type"],
        index=CATEGORY_VALUES["occupation_type"].index(preset["occupation_type"]),
    )
    chronotype = st.selectbox(
        FIELD_LABELS["chronotype"],
        options=CATEGORY_VALUES["chronotype"],
        index=CATEGORY_VALUES["chronotype"].index(preset["chronotype"]),
    )

profile = {
    "bedtime_phone_minutes": bedtime_phone_minutes,
    "screen_brightness_pct": screen_brightness_pct,
    "caffeine_post_5pm_mg": caffeine_post_5pm_mg,
    "physical_activity_min": physical_activity_min,
    "blue_light_filter_active": blue_light_filter_active,
    "primary_bedtime_app": primary_bedtime_app,
    "age": age,
    "gender": gender,
    "occupation_type": occupation_type,
    "chronotype": chronotype,
}

# Tính toán kết quả từ lớp quyết định
current_risk = predict_risk(profile)
current_level = risk_level(current_risk)
recs = recommend(profile)
top_plans = rank_plans(profile, top_k=3)

# ---------------------------------------------------------------------------
# Giao diện chia làm 3 Tab rõ ràng
# ---------------------------------------------------------------------------
tab_result, tab_whatif, tab_model = st.tabs(
    [
        "📊 1. Kết quả & Khuyến nghị",
        "🎛️ 2. Mô phỏng What-if & Xếp hạng Phương án",
        "🧠 3. Thông tin Mô hình & Giới hạn",
    ]
)

# === TAB 1: KẾT QUẢ & KHUYẾN NGHỊ ===
with tab_result:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Xác suất nợ ngủ đáng kể", f"{current_risk * 100:.1f}%")
    with col2:
        color = _badge_color(current_level)
        st.markdown(f"**Mức độ nguy cơ:** :{color}[**{current_level.upper()}**]")
        st.progress(min(max(current_risk, 0.0), 1.0))
    with col3:
        st.write("**Ngưỡng phân loại hiện tại:**")
        st.caption(
            f"- Thấp: < {LOW_RISK_THRESHOLD*100:.0f}%\n"
            f"- Trung bình: {LOW_RISK_THRESHOLD*100:.0f}% – < {HIGH_RISK_THRESHOLD*100:.0f}%\n"
            f"- Cao: ≥ {HIGH_RISK_THRESHOLD*100:.0f}%"
        )

    st.divider()
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.subheader("🔍 Các yếu tố chính trong hồ sơ của bạn")
        factors = []
        if profile["bedtime_phone_minutes"] > 60:
            factors.append(
                f"- **Thời gian dùng máy cao ({profile['bedtime_phone_minutes']} phút):** "
                "Đòn bẩy hành vi mạnh nhất làm tăng nguy cơ nợ ngủ."
            )
        elif profile["bedtime_phone_minutes"] <= 30:
            factors.append(
                f"- **Thời gian dùng máy hợp lý ({profile['bedtime_phone_minutes']} phút):** "
                "Giúp kiểm soát tốt nguy cơ từ màn hình."
            )
        else:
            factors.append(
                f"- **Thời gian dùng máy trung bình ({profile['bedtime_phone_minutes']} phút):** "
                "Có thể giảm xuống ≤ 30 phút để tối ưu giấc ngủ."
            )

        if profile["chronotype"] == "Night Owl":
            factors.append(
                "- **Nhịp sinh học Cú đêm (Night Owl):** Yếu tố ngữ cảnh cố định có nguy cơ nền "
                "rất cao (~96% trong dữ liệu)."
            )
        elif profile["chronotype"] == "Morning Lark":
            factors.append(
                "- **Nhịp sinh học Sơn ca (Morning Lark):** Yếu tố ngữ cảnh thuận lợi (~27% nguy cơ trong dữ liệu)."
            )

        if profile["occupation_type"] == "Healthcare / Shift Worker":
            factors.append(
                "- **Đặc thù công việc làm ca (Healthcare / Shift Worker):** Nguy cơ nền cao (~90% trong dữ liệu)."
            )

        for f in factors:
            st.markdown(f)

        with st.expander("Xem chi tiết 10 trường đầu vào hiện tại"):
            st.dataframe(
                pd.DataFrame([profile]).T.rename(columns={0: "Giá trị"}),
                use_container_width=True,
            )

    with col_right:
        st.subheader("💡 Khuyến nghị hành động cá nhân hóa")
        if not recs:
            st.success(
                "Thói quen sử dụng điện thoại buổi tối của bạn hiện rất lành mạnh! "
                "Hãy tiếp tục duy trì thời lượng dùng máy thấp và bật lọc ánh sáng xanh."
            )
        else:
            for idx, r in enumerate(recs, start=1):
                with st.container(border=True):
                    st.markdown(f"**{idx}. {r['hanh_dong']}**")
                    st.caption(f"📌 *Lý do:* {r['ly_do']}")

# === TAB 2: MÔ PHỎNG WHAT-IF & XẾP HẠNG PHƯƠNG ÁN ===
with tab_whatif:
    st.subheader("🎛️ Thử nghiệm kịch bản giả định (What-if Analysis)")
    st.write(
        "Điều chỉnh các **biến quyết định** bên dưới để xem nếu bạn thay đổi thói quen tối nay "
        "thì nguy cơ nợ ngủ sẽ giảm được bao nhiêu."
    )

    w_col1, w_col2 = st.columns([1.2, 1.2])
    with w_col1:
        new_minutes = st.slider(
            "Thử giảm số phút dùng điện thoại xuống (phút):",
            min_value=int(NUMERIC_RANGES["bedtime_phone_minutes"][0]),
            max_value=int(NUMERIC_RANGES["bedtime_phone_minutes"][1]),
            value=max(0, int(profile["bedtime_phone_minutes"]) - 30),
            step=5,
        )
        new_filter = st.selectbox(
            "Thử trạng thái bộ lọc ánh sáng xanh:",
            options=[0, 1],
            format_func=lambda x: "Bật (1)" if x == 1 else "Tắt (0)",
            index=1,
        )
        new_brightness = st.slider(
            "Thử độ sáng màn hình (%):",
            min_value=int(NUMERIC_RANGES["screen_brightness_pct"][0]),
            max_value=int(NUMERIC_RANGES["screen_brightness_pct"][1]),
            value=min(int(profile["screen_brightness_pct"]), 30),
            step=5,
        )
        new_app = st.selectbox(
            "Thử đổi ứng dụng dùng trước khi ngủ:",
            options=CATEGORY_VALUES["primary_bedtime_app"],
            index=CATEGORY_VALUES["primary_bedtime_app"].index(
                "News / Reading"
                if profile["primary_bedtime_app"] != "Messaging / Chat"
                else "Messaging / Chat"
            ),
        )
        new_caffeine = st.slider(
            "Thử mức caffeine sau 17h (mg):",
            min_value=int(NUMERIC_RANGES["caffeine_post_5pm_mg"][0]),
            max_value=int(NUMERIC_RANGES["caffeine_post_5pm_mg"][1]),
            value=0 if profile["caffeine_post_5pm_mg"] > 0 else int(profile["caffeine_post_5pm_mg"]),
            step=10,
        )
        new_activity = st.slider(
            "Thử thời gian vận động trong ngày (phút):",
            min_value=int(NUMERIC_RANGES["physical_activity_min"][0]),
            max_value=int(NUMERIC_RANGES["physical_activity_min"][1]),
            value=max(int(profile["physical_activity_min"]), 30),
            step=5,
        )

    changes_dict = {
        "bedtime_phone_minutes": new_minutes,
        "blue_light_filter_active": new_filter,
        "screen_brightness_pct": new_brightness,
        "primary_bedtime_app": new_app,
        "caffeine_post_5pm_mg": new_caffeine,
        "physical_activity_min": new_activity,
    }
    sim_result = what_if(profile, changes_dict)

    with w_col2:
        st.markdown("#### So sánh Trước và Sau khi thay đổi")
        m1, m2, m3 = st.columns(3)
        m1.metric("Nguy cơ Trước", f"{sim_result['truoc']*100:.1f}%", sim_result["muc_truoc"])
        m2.metric(
            "Nguy cơ Sau",
            f"{sim_result['sau']*100:.1f}%",
            delta=f"{sim_result['chenh_lech']*100:+.1f}%",
            delta_color="inverse",
        )
        m3.metric("Mức nguy cơ mới", sim_result["muc_sau"])

        chart_df = pd.DataFrame(
            {
                "Kịch bản": ["1. Hiện tại (Trước)", "2. Sau khi đổi (What-if)"],
                "Nguy cơ nợ ngủ (%)": [
                    round(sim_result["truoc"] * 100, 1),
                    round(sim_result["sau"] * 100, 1),
                ],
            }
        ).set_index("Kịch bản")
        st.bar_chart(chart_df, height=260)

    st.divider()
    st.subheader("🏆 Top 3 Phương án Giảm Nguy cơ Hiệu quả Nhất (Tự động xếp hạng)")

    # Kiểm tra trung thực cho nhóm có nguy cơ nền cao (Night Owl / Shift Worker)
    any_below_medium = any(p["xuong_muc_trung_binh_hoac_thap"] for p in top_plans)
    any_below_low = any(p["xuong_duoi_trung_binh"] for p in top_plans)

    if not any_below_medium:
        st.warning(
            "⚠️ **Thông báo trung thực từ mô hình:** Giảm số phút dùng điện thoại và điều chỉnh "
            "thói quen giúp giảm nguy cơ nợ ngủ nhưng **chưa đủ để đưa nguy cơ xuống mức Trung bình hoặc Thấp**. "
            f"Nguyên nhân là do yếu tố ngữ cảnh cố định trong hồ sơ của bạn (Nhịp sinh học: **{profile['chronotype']}**, "
            f"Nghề nghiệp: **{profile['occupation_type']}**) có nguy cơ nền rất cao trong dữ liệu (nhóm Night Owl "
            "dù dùng máy dưới 30 phút vẫn có tỉ lệ nợ ngủ khoảng 87%)."
        )
    elif not any_below_low and current_level != "Thấp":
        st.info(
            "ℹ️ **Lưu ý:** Các phương án bên dưới giúp đưa nguy cơ xuống mức **Trung bình**, nhưng chưa xuống "
            "được mức **Thấp (< 30%)** do ảnh hưởng nền từ nhịp sinh học hoặc đặc thù công việc."
        )

    if top_plans:
        display_rows = []
        min_effort_note = top_plans[0].get("phuong_an_it_no_luc_nhat_toan_bo")
        for p in top_plans:
            display_rows.append(
                {
                    "Hạng": f"#{p['hang']}",
                    "Phương án thay đổi": p["phuong_an"],
                    "Nguy cơ Trước (%)": f"{p['truoc']*100:.1f}%",
                    "Nguy cơ Sau (%)": f"{p['sau']*100:.1f}% ({p['muc_sau']})",
                    "Mức giảm (điểm %)": f"-{p['muc_giam']*100:.1f}%",
                    "Điểm nỗ lực": p["no_luc"],
                }
            )
        st.table(pd.DataFrame(display_rows))

        if min_effort_note:
            st.success(
                f"✨ **Phương án tốn ít nỗ lực nhất để đạt ngưỡng an toàn hơn:** {min_effort_note}"
            )
    else:
        st.info("Hồ sơ của bạn đã ở mức tối ưu, không cần thay đổi thêm.")

# === TAB 3: THÔNG TIN MÔ HÌNH & GIỚI HẠN ===
with tab_model:
    st.subheader("📈 Bảng So sánh Hiệu năng các Mô hình Học máy")
    metrics_path = ROOT / "models" / "metrics.csv"
    cv_metrics_path = ROOT / "models" / "cv_metrics.csv"
    fi_path = ROOT / "models" / "feature_importance.csv"

    if metrics_path.exists():
        st.markdown("**1. Kết quả đánh giá trên tập kiểm tra (`models/metrics.csv`):**")
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True)
    else:
        st.info("Chưa tìm thấy file `models/metrics.csv`.")

    if cv_metrics_path.exists():
        st.markdown("**2. Kết quả Cross-Validation 5-fold (`models/cv_metrics.csv`):**")
        st.dataframe(pd.read_csv(cv_metrics_path), use_container_width=True)

    if fi_path.exists():
        st.markdown("**3. Độ quan trọng của đặc trưng (`models/feature_importance.csv`):**")
        st.dataframe(pd.read_csv(fi_path), use_container_width=True)

    st.divider()
    st.subheader("⚠️ Hạn chế của Hệ thống & Lưu ý Quan trọng")
    st.markdown(
        """
        1. **Đặc điểm dữ liệu:** Bộ dữ liệu Kaggle (*Sleep Debt & Screen Time: Late Night Phone Habits*, 8.500 dòng) có các dấu hiệu của **dữ liệu mô phỏng** (phân phối rất đều, ranh giới giữa các nhóm rất rõ). Do đó, kết quả chủ yếu minh họa phương pháp xây dựng Hệ thống Hỗ trợ Quyết định (DSS).
        2. **Mối quan hệ tương quan vs. Nhân quả:** Phân tích *What-if* chỉ cho biết xác suất dự đoán thay đổi thế nào theo mô hình học máy, **không chứng minh quan hệ nhân quả tuyệt đối** vì dữ liệu là cắt ngang (cross-sectional), không theo dõi dọc theo thời gian trên cùng một cá nhân.
        3. **Chống rò rỉ dữ liệu (No Data Leakage):** Hệ thống tuyệt đối không dùng các cột kết quả giấc ngủ (`total_sleep_hours`, `sleep_latency_min`...) làm đầu vào dự đoán vì người dùng trước khi đi ngủ chưa thể biết các chỉ số này.
        4. **Miễn trừ trách nhiệm y tế:** *Hệ thống chỉ mang tính chất tham khảo hỗ trợ điều chỉnh thói quen cá nhân, hoàn toàn **không thay thế chẩn đoán hay tư vấn y tế chuyên khoa**.*
        """
    )