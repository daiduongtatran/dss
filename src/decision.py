"""
src/decision.py - Lớp quyết định cho hệ thống hỗ trợ quyết định (DSS)
Người phụ trách: Chính

Cung cấp đúng 6 hàm theo hợp đồng chung của nhóm:
    1. load_model()
    2. predict_risk(profile)
    3. risk_level(p)
    4. recommend(profile)
    5. what_if(profile, changes)
    6. rank_plans(profile, top_k=3)
"""
from __future__ import annotations

import numbers
import sys
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Đảm bảo import được src.preprocess dù chạy từ bất kỳ thư mục nào
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import (
    CATEGORY_VALUES,
    CONTEXT_VARS,
    DECISION_VARS,
    FEATURES,
    NUMERIC_RANGES,
    VALID_LIMITS,
    build_preprocessor,
    prepare_data,
    profile_to_frame,
)

# ---------------------------------------------------------------------------
# Cấu hình ngưỡng & đường dẫn
# ---------------------------------------------------------------------------
MODEL_PATH = ROOT / "models" / "model.joblib"
FEATURE_IMPORTANCE_PATH = ROOT / "models" / "feature_importance.csv"

LOW_RISK_THRESHOLD = 0.30   # < 0.30: Thấp
HIGH_RISK_THRESHOLD = 0.60  # >= 0.60: Cao (0.30 <= p < 0.60: Trung bình)

HIGH_STIMULATION_APPS = {
    "TikTok / Reels",
    "Instagram / Reddit",
    "YouTube",
    "Streaming (Netflix/Hulu)",
}
LOW_STIMULATION_APPS = ["News / Reading", "Messaging / Chat"]

_MODEL_CACHE: Any = None


# ---------------------------------------------------------------------------
# Mô hình giả dự phòng (chỉ dùng khi chưa có models/model.joblib)
# ---------------------------------------------------------------------------
class _FallbackSleepModel:
    """Mô hình dự phòng theo đúng quy luật EDA nếu chưa có file model.joblib."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = []
        for _, row in X.iterrows():
            logit = -1.2
            # Phút dùng máy là đòn bẩy mạnh nhất
            mins = float(row["bedtime_phone_minutes"])
            logit += (mins - 45.0) * 0.032

            # Chronotype (biến ngữ cảnh rất mạnh)
            chrono = row["chronotype"]
            if chrono == "Night Owl":
                logit += 2.4
            elif chrono == "Morning Lark":
                logit -= 1.1

            # Nghề nghiệp
            if row["occupation_type"] == "Healthcare / Shift Worker":
                logit += 1.6

            # Các biến quyết định phụ (tác động nhỏ theo EDA)
            if int(row["blue_light_filter_active"]) == 0:
                logit += 0.15
            logit += (float(row["screen_brightness_pct"]) - 50.0) * 0.003
            logit += float(row["caffeine_post_5pm_mg"]) * 0.0015
            logit -= float(row["physical_activity_min"]) * 0.002
            if row["primary_bedtime_app"] in HIGH_STIMULATION_APPS:
                logit += 0.12

            p1 = float(1.0 / (1.0 + np.exp(-logit)))
            p1 = min(max(p1, 0.01), 0.99)
            probs.append([1.0 - p1, p1])
        return np.asarray(probs, dtype=float)


# ---------------------------------------------------------------------------
# 1. Nạp mô hình
# ---------------------------------------------------------------------------
def load_model(path: str | Path = MODEL_PATH) -> Any:
    """
    Nạp Pipeline từ models/model.joblib.
    Có bộ nhớ đệm để gọi nhiều lần trong what_if và rank_plans không bị chậm.
    """
    global _MODEL_CACHE
    path = Path(path)

    if path == MODEL_PATH and _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if path.exists():
        model = joblib.load(path)
    else:
        try:
            X_train, _, y_train, _ = prepare_data()
            model = Pipeline(
                [
                    ("prep", build_preprocessor(scale=True)),
                    ("model", LogisticRegression(max_iter=1000, random_state=42)),
                ]
            )
            model.fit(X_train, y_train)
        except Exception:
            model = _FallbackSleepModel()

    if path == MODEL_PATH:
        _MODEL_CACHE = model
    return model


# ---------------------------------------------------------------------------
# 2. Dự đoán xác suất nguy cơ nợ ngủ
# ---------------------------------------------------------------------------
def predict_risk(profile: dict) -> float:
    """
    Nhận dict hồ sơ người dùng (10 đặc trưng trong FEATURES),
    kiểm tra hợp lệ qua profile_to_frame và trả về xác suất nợ ngủ (float 0.0 - 1.0).
    """
    df_row = profile_to_frame(profile)
    model = load_model()
    proba = model.predict_proba(df_row)
    risk = float(proba[0, 1])
    return float(np.clip(risk, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 3. Phân mức nguy cơ
# ---------------------------------------------------------------------------
def risk_level(p: float) -> str:
    """
    Chia mức nguy cơ theo xác suất p:
        - p < 0.30  -> "Thấp"
        - p < 0.60  -> "Trung bình"
        - p >= 0.60 -> "Cao"
    """
    if not isinstance(p, numbers.Real) or isinstance(p, bool):
        raise ValueError(f"Xác suất p phải là số thực trong [0, 1], nhận được: {p!r}")
    p_val = float(p)
    if not 0.0 <= p_val <= 1.0:
        raise ValueError(f"Xác suất p={p_val} nằm ngoài khoảng hợp lệ [0, 1].")

    if p_val < LOW_RISK_THRESHOLD:
        return "Thấp"
    if p_val < HIGH_RISK_THRESHOLD:
        return "Trung bình"
    return "Cao"


# ---------------------------------------------------------------------------
# 4. Mô phỏng What-if
# ---------------------------------------------------------------------------
def what_if(profile: dict, changes: dict) -> dict:
    """
    Thay đổi một hoặc nhiều biến quyết định (DECISION_VARS), tính lại nguy cơ.
    Báo lỗi ValueError rõ ràng nếu changes chứa biến ngữ cảnh (CONTEXT_VARS)
    hoặc biến lạ. Không sửa dict hồ sơ gốc.
    """
    if not isinstance(changes, dict):
        raise TypeError("Tham số 'changes' phải là một dict.")

    # Kiểm tra chặt: tuyệt đối không cho đổi biến ngữ cảnh
    for key in changes:
        if key in CONTEXT_VARS:
            raise ValueError(
                f"Không được phép thay đổi biến ngữ cảnh '{key}'. "
                f"Chỉ được thay đổi các biến quyết định: {DECISION_VARS}"
            )
        if key not in DECISION_VARS:
            raise ValueError(
                f"Biến '{key}' không hợp lệ. Chỉ chấp nhận: {DECISION_VARS}"
            )

    # Kiểm tra hồ sơ gốc và tính nguy cơ trước khi đổi
    truoc = predict_risk(profile)

    # Tạo bản sao mới để không làm biến đổi hồ sơ gốc
    modified_profile = dict(profile)
    modified_profile.update(changes)

    # Tính nguy cơ sau khi đổi (profile_to_frame bên trong sẽ kiểm tra miền giá trị)
    sau = predict_risk(modified_profile)
    chenh_lech = float(sau - truoc)

    return {
        "truoc": float(truoc),
        "sau": float(sau),
        "chenh_lech": chenh_lech,
        "muc_giam": float(truoc - sau),
        "muc_truoc": risk_level(truoc),
        "muc_sau": risk_level(sau),
        "thay_doi": dict(changes),
    }


# ---------------------------------------------------------------------------
# 5. Sinh lời khuyên cá nhân hóa
# ---------------------------------------------------------------------------
def recommend(profile: dict) -> list[dict]:
    """
    Trả về danh sách khuyến nghị dạng list[dict], mỗi phần tử có:
        - 'hanh_dong': hành động cụ thể cần làm
        - 'ly_do': lý do dựa trên EDA và mô hình (ghi rõ khuyến nghị chính/phụ)
    Đảm bảo:
        - Giảm số phút dùng máy là khuyến nghị chính.
        - Các yếu tố khác là khuyến nghị phụ, ghi rõ tác dụng nhỏ.
        - Hồ sơ nặng có ít nhất 2 khuyến nghị; hồ sơ nhẹ không bị khuyên vô lý.
    """
    # Kiểm tra tính hợp lệ của hồ sơ
    _ = profile_to_frame(profile)
    current_risk = predict_risk(profile)
    level = risk_level(current_risk)

    mins = float(profile["bedtime_phone_minutes"])
    brightness = float(profile["screen_brightness_pct"])
    caffeine = float(profile["caffeine_post_5pm_mg"])
    activity = float(profile["physical_activity_min"])
    blue_filter = int(profile["blue_light_filter_active"])
    app = str(profile["primary_bedtime_app"])
    chronotype = str(profile["chronotype"])
    occupation = str(profile["occupation_type"])

    recommendations: list[dict] = []

    # 1. KHUYẾN NGHỊ CHÍNH: Giảm số phút dùng điện thoại trước khi ngủ
    if mins > 30:
        target_mins = 30 if mins <= 90 else max(30, int(mins - 60))
        sim = what_if(profile, {"bedtime_phone_minutes": target_mins})
        drop_pct = max(0.0, (sim["truoc"] - sim["sau"]) * 100.0)
        recommendations.append(
            {
                "hanh_dong": (
                    f"[Khuyến nghị chính] Giảm thời gian dùng điện thoại trước khi ngủ "
                    f"từ {int(mins)} phút xuống còn khoảng {target_mins} phút."
                ),
                "ly_do": (
                    f"Theo phân tích dữ liệu (EDA) và độ quan trọng đặc trưng, số phút dùng "
                    f"máy trước khi ngủ là đòn bẩy hành vi mạnh nhất gây nợ ngủ (tỉ lệ nợ ngủ "
                    f"tăng từ ~36% ở nhóm 0–30 phút lên >90% khi dùng quá 90 phút). "
                    f"Nếu giảm xuống {target_mins} phút, nguy cơ dự kiến giảm khoảng "
                    f"{drop_pct:.1f} điểm phần trăm (từ {sim['truoc']*100:.1f}% xuống {sim['sau']*100:.1f}%)."
                ),
            }
        )

    # 2. KHUYẾN NGHỊ PHỤ 1: Bật bộ lọc ánh sáng xanh (chỉ khuyên khi có dùng máy > 15 phút)
    if mins > 15 and blue_filter == 0:
        recommendations.append(
            {
                "hanh_dong": (
                    "[Khuyến nghị phụ] Bật bộ lọc ánh sáng xanh (Night Shift / Eye Comfort) "
                    "trên điện thoại vào buổi tối."
                ),
                "ly_do": (
                    "Giúp giảm kích thích thị giác ban đêm. Lưu ý: Theo dữ liệu EDA, đây là "
                    "khuyến nghị phụ có tác dụng nhỏ lên xác suất nợ ngủ, cần kết hợp với "
                    "việc giảm số phút dùng máy để đạt hiệu quả rõ rệt."
                ),
            }
        )

    # 3. KHUYẾN NGHỊ PHỤ 2: Đổi ứng dụng dùng trước khi ngủ
    if mins > 15 and app in HIGH_STIMULATION_APPS:
        recommendations.append(
            {
                "hanh_dong": (
                    f"[Khuyến nghị phụ] Chuyển từ ứng dụng '{app}' sang hoạt động ít kích thích "
                    f"hơn như đọc tin tức/sách ('News / Reading') hoặc nhắn tin ngắn."
                ),
                "ly_do": (
                    "Các ứng dụng video ngắn và mạng xã hội có cơ chế cuộn vô tận làm người dùng "
                    "khó dừng lại đúng giờ. Tuy nhiên, theo dữ liệu, việc chỉ đổi loại ứng dụng "
                    "mà giữ nguyên số phút nhìn màn hình chỉ có tác dụng nhỏ."
                ),
            }
        )

    # 4. KHUYẾN NGHỊ PHỤ 3: Hạ độ sáng màn hình
    if mins > 15 and brightness > 50:
        recommendations.append(
            {
                "hanh_dong": (
                    f"[Khuyến nghị phụ] Hạ độ sáng màn hình từ {int(brightness)}% xuống mức 20%–30%."
                ),
                "ly_do": (
                    "Độ sáng màn hình cao làm tăng sự tỉnh táo trước giờ ngủ. Trong mô hình, "
                    "đây là yếu tố phụ có tác dụng nhỏ, đóng vai trò hỗ trợ thêm cho việc giảm "
                    "thời gian dùng máy."
                ),
            }
        )

    # 5. KHUYẾN NGHỊ PHỤ 4: Giảm lượng caffeine sau 17h
    if caffeine > 50:
        recommendations.append(
            {
                "hanh_dong": (
                    f"[Khuyến nghị phụ] Giảm lượng caffeine tiêu thụ sau 17h từ {int(caffeine)} mg "
                    f"xuống 0–50 mg."
                ),
                "ly_do": (
                    "Caffeine buổi tối làm kéo dài thời gian vào giấc. Trong tập dữ liệu này, "
                    "caffeine có mức ảnh hưởng nhỏ hơn nhiều so với số phút dùng điện thoại, "
                    "nhưng vẫn nên hạn chế để cải thiện chất lượng ngủ."
                ),
            }
        )

    # 6. KHUYẾN NGHỊ PHỤ 5: Tăng vận động thể chất ban ngày
    if activity < 20 and (level != "Thấp" or mins > 45):
        recommendations.append(
            {
                "hanh_dong": (
                    f"[Khuyến nghị phụ] Tăng thời gian vận động thể chất trong ngày từ "
                    f"{int(activity)} phút lên ít nhất 30 phút."
                ),
                "ly_do": (
                    "Vận động ban ngày hỗ trợ điều hòa nhịp sinh học. Theo phân tích đặc trưng, "
                    "biến này có tác dụng nhỏ trong mô hình dự đoán nhưng tốt cho sức khỏe giấc ngủ tổng thể."
                ),
            }
        )

    # Đảm bảo hồ sơ nguy cơ Cao hoặc dùng máy rất nhiều luôn có ít nhất 2 khuyến nghị
    if (level == "Cao" or mins >= 90) and len(recommendations) < 2:
        if mins > 15:
            step_target = 15
            sim_strict = what_if(profile, {"bedtime_phone_minutes": step_target})
            recommendations.append(
                {
                    "hanh_dong": (
                        f"[Khuyến nghị bổ sung] Đặt giới hạn tắt màn hình hoàn toàn 30–45 phút "
                        f"trước giờ ngủ (đưa thời lượng dùng máy về mức tối thiểu {step_target} phút)."
                    ),
                    "ly_do": (
                        f"Hồ sơ hiện tại có mức nguy cơ nợ ngủ cao ({current_risk*100:.1f}%). "
                        f"Cắt giảm triệt để thời gian dùng điện thoại xuống {step_target} phút "
                        f"giúp đưa nguy cơ về mức {sim_strict['sau']*100:.1f}%."
                    ),
                }
            )
        if len(recommendations) < 2 and (
            chronotype == "Night Owl" or occupation == "Healthcare / Shift Worker"
        ):
            recommendations.append(
                {
                    "hanh_dong": (
                        "[Lưu ý cá nhân hóa] Duy trì giờ đi ngủ - thức dậy cố định và vệ sinh giấc ngủ "
                        "nghiêm ngặt do đặc thù nhịp sinh học / ca làm việc."
                    ),
                    "ly_do": (
                        f"Dữ liệu cho thấy nhóm '{chronotype}' hoặc làm việc '{occupation}' có nguy cơ "
                        f"nền rất cao (87%–96%). Giảm dùng điện thoại giúp giảm nguy cơ đáng kể nhưng "
                        f"cần kết hợp ổn định lịch sinh hoạt lâu dài."
                    ),
                }
            )

    return recommendations


# ---------------------------------------------------------------------------
# 6. Xếp hạng phương án hành động (rank_plans)
# ---------------------------------------------------------------------------
def _compute_effort(profile: dict, changes: dict) -> float:
    """Tính điểm nỗ lực (càng nhỏ càng dễ thực hiện) để tìm phương án ít nỗ lực nhất."""
    effort = 0.0
    if "bedtime_phone_minutes" in changes:
        delta_min = max(
            0.0, float(profile["bedtime_phone_minutes"]) - float(changes["bedtime_phone_minutes"])
        )
        effort += (delta_min / 15.0) * 1.0
    if "blue_light_filter_active" in changes and int(changes["blue_light_filter_active"]) != int(
        profile["blue_light_filter_active"]
    ):
        effort += 0.2
    if "screen_brightness_pct" in changes:
        delta_b = max(
            0.0, float(profile["screen_brightness_pct"]) - float(changes["screen_brightness_pct"])
        )
        effort += (delta_b / 30.0) * 0.3
    if (
        "primary_bedtime_app" in changes
        and changes["primary_bedtime_app"] != profile["primary_bedtime_app"]
    ):
        effort += 0.6
    if "caffeine_post_5pm_mg" in changes:
        delta_c = max(
            0.0, float(profile["caffeine_post_5pm_mg"]) - float(changes["caffeine_post_5pm_mg"])
        )
        effort += (delta_c / 100.0) * 0.5
    return round(effort, 3)


def _describe_changes(profile: dict, changes: dict) -> str:
    """Tạo mô tả tiếng Việt ngắn gọn, dễ đọc cho một tập thay đổi."""
    parts = []
    if "bedtime_phone_minutes" in changes:
        old_m = int(profile["bedtime_phone_minutes"])
        new_m = int(changes["bedtime_phone_minutes"])
        parts.append(f"Giảm phút dùng máy từ {old_m} xuống {new_m} phút (-{old_m - new_m} phút)")
    if "blue_light_filter_active" in changes:
        parts.append("Bật lọc ánh sáng xanh")
    if "screen_brightness_pct" in changes:
        parts.append(f"Hạ độ sáng xuống {int(changes['screen_brightness_pct'])}%")
    if "primary_bedtime_app" in changes:
        parts.append(f"Đổi app sang '{changes['primary_bedtime_app']}'")
    if "caffeine_post_5pm_mg" in changes:
        parts.append(f"Giảm caffeine sau 17h xuống {int(changes['caffeine_post_5pm_mg'])} mg")
    return " + ".join(parts) if parts else "Giữ nguyên thói quen hiện tại"


def rank_plans(profile: dict, top_k: int = 3) -> list[dict]:
    """
    Thử lưới các thay đổi khả thi trên nhóm biến quyết định, lọc phương án hợp lệ,
    sắp xếp giảm dần theo mức giảm nguy cơ (muc_giam) và trả về top_k phương án.
    Đánh dấu phương án tốn ít nỗ lực nhất giúp đưa nguy cơ xuống dưới ngưỡng Trung bình (< 0.30)
    hoặc xuống mức Trung bình (< 0.60).
    """
    if top_k <= 0:
        raise ValueError("top_k phải là số nguyên dương.")

    base_risk = predict_risk(profile)
    cur_mins = int(profile["bedtime_phone_minutes"])
    cur_filter = int(profile["blue_light_filter_active"])
    cur_bright = int(profile["screen_brightness_pct"])
    cur_app = str(profile["primary_bedtime_app"])
    cur_caffeine = int(profile["caffeine_post_5pm_mg"])

    # Lưới lựa chọn cho từng biến quyết định
    min_options = [cur_mins]
    for step in (15, 30, 45, 60, 90):
        cand = max(0, cur_mins - step)
        if cand < cur_mins and cand not in min_options:
            min_options.append(cand)
    if cur_mins > 15 and 15 not in min_options:
        min_options.append(15)
    if cur_mins > 0 and 0 not in min_options:
        min_options.append(0)

    filter_options = [cur_filter] if cur_filter == 1 else [0, 1]
    bright_options = [cur_bright, 30] if cur_bright > 35 else [cur_bright]
    app_options = (
        [cur_app, "News / Reading"] if cur_app in HIGH_STIMULATION_APPS else [cur_app]
    )
    caffeine_options = [cur_caffeine, 0] if cur_caffeine > 50 else [cur_caffeine]

    candidates: list[dict] = []

    for m, flt, br, ap, caf in product(
        min_options, filter_options, bright_options, app_options, caffeine_options
    ):
        changes: dict[str, Any] = {}
        if m != cur_mins:
            changes["bedtime_phone_minutes"] = m
        if flt != cur_filter:
            changes["blue_light_filter_active"] = flt
        if br != cur_bright:
            changes["screen_brightness_pct"] = br
        if ap != cur_app:
            changes["primary_bedtime_app"] = ap
        if caf != cur_caffeine:
            changes["caffeine_post_5pm_mg"] = caf

        if not changes:
            continue

        sim = what_if(profile, changes)
        muc_giam = float(sim["truoc"] - sim["sau"])
        effort = _compute_effort(profile, changes)

        candidates.append(
            {
                "phuong_an": _describe_changes(profile, changes),
                "thay_doi": changes,
                "truoc": sim["truoc"],
                "sau": sim["sau"],
                "chenh_lech": sim["chenh_lech"],
                "muc_giam": muc_giam,
                "muc_truoc": sim["muc_truoc"],
                "muc_sau": sim["muc_sau"],
                "no_luc": effort,
                "xuong_duoi_trung_binh": bool(sim["sau"] < LOW_RISK_THRESHOLD),
                "xuong_muc_trung_binh_hoac_thap": bool(sim["sau"] < HIGH_RISK_THRESHOLD),
                "it_no_luc_nhat": False,
            }
        )

    if not candidates:
        return []

    # Tìm phương án ít nỗ lực nhất đạt mục tiêu giảm nguy cơ xuống dưới ngưỡng Trung bình (< 0.30)
    # Nếu không có (ví dụ Night Owl), xét phương án ít nỗ lực nhất xuống dưới ngưỡng Cao (< 0.60)
    low_target_plans = [c for c in candidates if c["xuong_duoi_trung_binh"] and c["muc_giam"] > 0]
    med_target_plans = [
        c for c in candidates if c["xuong_muc_trung_binh_hoac_thap"] and c["muc_giam"] > 0
    ]

    best_min_effort_plan = None
    if low_target_plans:
        best_min_effort_plan = min(low_target_plans, key=lambda x: (x["no_luc"], x["sau"]))
        best_min_effort_plan["it_no_luc_nhat"] = True
    elif med_target_plans:
        best_min_effort_plan = min(med_target_plans, key=lambda x: (x["no_luc"], x["sau"]))
        best_min_effort_plan["it_no_luc_nhat"] = True

    # Ưu tiên các phương án thực sự giúp giảm nguy cơ (muc_giam > 0)
    positive_plans = [c for c in candidates if c["muc_giam"] > 0]
    pool = positive_plans if positive_plans else candidates

    # Sắp xếp giảm dần theo mức giảm nguy cơ (muc_giam), nếu bằng nhau thì ưu tiên ít nỗ lực hơn
    pool.sort(key=lambda x: (-round(x["muc_giam"], 6), x["no_luc"]))

    top_plans = pool[:top_k]

    # Nếu có phương án ít nỗ lực nhất mà chưa nằm trong top_k, gắn thông tin tham chiếu
    for idx, item in enumerate(top_plans, start=1):
        item["hang"] = idx
        item["phuong_an_it_no_luc_nhat_toan_bo"] = (
            best_min_effort_plan["phuong_an"] if best_min_effort_plan else None
        )

    # Đảm bảo thứ tự trả về luôn xếp đúng theo mức giảm nguy cơ từ cao xuống thấp
    top_plans.sort(key=lambda x: (-round(x["muc_giam"], 6), x["no_luc"]))
    for idx, item in enumerate(top_plans, start=1):
        item["hang"] = idx

    return top_plans


# ---------------------------------------------------------------------------
# Chạy kiểm tra nhanh khi gọi trực tiếp: python src/decision.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_heavy = {
        "bedtime_phone_minutes": 120,
        "screen_brightness_pct": 85,
        "caffeine_post_5pm_mg": 150,
        "physical_activity_min": 10,
        "blue_light_filter_active": 0,
        "primary_bedtime_app": "TikTok / Reels",
        "age": 22,
        "gender": "Male",
        "occupation_type": "Student",
        "chronotype": "Intermediate",
    }
    p = predict_risk(sample_heavy)
    print(f"Nguy cơ ban đầu: {p:.3f} ({risk_level(p)})")
    print("\nKhuyến nghị:")
    for r in recommend(sample_heavy):
        print(" -", r["hanh_dong"])
    print("\nWhat-if (giảm còn 30 phút, bật lọc ánh sáng xanh):")
    print(what_if(sample_heavy, {"bedtime_phone_minutes": 30, "blue_light_filter_active": 1}))
    print("\nTop 3 phương án:")
    for plan in rank_plans(sample_heavy, top_k=3):
        print(f" #{plan['hang']}: {plan['phuong_an']} -> sau={plan['sau']:.3f} (giảm {plan['muc_giam']:.3f})")