"""
src/preprocess.py - Tiền xử lý dữ liệu cho dự án DSS (người phụ trách: Dương)

Đây là NƠI DUY NHẤT định nghĩa tên cột và danh sách đặc trưng. Các file khác
(train.py, decision.py, app.py) chỉ import từ đây, không tự gõ lại tên cột.

Hợp đồng chung của nhóm:
    FEATURES        toàn bộ cột đầu vào của mô hình
    DECISION_VARS   biến quyết định (người dùng đổi được)
    CONTEXT_VARS    biến ngữ cảnh (không đổi được)
    TARGET          tên cột nhãn nhị phân (1 = nợ ngủ đáng kể)

Dùng nhanh:
    from src.preprocess import prepare_data, build_preprocessor
    X_train, X_test, y_train, y_test = prepare_data()

Chạy trực tiếp để làm sạch và lưu data/clean/:
    python src/preprocess.py
"""
from __future__ import annotations

import numbers
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 1. Đường dẫn
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "bedtime_screentime_sleep_debt.csv"
CLEAN_PATH = ROOT / "data" / "clean" / "sleep_clean.csv"

# ---------------------------------------------------------------------------
# 2. Nhãn
# ---------------------------------------------------------------------------
LABEL_COL = "sleep_debt_category"  # nhãn gốc, 4 mức
LABEL_ORDER = ["Optimal Recovery", "Mild Deficit", "Moderate Debt", "Severe Sleep Debt"]
HIGH_DEBT_LEVELS = ["Moderate Debt", "Severe Sleep Debt"]  # được gán nhãn 1
TARGET = "high_sleep_debt"  # nhãn nhị phân do make_target() tạo ra

ID_COL = "user_id"  # chỉ là mã định danh, không đưa vào mô hình

# ---------------------------------------------------------------------------
# 3. Biến đầu vào
# ---------------------------------------------------------------------------
# Biến quyết định: người dùng thay đổi được (what-if chỉ đổi nhóm này)
DECISION_NUMERIC = [
    "bedtime_phone_minutes",
    "screen_brightness_pct",
    "caffeine_post_5pm_mg",
    "physical_activity_min",
]
DECISION_BINARY = ["blue_light_filter_active"]
DECISION_CATEGORICAL = ["primary_bedtime_app"]
DECISION_VARS = DECISION_NUMERIC + DECISION_BINARY + DECISION_CATEGORICAL

# Biến ngữ cảnh: không đổi được
CONTEXT_NUMERIC = ["age"]
CONTEXT_CATEGORICAL = ["gender", "occupation_type"]
CONTEXT_ORDINAL = ["chronotype"]
CONTEXT_VARS = CONTEXT_NUMERIC + CONTEXT_CATEGORICAL + CONTEXT_ORDINAL

# Toàn bộ đầu vào của mô hình
FEATURES = DECISION_VARS + CONTEXT_VARS

# Cột kết quả giấc ngủ: KHÔNG được đưa vào đầu vào (rò rỉ dữ liệu).
# Chỉ dùng cho EDA và thí nghiệm so sánh có/không rò rỉ (include_leakage=True).
LEAKAGE_COLS = [
    "sleep_latency_min",
    "total_sleep_hours",
    "deep_sleep_pct",
    "rem_sleep_pct",
    "morning_alarm_snoozes",
    "next_day_fatigue_score",
]

assert not set(FEATURES) & set(LEAKAGE_COLS), "FEATURES không được chứa cột rò rỉ"

# Giá trị hợp lệ của các biến phân loại (dùng cho form nhập và kiểm tra)
CHRONOTYPE_ORDER = ["Morning Lark", "Intermediate", "Night Owl"]
CATEGORY_VALUES = {
    "primary_bedtime_app": [
        "TikTok / Reels",
        "Instagram / Reddit",
        "YouTube",
        "Streaming (Netflix/Hulu)",
        "Messaging / Chat",
        "News / Reading",
    ],
    "gender": ["Female", "Male", "Non-Binary"],
    "occupation_type": [
        "Student",
        "Corporate 9-to-5",
        "Remote Tech",
        "Freelance / Creative",
        "Healthcare / Shift Worker",
    ],
    "chronotype": CHRONOTYPE_ORDER,
}

# Khoảng giá trị trong dữ liệu (dùng cho thanh trượt giao diện)
NUMERIC_RANGES = {
    "age": (18, 65),
    "bedtime_phone_minutes": (0, 180),
    "screen_brightness_pct": (10, 100),
    "caffeine_post_5pm_mg": (0, 250),
    "physical_activity_min": (0, 112),
}

# Giới hạn cứng để kiểm tra đầu vào (ngoài khoảng này là dữ liệu vô lý)
VALID_LIMITS = {
    "age": (10, 100),
    "bedtime_phone_minutes": (0, 1440),
    "screen_brightness_pct": (0, 100),
    "caffeine_post_5pm_mg": (0, 1000),
    "physical_activity_min": (0, 1440),
}


# ---------------------------------------------------------------------------
# 4. Nạp, kiểm tra, làm sạch
# ---------------------------------------------------------------------------
def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Đọc file CSV gốc và kiểm tra có đủ cột cần thiết."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file dữ liệu: {path}")
    df = pd.read_csv(path)
    required = [LABEL_COL] + FEATURES + LEAKAGE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"File thiếu các cột: {missing}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch: bỏ khoảng trắng thừa, bỏ dòng trùng/thiếu, ép kiểu, kiểm tra nhãn."""
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) in ("str", "string"):
            df[col] = df[col].astype(str).str.strip()

    # trùng lặp: so sánh trên mọi cột trừ mã định danh
    cols = [c for c in df.columns if c != ID_COL]
    df = df.drop_duplicates(subset=cols)

    # thiếu giá trị ở các cột cần dùng
    needed = [LABEL_COL] + FEATURES + LEAKAGE_COLS
    df = df.dropna(subset=needed)

    bad = set(df[LABEL_COL].unique()) - set(LABEL_ORDER)
    if bad:
        raise ValueError(f"Nhãn lạ trong cột {LABEL_COL}: {bad}")

    df[DECISION_BINARY[0]] = df[DECISION_BINARY[0]].astype(int)
    return df.reset_index(drop=True)


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột nhãn nhị phân TARGET: 1 nếu Moderate Debt hoặc Severe Sleep Debt."""
    df = df.copy()
    df[TARGET] = df[LABEL_COL].isin(HIGH_DEBT_LEVELS).astype(int)
    return df


def load_dataset(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Nạp, làm sạch và thêm nhãn nhị phân. Trả về DataFrame đầy đủ cột."""
    return make_target(clean(load_raw(path)))


def save_clean(df: pd.DataFrame, path: str | Path = CLEAN_PATH) -> Path:
    """Lưu dữ liệu đã làm sạch (bỏ cột user_id) vào data/clean/."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[ID_COL], errors="ignore").to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 5. Tách đầu vào / nhãn và chia train/test
# ---------------------------------------------------------------------------
def get_xy(
    df: pd.DataFrame,
    multiclass: bool = False,
    include_leakage: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Tách X (đặc trưng) và y (nhãn).

    multiclass=False  -> y là TARGET nhị phân (mặc định, bản chính)
    multiclass=True   -> y là nhãn gốc 4 mức (bản mở rộng)
    include_leakage   -> thêm cột kết quả giấc ngủ vào X.
                         CHỈ dùng cho thí nghiệm minh họa rò rỉ dữ liệu.
    """
    if TARGET not in df.columns:
        df = make_target(df)
    cols = FEATURES + (LEAKAGE_COLS if include_leakage else [])
    X = df[cols].copy()
    y = df[LABEL_COL].copy() if multiclass else df[TARGET].copy()
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Chia train/test có phân tầng theo nhãn. Trả về X_train, X_test, y_train, y_test."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def prepare_data(
    path: str | Path = RAW_PATH,
    multiclass: bool = False,
    include_leakage: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Một lệnh từ file CSV gốc đến X_train, X_test, y_train, y_test."""
    df = load_dataset(path)
    X, y = get_xy(df, multiclass=multiclass, include_leakage=include_leakage)
    return split_data(X, y, test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# 6. Bộ tiền xử lý cho mô hình (đặt vào đầu Pipeline)
# ---------------------------------------------------------------------------
def build_preprocessor(
    scale: bool = True, include_leakage: bool = False
) -> ColumnTransformer:
    """
    Tạo ColumnTransformer nhận DataFrame có đúng các cột thô trong FEATURES.

    Cách dùng (train.py):
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("prep", build_preprocessor()), ("model", clf)])
        pipe.fit(X_train, y_train)

    Lưu cả `pipe` bằng joblib để app chỉ cần đưa DataFrame thô vào predict_proba.

    - Biến số: chuẩn hóa (tắt bằng scale=False nếu chỉ dùng mô hình cây)
    - Biến nhị phân: giữ nguyên
    - Biến phân loại: one-hot, giá trị lạ gặp lúc dự đoán sẽ bị bỏ qua
    - chronotype: mã hóa thứ tự Morning Lark < Intermediate < Night Owl
    """
    numeric = DECISION_NUMERIC + CONTEXT_NUMERIC
    if include_leakage:
        numeric = numeric + LEAKAGE_COLS

    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler() if scale else "passthrough", numeric),
            ("bin", "passthrough", DECISION_BINARY),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                DECISION_CATEGORICAL + CONTEXT_CATEGORICAL,
            ),
            (
                "ord",
                OrdinalEncoder(
                    categories=[CHRONOTYPE_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CONTEXT_ORDINAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    prep.set_output(transform="pandas")
    return prep


# ---------------------------------------------------------------------------
# 7. Chuyển hồ sơ người dùng thành DataFrame (cho decision.py và app.py)
# ---------------------------------------------------------------------------
def profile_to_frame(profile: dict) -> pd.DataFrame:
    """
    Biến một dict hồ sơ thành DataFrame một dòng đúng thứ tự cột FEATURES,
    sau khi kiểm tra giá trị. Dùng trước mọi lần gọi mô hình.

    Báo lỗi KeyError nếu thiếu cột, ValueError nếu giá trị vô lý.
    Giá trị nằm ngoài khoảng của dữ liệu nhưng vẫn hợp lý (ví dụ 240 phút)
    được chấp nhận; mô hình sẽ ngoại suy, nên khi hiển thị cần lưu ý.
    """
    missing = [c for c in FEATURES if c not in profile]
    if missing:
        raise KeyError(f"Hồ sơ thiếu các trường: {missing}")

    row = {c: profile[c] for c in FEATURES}

    for col, allowed in CATEGORY_VALUES.items():
        if row[col] not in allowed:
            raise ValueError(f"{col}='{row[col]}' không hợp lệ, chọn một trong {allowed}")

    for col, (lo, hi) in VALID_LIMITS.items():
        value = row[col]
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            raise ValueError(f"{col} phải là số, nhận được {value!r}")
        if not lo <= value <= hi:
            raise ValueError(f"{col}={value} ngoài giới hạn hợp lệ [{lo}, {hi}]")

    if row["blue_light_filter_active"] not in (0, 1, True, False):
        raise ValueError("blue_light_filter_active phải là 0 hoặc 1")
    row["blue_light_filter_active"] = int(row["blue_light_filter_active"])

    return pd.DataFrame([row], columns=FEATURES)


# ---------------------------------------------------------------------------
# 8. Chạy trực tiếp: làm sạch, in tóm tắt, lưu data/clean/
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_dataset()
    out = save_clean(df)
    print(f"Đã lưu {len(df)} dòng vào {out}")
    print(f"Số đặc trưng đầu vào: {len(FEATURES)} "
          f"({len(DECISION_VARS)} biến quyết định, {len(CONTEXT_VARS)} biến ngữ cảnh)")
    print("\nPhân phối nhãn gốc:")
    print(df[LABEL_COL].value_counts().reindex(LABEL_ORDER))
    print(f"\nTỉ lệ nhãn nhị phân ({TARGET}=1): {df[TARGET].mean():.3f}")


if __name__ == "__main__":
    main()
