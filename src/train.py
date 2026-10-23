
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # vẽ hình không cần màn hình
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # máy chưa cài xgboost: bỏ qua mô hình này
    XGBClassifier = None

try:
    import shap
except ImportError:  # máy chưa cài shap: dùng permutation importance
    shap = None

if __package__:
    from .preprocess import build_preprocessor, prepare_data
else:
    from preprocess import build_preprocessor, prepare_data

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "figures"
RANDOM_STATE = 42

# Mô hình đưa vào ứng dụng. Chọn Logistic Regression vì:
#   - điểm CV ngang các mô hình mạnh nhất (chênh lệch nhỏ hơn độ lệch chuẩn giữa các fold),
#   - dễ giải thích (hệ số theo từng đặc trưng),
#   - xác suất thay đổi mượt theo số phút, phù hợp cho what-if và xếp hạng phương án.
# Đặt DEPLOY_MODEL = None để tự chọn mô hình có ROC-AUC CV cao nhất.
DEPLOY_MODEL: str | None = "Logistic Regression"

# Các ngưỡng xác suất được phân tích (lớp quyết định đang dùng 0,30 và 0,60)
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
TEST_METRIC_COLUMNS = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
CV_METRIC_COLUMNS = ["model"] + [
    f"{metric}_{stat}" for metric in SCORING for stat in ("mean", "std")
]


# ---------------------------------------------------------------------------
# 1. Các hàm tiện ích
# ---------------------------------------------------------------------------
def make_pipeline(classifier, *, include_leakage: bool = False) -> Pipeline:
    """Pipeline = tiền xử lý + mô hình, để tiền xử lý được fit lại trong từng fold CV."""
    return Pipeline(
        [
            ("prep", build_preprocessor(include_leakage=include_leakage)),
            ("model", classifier),
        ]
    )


def candidate_specs() -> dict:
    """Các mô hình ứng viên và lưới tham số cho GridSearchCV."""
    specs = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [15, 31, 51],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "model__max_depth": [4, 6, 8],
                "model__min_samples_leaf": [10, 30],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 12],
                "model__min_samples_leaf": [1, 3],
                "model__class_weight": [None, "balanced"],
            },
        ),
    }
    if XGBClassifier is not None:
        specs["XGBoost"] = (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
            },
        )
    else:
        print("Chưa cài xgboost: bỏ qua mô hình XGBoost.")
    specs["ANN (MLP)"] = (
        MLPClassifier(max_iter=500, early_stopping=True, random_state=RANDOM_STATE),
        {
            "model__hidden_layer_sizes": [(32,), (64, 32)],
            "model__alpha": [1e-4, 1e-3],
        },
    )
    return specs


def _cv_summary(model_name: str, means: dict, stds: dict) -> dict:
    summary = {"model": model_name}
    for metric in SCORING:
        summary[f"{metric}_mean"] = means[metric]
        summary[f"{metric}_std"] = stds[metric]
    return summary


def _evaluate(model_name: str, model: Pipeline, X_test, y_test) -> tuple[dict, dict]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    matrix = {"model": model_name, "tn": tn, "fp": fp, "fn": fn, "tp": tp}
    return metrics, matrix


# ---------------------------------------------------------------------------
# 2. Độ quan trọng đặc trưng
# ---------------------------------------------------------------------------
def feature_importance(model: Pipeline, X_train, X_test, y_test) -> pd.DataFrame:
    """
    Trả về bảng (feature, importance, method), sắp xếp giảm dần.

    - Logistic Regression: SHAP LinearExplainer (trên đặc trưng sau mã hóa).
    - Mô hình cây (Decision Tree, Random Forest, XGBoost): SHAP TreeExplainer.
    - KNN, MLP, hoặc khi chưa cài shap: permutation importance trên cột thô,
      đo mức giảm ROC-AUC trên tập test khi xáo trộn từng cột.
    """
    preprocessor = model.named_steps["prep"]
    classifier = model.named_steps["model"]
    is_tree = isinstance(classifier, (DecisionTreeClassifier, RandomForestClassifier)) or (
        XGBClassifier is not None and isinstance(classifier, XGBClassifier)
    )

    if shap is not None and (isinstance(classifier, LogisticRegression) or is_tree):
        transformed = preprocessor.transform(X_train)
        if isinstance(classifier, LogisticRegression):
            explainer = shap.LinearExplainer(classifier, transformed)
        else:
            explainer = shap.TreeExplainer(classifier)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):  # bản shap cũ: danh sách theo lớp
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:  # (mẫu, đặc trưng, lớp): lấy lớp 1 = nợ ngủ đáng kể
            values = values[:, :, 1]
        return pd.DataFrame(
            {
                "feature": preprocessor.get_feature_names_out(),
                "importance": np.abs(values).mean(axis=0),
                "method": "mean |SHAP|",
            }
        ).sort_values("importance", ascending=False)

    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE
    )
    return pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": result.importances_mean,
            "method": "permutation (giảm ROC-AUC)",
        }
    ).sort_values("importance", ascending=False)


# ---------------------------------------------------------------------------
# 3. Phân tích ngưỡng và phân tích lỗi (cho báo cáo và vấn đáp)
# ---------------------------------------------------------------------------
def threshold_analysis(model: Pipeline, X_test, y_test) -> pd.DataFrame:
    """Precision, Recall, F1 và tỉ lệ bị cảnh báo ở từng ngưỡng xác suất."""
    proba = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in THRESHOLDS:
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "ti_le_bi_canh_bao": pred.mean(),
            }
        )
    return pd.DataFrame(rows)


def error_analysis(model: Pipeline, X_test, y_test) -> pd.DataFrame:
    """Tỉ lệ dự đoán sai theo nhóm: nhịp sinh học, nghề nghiệp, khoảng số phút dùng máy."""
    df = X_test.copy()
    df["y_true"] = np.asarray(y_test)
    df["y_pred"] = model.predict(X_test)
    df["sai"] = (df["y_true"] != df["y_pred"]).astype(int)
    df["khoang_phut"] = pd.cut(
        df["bedtime_phone_minutes"],
        bins=[-1, 30, 60, 90, 120, 1440],
        labels=["0-30", "31-60", "61-90", "91-120", ">120"],
    )
    rows = []
    for col in ["chronotype", "occupation_type", "khoang_phut"]:
        grouped = df.groupby(col, observed=True).agg(
            so_mau=("sai", "size"),
            ti_le_sai=("sai", "mean"),
            ti_le_no_ngu_thuc_te=("y_true", "mean"),
        )
        for value, r in grouped.iterrows():
            rows.append({"nhom": col, "gia_tri": str(value), **r.to_dict()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Hình cho báo cáo
# ---------------------------------------------------------------------------
def save_figures(fitted_models: dict, deployed_name: str, X_test, y_test, fi: pd.DataFrame) -> None:
    deployed = fitted_models[deployed_name]
    real_models = {k: v for k, v in fitted_models.items() if k != "Most frequent"}

    # ROC của tất cả mô hình trên cùng một hình
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in real_models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Đoán ngẫu nhiên")
    ax.set_title("Đường ROC trên tập kiểm tra")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curves.png", dpi=200)
    plt.close(fig)

    # Ma trận nhầm lẫn của từng mô hình
    n = len(real_models)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, model) in zip(axes, real_models.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, display_labels=["Không", "Nợ ngủ"],
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(name, fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrices.png", dpi=200)
    plt.close(fig)

    # Độ quan trọng đặc trưng (top 10) của mô hình triển khai
    top = fi.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"], color="#4C72B0")
    ax.set_xlabel(top["method"].iloc[0])
    ax.set_title(f"10 đặc trưng quan trọng nhất - {deployed_name}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=200)
    plt.close(fig)

    # Đường hiệu chỉnh xác suất của mô hình triển khai
    fig, ax = plt.subplots(figsize=(7, 6))
    CalibrationDisplay.from_estimator(deployed, X_test, y_test, n_bins=10, strategy="uniform", ax=ax)
    ax.set_title(f"Đường hiệu chỉnh xác suất - {deployed_name}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "calibration_curve.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Quy trình chính
# ---------------------------------------------------------------------------
def train_and_evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tinh chỉnh trên tập train, đánh giá một lần trên tập test, lưu mọi kết quả."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_data(random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fitted_models: dict[str, Pipeline] = {}
    cv_rows = []

    # Baseline: luôn đoán lớp phổ biến nhất
    baseline_name = "Most frequent"
    baseline = make_pipeline(DummyClassifier(strategy="most_frequent"))
    baseline_scores = cross_validate(
        baseline, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1, error_score="raise"
    )
    baseline.fit(X_train, y_train)
    fitted_models[baseline_name] = baseline
    cv_rows.append(
        _cv_summary(
            baseline_name,
            {m: baseline_scores[f"test_{m}"].mean() for m in SCORING},
            {m: baseline_scores[f"test_{m}"].std(ddof=0) for m in SCORING},
        )
    )

    best_params_rows = []
    for model_name, (classifier, param_grid) in candidate_specs().items():
        print(f"Đang tinh chỉnh {model_name}...")
        search = GridSearchCV(
            estimator=make_pipeline(classifier),
            param_grid=param_grid,
            scoring=SCORING,
            refit="roc_auc",
            cv=cv,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        fitted_models[model_name] = search.best_estimator_
        best = search.best_index_
        cv_rows.append(
            _cv_summary(
                model_name,
                {m: search.cv_results_[f"mean_test_{m}"][best] for m in SCORING},
                {m: search.cv_results_[f"std_test_{m}"][best] for m in SCORING},
            )
        )
        best_params_rows.append({"model": model_name, "best_params": str(search.best_params_)})
        print(f"  tham số tốt nhất: {search.best_params_}")

    cv_metrics = pd.DataFrame(cv_rows, columns=CV_METRIC_COLUMNS)
    cv_metrics.to_csv(MODELS_DIR / "cv_metrics.csv", index=False)
    pd.DataFrame(best_params_rows).to_csv(MODELS_DIR / "best_params.csv", index=False)

    # Chọn mô hình: chỉ dựa trên CV của tập train, không bao giờ dựa trên tập test
    cv_auc = cv_metrics.set_index("model")["roc_auc_mean"].drop(baseline_name)
    best_by_cv = cv_auc.idxmax()
    if DEPLOY_MODEL is not None and DEPLOY_MODEL in fitted_models:
        deployed_name = DEPLOY_MODEL
    else:
        deployed_name = best_by_cv
    deployed_model = fitted_models[deployed_name]

    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(deployed_model, model_path)
    persisted = joblib.load(model_path)
    if not isinstance(persisted, Pipeline) or not callable(getattr(persisted, "predict_proba", None)):
        raise TypeError(f"Model lưu tại {model_path} phải là một Pipeline có predict_proba.")

    # Đánh giá một lần trên tập test
    metric_rows, confusion_rows = [], []
    for model_name, model in fitted_models.items():
        m, cm = _evaluate(model_name, model, X_test, y_test)
        metric_rows.append(m)
        confusion_rows.append(cm)
    metrics = pd.DataFrame(metric_rows, columns=TEST_METRIC_COLUMNS)
    metrics.to_csv(MODELS_DIR / "metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(MODELS_DIR / "confusion_matrices.csv", index=False)

    # Thí nghiệm rò rỉ dữ liệu: cùng một mô hình, có và không có cột kết quả giấc ngủ
    leakage_rows = []
    for include_leakage in (False, True):
        lx_train, lx_test, ly_train, ly_test = prepare_data(
            include_leakage=include_leakage, random_state=RANDOM_STATE
        )
        leak_model = make_pipeline(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
            include_leakage=include_leakage,
        )
        leak_model.fit(lx_train, ly_train)
        lm, _ = _evaluate("Logistic Regression", leak_model, lx_test, ly_test)
        leakage_rows.append({"include_leakage": include_leakage, **lm})
    leakage_df = pd.DataFrame(leakage_rows)
    leakage_df.to_csv(MODELS_DIR / "leakage_comparison.csv", index=False)

    # Độ quan trọng đặc trưng, ngưỡng, lỗi, hình
    fi = feature_importance(deployed_model, X_train, X_test, y_test)
    fi.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    thr = threshold_analysis(deployed_model, X_test, y_test)
    thr.to_csv(MODELS_DIR / "threshold_analysis.csv", index=False)
    err = error_analysis(deployed_model, X_test, y_test)
    err.to_csv(MODELS_DIR / "error_analysis.csv", index=False)
    save_figures(fitted_models, deployed_name, X_test, y_test, fi)

    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    print(f"\nSố dòng train: {len(X_train)}; test: {len(X_test)}")
    print(f"Mô hình có ROC-AUC CV cao nhất: {best_by_cv}")
    print(f"Mô hình triển khai (model.joblib): {deployed_name}")
    print("\nCV 5-fold (trung bình và độ lệch chuẩn):")
    print(cv_metrics.to_string(index=False, float_format=fmt))
    print("\nKết quả trên tập test:")
    print(metrics.to_string(index=False, float_format=fmt))
    print("\nThí nghiệm rò rỉ dữ liệu:")
    print(leakage_df.to_string(index=False, float_format=fmt))
    print(f"\nPhân tích ngưỡng ({deployed_name}):")
    print(thr.to_string(index=False, float_format=fmt))
    print(f"\nĐã lưu bảng vào {MODELS_DIR} và hình vào {REPORTS_DIR}")
    return metrics, cv_metrics


if __name__ == "__main__":
    train_and_evaluate()
