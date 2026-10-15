"""Train, cross-validate, evaluate, and persist the Sleep DSS models."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

if __package__:
    from .preprocess import build_preprocessor, prepare_data
else:
    from preprocess import build_preprocessor, prepare_data

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
TEST_METRIC_COLUMNS = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
CV_METRIC_COLUMNS = ["model"] + [
    f"{metric}_{stat}"
    for metric in SCORING
    for stat in ("mean", "std")
]


def make_pipeline(classifier) -> Pipeline:
    """Build a pipeline so preprocessing is fitted inside each CV fold."""
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            ("model", classifier),
        ]
    )


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


def train_and_evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune on the training split, evaluate once on the held-out test split."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_data(random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidate_specs = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 12],
                "model__min_samples_leaf": [1, 3],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "XGBoost": (
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
        ),
    }

    fitted_models: dict[str, Pipeline] = {}
    cv_rows = []

    baseline_name = "Most frequent"
    baseline = make_pipeline(DummyClassifier(strategy="most_frequent"))
    baseline_scores = cross_validate(
        baseline,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
        error_score="raise",
    )
    baseline.fit(X_train, y_train)
    fitted_models[baseline_name] = baseline
    cv_rows.append(
        _cv_summary(
            baseline_name,
            {metric: baseline_scores[f"test_{metric}"].mean() for metric in SCORING},
            {metric: baseline_scores[f"test_{metric}"].std(ddof=0) for metric in SCORING},
        )
    )

    for model_name, (classifier, param_grid) in candidate_specs.items():
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

        best_index = search.best_index_
        cv_rows.append(
            _cv_summary(
                model_name,
                {
                    metric: search.cv_results_[f"mean_test_{metric}"][best_index]
                    for metric in SCORING
                },
                {
                    metric: search.cv_results_[f"std_test_{metric}"][best_index]
                    for metric in SCORING
                },
            )
        )
        print(f"{model_name} best params: {search.best_params_}")

    cv_metrics = pd.DataFrame(cv_rows, columns=CV_METRIC_COLUMNS)
    cv_metrics.to_csv(MODELS_DIR / "cv_metrics.csv", index=False)

    # Choose the final deployable model using training-only CV results; never use
    # the held-out test scores to select the model.
    cv_auc = cv_metrics.set_index("model")["roc_auc_mean"].drop(baseline_name)
    selected_name = cv_auc.idxmax()
    selected_model = fitted_models[selected_name]
    joblib.dump(selected_model, MODELS_DIR / "model.joblib")

    metric_rows = []
    confusion_rows = []
    for model_name, model in fitted_models.items():
        metrics, matrix = _evaluate(model_name, model, X_test, y_test)
        metric_rows.append(metrics)
        confusion_rows.append(matrix)

    metrics = pd.DataFrame(metric_rows, columns=TEST_METRIC_COLUMNS)
    metrics.to_csv(MODELS_DIR / "metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(
        MODELS_DIR / "confusion_matrices.csv",
        index=False,
    )

    print(f"Training rows: {len(X_train)}; test rows: {len(X_test)}")
    print(f"Selected model by mean 5-fold CV ROC-AUC: {selected_name}")
    print("\n5-fold CV mean and standard deviation:")
    print(cv_metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nHeld-out test metrics:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved final pipeline to {MODELS_DIR / 'model.joblib'}")
    print(f"Saved test metrics to {MODELS_DIR / 'metrics.csv'}")
    print(f"Saved CV metrics to {MODELS_DIR / 'cv_metrics.csv'}")
    print(f"Saved confusion matrices to {MODELS_DIR / 'confusion_matrices.csv'}")
    return metrics, cv_metrics


if __name__ == "__main__":
    train_and_evaluate()