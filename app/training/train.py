from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "features" / "match_features.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "latest"
DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "lol-game-prediction"
VALIDATION_START = "2025-06-01"
VALIDATION_END = "2025-12-31"
TEST_YEAR = 2026
TARGET_COLUMN = "blue_win"
METADATA_COLUMNS = {"match_id", "match_date"}
PARAM_GRID = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "num_leaves": [15, 32, 64, 128, 256],
    "n_estimators": [50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000],
    "max_depth": [6, 8, 10, 12, 16, 20, 24, 28, 32],
    "max_bin": [64, 128, 256],
}


def _feature_columns(features: pd.DataFrame) -> list[str]:
    return [column for column in features.columns if column not in METADATA_COLUMNS | {TARGET_COLUMN}]


def _split_train_validation_test(
    features: pd.DataFrame,
    validation_start: str = VALIDATION_START,
    validation_end: str = VALIDATION_END,
    test_year: int = TEST_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dated = features.copy()
    dated["match_date"] = pd.to_datetime(dated["match_date"])
    validation_start_date = pd.Timestamp(validation_start)
    validation_end_date = pd.Timestamp(validation_end)
    train = dated[dated["match_date"] < validation_start_date].copy()
    validation = dated[
        (dated["match_date"] >= validation_start_date) & (dated["match_date"] <= validation_end_date)
    ].copy()
    test = dated[dated["year"] == test_year].copy()
    if train.empty:
        raise ValueError(f"No training rows found before {validation_start}.")
    if validation.empty:
        raise ValueError(f"No validation rows found from {validation_start} to {validation_end}.")
    if test.empty:
        raise ValueError(f"No test rows found for {test_year}.")
    if train[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training data must contain both classes.")
    if validation[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Validation data must contain both classes.")
    return train, validation, test


def _categorical_columns(feature_frame: pd.DataFrame) -> list[str]:
    return feature_frame.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def _prepare_features(feature_frame: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    prepared = feature_frame.copy()
    for column in categorical_columns:
        prepared[column] = prepared[column].astype("category")
    return prepared


def _build_model(params: dict[str, Any] | None = None) -> LGBMClassifier:
    model = LGBMClassifier(objective="binary", random_state=42, verbosity=-1)
    if params:
        model.set_params(**params)
    return model


def _metrics(y_true: pd.Series, probabilities: pd.Series) -> dict[str, float | None]:
    predictions = probabilities >= 0.5
    metrics: dict[str, float | None] = {
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def _date_range(frame: pd.DataFrame) -> dict[str, str]:
    dates = pd.to_datetime(frame["match_date"])
    return {"start": dates.min().date().isoformat(), "end": dates.max().date().isoformat()}


def _log_mlflow_run(metadata: dict[str, Any], output_dir: Path, tracking_uri: str | None) -> str:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        metadata["mlflow_run_id"] = run.info.run_id
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        mlflow.log_params(
            {
                "test_year": metadata["test_year"],
                "train_rows": metadata["train_rows"],
                "validation_rows": metadata["validation_rows"],
                "test_rows": metadata["test_rows"],
                **metadata["best_params"],
            }
        )
        mlflow.log_metrics({key: value for key, value in metadata["metrics"].items() if value is not None})
        for artifact in ("model.joblib", "preprocessor.joblib", "feature_columns.json", "metadata.json"):
            mlflow.log_artifact(str(output_dir / artifact))
        return run.info.run_id


def train_model(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_year: int = TEST_YEAR,
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
) -> dict[str, Any]:
    features = pd.read_parquet(input_path)
    train, validation, test = _split_train_validation_test(features, test_year=test_year)
    feature_columns = _feature_columns(features)
    categorical_columns = _categorical_columns(train[feature_columns])
    x_train = _prepare_features(train[feature_columns], categorical_columns)
    y_train = train[TARGET_COLUMN].astype(int)
    x_validation = _prepare_features(validation[feature_columns], categorical_columns)
    y_validation = validation[TARGET_COLUMN].astype(int)
    x_test = _prepare_features(test[feature_columns], categorical_columns)
    y_test = test[TARGET_COLUMN].astype(int)

    tuning_results: list[dict[str, Any]] = []
    best_model: LGBMClassifier | None = None
    best_params: dict[str, Any] | None = None
    best_score: float | None = None
    parameter_grid = list(ParameterGrid(PARAM_GRID))
    for params in tqdm(parameter_grid, desc="Training LightGBM models", unit="model"):
        model = _build_model(params)
        model.fit(x_train, y_train, categorical_feature=categorical_columns)
        train_probabilities = model.predict_proba(x_train)[:, 1]
        validation_probabilities = model.predict_proba(x_validation)[:, 1]
        score = float(log_loss(y_validation, validation_probabilities, labels=[0, 1]))
        test_probabilities = model.predict_proba(x_test)[:, 1]
        tuning_results.append(
            {
                "params": params,
                "train_metrics": _metrics(y_train, pd.Series(train_probabilities)),
                "validation_metrics": _metrics(y_validation, pd.Series(validation_probabilities)),
                "test_metrics": _metrics(y_test, pd.Series(test_probabilities)),
            }
        )
        if best_score is None or score < best_score:
            best_score = score
            best_params = params
            best_model = model

    if best_model is None or best_params is None:
        raise RuntimeError("No LightGBM model was trained.")

    test_probabilities = best_model.predict_proba(x_test)[:, 1]
    metrics = _metrics(y_test, pd.Series(test_probabilities))
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "test_year": test_year,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_date_range": _date_range(train),
        "validation_date_range": _date_range(validation),
        "test_date_range": _date_range(test),
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "parameter_grid": PARAM_GRID,
        "best_params": best_params,
        "tuning_results": tuning_results,
        "metrics": metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output_dir / "model.joblib")
    joblib.dump({"categorical_columns": categorical_columns}, output_dir / "preprocessor.joblib")
    (output_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _log_mlflow_run(metadata, output_dir, mlflow_tracking_uri)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-year", type=int, default=TEST_YEAR)
    parser.add_argument("--mlflow-tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    args = parser.parse_args()

    metadata = train_model(args.input, args.output_dir, args.test_year, args.mlflow_tracking_uri)
    print(json.dumps({"metrics": metadata["metrics"], "best_params": metadata["best_params"]}, indent=2))


if __name__ == "__main__":
    main()
