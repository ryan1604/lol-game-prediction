from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from mlflow.tracking import MlflowClient

from app.training import train as train_module


def _feature_row(match_id: int, match_date: str, blue_win: bool) -> dict[str, object]:
    year = int(match_date[:4])
    return {
        "match_id": match_id,
        "match_date": match_date,
        "year": year,
        "region": "LCK",
        "split": "Spring",
        "stage": "WEEK1",
        "blue_team_name": "AAA" if match_id % 2 else "BBB",
        "red_team_name": "CCC" if match_id % 2 else "DDD",
        "blue_has_first_pick": match_id % 2 == 0,
        "is_international": False,
        "blue_team_split_games_before": match_id,
        "red_team_split_games_before": match_id + 1,
        "team_split_win_rate_diff": 0.2 if blue_win else -0.2,
        "blue_win": blue_win,
    }


def test_train_model_writes_artifacts_and_uses_2026_test_set(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "match_features.parquet"
    output_dir = tmp_path / "model"
    mlflow_tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setattr(
        train_module,
        "PARAM_GRID",
        {"learning_rate": [0.03], "num_leaves": [15], "n_estimators": [50]},
    )
    rows = [
        _feature_row(1, "2024-01-01", True),
        _feature_row(2, "2024-01-02", False),
        _feature_row(3, "2025-01-01", True),
        _feature_row(4, "2025-01-02", False),
        _feature_row(5, "2025-06-01", True),
        _feature_row(6, "2025-12-31", False),
        _feature_row(7, "2026-01-01", True),
        _feature_row(8, "2026-01-02", False),
    ]
    pd.DataFrame(rows).to_parquet(input_path, index=False)

    metadata = train_module.train_model(input_path, output_dir, mlflow_tracking_uri=mlflow_tracking_uri)

    saved_metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(output_dir / "model.joblib")
    preprocessor = joblib.load(output_dir / "preprocessor.joblib")
    feature_columns = json.loads((output_dir / "feature_columns.json").read_text(encoding="utf-8"))
    x_test = pd.DataFrame(rows).loc[6:, feature_columns].copy()
    for column in preprocessor["categorical_columns"]:
        x_test[column] = x_test[column].astype("category")
    probabilities = model.predict_proba(x_test)[:, 1]

    assert metadata["train_rows"] == 4
    assert metadata["validation_rows"] == 2
    assert metadata["test_rows"] == 2
    assert metadata["test_year"] == 2026
    assert saved_metadata["mlflow_run_id"] == metadata["mlflow_run_id"]
    assert saved_metadata["best_params"]
    assert saved_metadata["tuning_results"]
    assert all("validation_metrics" in result for result in saved_metadata["tuning_results"])
    assert all("test_metrics" in result for result in saved_metadata["tuning_results"])
    assert all("log_loss" in result["validation_metrics"] for result in saved_metadata["tuning_results"])
    assert all("log_loss" in result["test_metrics"] for result in saved_metadata["tuning_results"])
    assert set(saved_metadata["metrics"]) == {"log_loss", "accuracy", "brier_score", "roc_auc"}
    assert preprocessor["categorical_columns"] == saved_metadata["categorical_columns"]
    assert len(probabilities) == 2

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    run = client.get_run(metadata["mlflow_run_id"])
    artifacts = {artifact.path for artifact in client.list_artifacts(metadata["mlflow_run_id"])}
    assert run.data.params["test_year"] == "2026"
    assert run.data.params["validation_rows"] == "2"
    assert "log_loss" in run.data.metrics
    assert {"model.joblib", "preprocessor.joblib", "feature_columns.json", "metadata.json"} <= artifacts
