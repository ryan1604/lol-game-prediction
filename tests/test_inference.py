from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.features.build_features import ROLE_ORDER, SIDES, build_match_features
from app.inference.predictor import Predictor
from app.inference.schemas import CompletedDraftRequest


class ConstantModel:
    def predict_proba(self, frame: pd.DataFrame) -> list[list[float]]:
        return [[0.25, 0.75] for _ in range(len(frame))]


def _match(
    *,
    match_id: int,
    match_date: str,
    year: int,
    blue_team: str = "AAA",
    red_team: str = "BBB",
    blue_win: bool = True,
    blue_top_player: str = "Alice",
    blue_top_champion: str = "Aatrox",
) -> dict[str, object]:
    row: dict[str, object] = {
        "match_id": match_id,
        "match_date": match_date,
        "region": "LCK",
        "year": year,
        "split": "Spring",
        "stage": "WEEK1",
        "blue_win": blue_win,
        "first_pick_side": "blue",
        "blue_team_name": blue_team,
        "red_team_name": red_team,
    }
    for side in SIDES:
        for role in ROLE_ORDER:
            row[f"{side}_{role}_player"] = f"{side}_{role}_{match_id}"
            row[f"{side}_{role}_champion"] = f"{role}_{match_id}"
    row["blue_top_player"] = blue_top_player
    row["blue_top_champion"] = blue_top_champion
    return row


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "region": "LCK",
        "year": 2026,
        "split": "Spring",
        "stage": "WEEK1",
        "first_pick_side": "blue",
        "blue": {
            "team_name": "AAA",
            "players": {role: f"blue_{role}_player" for role in ROLE_ORDER},
            "champions": {role: f"Blue{role.title()}" for role in ROLE_ORDER},
        },
        "red": {
            "team_name": "BBB",
            "players": {role: f"red_{role}_player" for role in ROLE_ORDER},
            "champions": {role: f"Red{role.title()}" for role in ROLE_ORDER},
        },
    }
    payload.update(overrides)
    return payload


def _write_model_dir(path: Path, feature_columns: list[str]) -> None:
    path.mkdir()
    joblib.dump(ConstantModel(), path / "model.joblib")
    joblib.dump(
        {"categorical_columns": ["region", "split", "stage", "first_pick_side", "blue_team_name", "red_team_name"]},
        path / "preprocessor.joblib",
    )
    (path / "feature_columns.json").write_text(json.dumps(feature_columns), encoding="utf-8")
    (path / "metadata.json").write_text(
        json.dumps({"mlflow_run_id": "run-123", "trained_at": "2026-07-04T12:55:28+00:00"}),
        encoding="utf-8",
    )


def _write_history(path: Path) -> tuple[pd.DataFrame, list[str]]:
    history = pd.DataFrame(
        [
            _match(match_id=1, match_date="2024-02-01", year=2024),
            _match(match_id=2, match_date="2026-01-01", year=2026),
        ]
    )
    feature_columns = [
        column
        for column in build_match_features(history).columns
        if column not in {"match_id", "match_date", "blue_win"}
    ]
    history.to_parquet(path, index=False)
    return history, feature_columns


def test_predictor_builds_model_ready_row_from_completed_draft(tmp_path: Path) -> None:
    history_path = tmp_path / "matches_clean.parquet"
    _, feature_columns = _write_history(history_path)
    model_dir = tmp_path / "model"
    _write_model_dir(model_dir, feature_columns)
    request = CompletedDraftRequest.from_mapping(
        _payload(
            blue={
                "team_name": "AAA",
                "players": {"top": "Alice", **{role: f"blue_{role}_player" for role in ROLE_ORDER if role != "top"}},
                "champions": {"top": "Aatrox", **{role: f"Blue{role.title()}" for role in ROLE_ORDER if role != "top"}},
            }
        )
    )

    row = Predictor(model_dir, history_path).build_feature_row(request)

    assert row.columns.tolist() == feature_columns
    assert str(row["region"].dtype) == "category"
    assert row.loc[0, "blue_team_split_games_before"] == 1
    assert row.loc[0, "blue_top_champion_role_games_before"] == 1
    assert row.loc[0, "blue_top_player_champion_games_last_2y"] == 2
    assert row.loc[0, "blue_top_player_champion_win_rate_last_2y"] == 1.0


def test_predictor_uses_missing_history_fallbacks(tmp_path: Path) -> None:
    history_path = tmp_path / "matches_clean.parquet"
    _, feature_columns = _write_history(history_path)
    model_dir = tmp_path / "model"
    _write_model_dir(model_dir, feature_columns)
    request = CompletedDraftRequest.from_mapping(
        _payload(
            year=2027,
            blue={
                "team_name": "NEW",
                "players": {role: f"new_{role}" for role in ROLE_ORDER},
                "champions": {role: f"New{role.title()}" for role in ROLE_ORDER},
            },
        )
    )

    row = Predictor(model_dir, history_path).build_feature_row(request)

    assert row.loc[0, "blue_team_split_games_before"] == 0
    assert row.loc[0, "blue_team_split_win_rate_before"] == 0.5
    assert row.loc[0, "blue_top_champion_role_games_before"] == 0
    assert row.loc[0, "blue_top_champion_role_win_rate_before"] == 0.5
    assert row.loc[0, "blue_top_player_champion_games_last_2y"] == 0
    assert row.loc[0, "blue_top_player_champion_win_rate_last_2y"] == 0.5


def test_predictor_returns_probability_and_model_metadata(tmp_path: Path) -> None:
    history_path = tmp_path / "matches_clean.parquet"
    _, feature_columns = _write_history(history_path)
    model_dir = tmp_path / "model"
    _write_model_dir(model_dir, feature_columns)
    request = CompletedDraftRequest.from_mapping(_payload())

    response = Predictor(model_dir, history_path).predict(request)

    assert response.blue_win_probability == 0.75
    assert response.predicted_winner == "blue"
    assert response.model_version == "run-123"
    assert response.trained_at == "2026-07-04T12:55:28+00:00"


def test_api_predicts_with_loaded_predictor_artifacts(tmp_path: Path) -> None:
    history_path = tmp_path / "matches_clean.parquet"
    _, feature_columns = _write_history(history_path)
    model_dir = tmp_path / "model"
    _write_model_dir(model_dir, feature_columns)
    had_predictor = hasattr(app.state, "predictor")
    previous_predictor = getattr(app.state, "predictor", None)
    app.state.predictor = Predictor(model_dir, history_path)

    try:
        response = TestClient(app).post("/predict", json=_payload())
    finally:
        if had_predictor:
            app.state.predictor = previous_predictor
        else:
            del app.state.predictor

    assert response.status_code == 200
    assert response.json() == {
        "blue_win_probability": 0.75,
        "predicted_winner": "blue",
        "model_version": "run-123",
        "trained_at": "2026-07-04T12:55:28+00:00",
    }


def test_completed_draft_request_rejects_invalid_payloads() -> None:
    payload = _payload(first_pick_side="middle")

    with pytest.raises(ValueError, match="first_pick_side must be blue or red"):
        CompletedDraftRequest.from_mapping(payload)

    payload = _payload(blue={"team_name": "AAA", "players": {}, "champions": {}})

    with pytest.raises(ValueError, match="blue.players.top is required"):
        CompletedDraftRequest.from_mapping(payload)
