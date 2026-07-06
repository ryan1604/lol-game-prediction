from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.features.build_features import ROLE_ORDER
from app.inference.schemas import PredictionResponse


class StubPredictor:
    model_dir = Path("stub-model")
    metadata = {"mlflow_run_id": "run-123"}
    history = pd.DataFrame(
        [
            {
                "region": "LCK",
                "year": 2026,
                "split": "Spring",
                "stage": "WEEK1",
                "blue_team_name": "AAA",
                "red_team_name": "BBB",
                "blue_top_player": "Alice",
                "red_top_player": "Bob",
                "blue_top_champion": "Aatrox",
                "red_top_champion": "Gnar",
            }
        ]
    )

    def predict(self, request: object) -> PredictionResponse:
        return PredictionResponse(
            blue_win_probability=0.75,
            predicted_winner="blue",
            model_version="run-123",
            trained_at="2026-07-04T12:55:28+00:00",
        )


@pytest.fixture(autouse=True)
def stub_predictor() -> None:
    app.state.predictor = StubPredictor()
    yield
    del app.state.predictor


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


def test_health_returns_status() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metadata_returns_selector_values_and_model_version() -> None:
    response = TestClient(app).get("/metadata")

    assert response.status_code == 200
    assert response.json() == {
        "regions": ["LCK"],
        "years": [2026],
        "splits": ["Spring"],
        "stages": ["WEEK1"],
        "contexts": [{"region": "LCK", "year": 2026, "split": "Spring", "stage": "WEEK1"}],
        "teams": ["AAA", "BBB"],
        "players": ["Alice", "Bob"],
        "champions": ["Aatrox", "Gnar"],
        "model_version": "run-123",
    }


def test_predict_returns_prediction_schema() -> None:
    response = TestClient(app).post("/predict", json=_payload())

    assert response.status_code == 200
    assert response.json() == {
        "blue_win_probability": 0.75,
        "predicted_winner": "blue",
        "model_version": "run-123",
        "trained_at": "2026-07-04T12:55:28+00:00",
    }


def test_predict_openapi_uses_concrete_schemas() -> None:
    path = TestClient(app).get("/openapi.json").json()["paths"]["/predict"]["post"]
    schemas = TestClient(app).get("/openapi.json").json()["components"]["schemas"]

    assert path["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/CompletedDraftPayload"
    )
    assert path["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/PredictionPayload"
    )
    assert set(schemas["RoleSelections"]["properties"]) == {"top", "jungle", "mid", "bot", "support"}
    assert set(schemas["PredictionPayload"]["properties"]) == {
        "blue_win_probability",
        "predicted_winner",
        "model_version",
        "trained_at",
    }


def test_predict_rejects_invalid_payload() -> None:
    response = TestClient(app).post("/predict", json=_payload(first_pick_side="middle"))

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "first_pick_side"]
