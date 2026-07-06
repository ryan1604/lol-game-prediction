from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.features.build_features import DEFAULT_INPUT_PATH, HISTORY_PRIOR, ROLE_ORDER, SIDES, build_match_features
from app.training.train import DEFAULT_OUTPUT_DIR
from app.inference.schemas import CompletedDraftRequest, PredictionResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_PATH = DEFAULT_INPUT_PATH


class Predictor:
    def __init__(
        self,
        model_dir: Path = DEFAULT_OUTPUT_DIR,
        history_path: Path = DEFAULT_HISTORY_PATH,
    ) -> None:
        self.model_dir = model_dir
        self.model = joblib.load(model_dir / "model.joblib")
        self.preprocessor = joblib.load(model_dir / "preprocessor.joblib")
        self.feature_columns = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))
        self.metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.history = pd.read_parquet(history_path)

    def build_feature_row(self, request: CompletedDraftRequest) -> pd.DataFrame:
        latest_date = pd.to_datetime(self.history["match_date"]).max().date() if not self.history.empty else None
        draft_date = latest_date + timedelta(days=1) if latest_date else pd.Timestamp.today().date()
        draft = self._draft_row(request, draft_date.isoformat())
        rows = pd.concat([self.history, pd.DataFrame([draft])], ignore_index=True)
        features = build_match_features(rows)
        feature_row = features[features["match_id"] == draft["match_id"]].tail(1)
        if feature_row.empty:
            raise RuntimeError("Could not build inference feature row")
        prepared = feature_row.loc[:, self.feature_columns].reset_index(drop=True)
        for column in self.preprocessor.get("categorical_columns", []):
            if column in prepared:
                prepared[column] = prepared[column].astype("category")
        return prepared

    def predict(self, request: CompletedDraftRequest) -> PredictionResponse:
        row = self.build_feature_row(request)
        probability = float(self.model.predict_proba(row)[0][1])
        return PredictionResponse(
            blue_win_probability=probability,
            predicted_winner="blue" if probability >= 0.5 else "red",
            model_version=str(self.metadata.get("mlflow_run_id") or self.metadata.get("trained_at") or self.model_dir.name),
            trained_at=self.metadata.get("trained_at"),
        )

    def _draft_row(self, request: CompletedDraftRequest, match_date: str) -> dict[str, Any]:
        next_id = int(self.history["match_id"].max()) + 1 if not self.history.empty else 1
        row: dict[str, Any] = {
            "match_id": next_id,
            "match_date": match_date,
            "region": request.region,
            "year": request.year,
            "split": request.split,
            "stage": request.stage,
            "blue_win": False,
            "first_pick_side": request.first_pick_side,
            "blue_team_name": request.blue.team_name,
            "red_team_name": request.red.team_name,
        }
        for side in SIDES:
            draft_side = getattr(request, side)
            for role in ROLE_ORDER:
                row[f"{side}_{role}_player"] = draft_side.players[role]
                row[f"{side}_{role}_champion"] = draft_side.champions[role]
        return row


def history_fallbacks() -> dict[str, float]:
    return {"games": 0, "win_rate": HISTORY_PRIOR}

