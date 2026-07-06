from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.features.build_features import ROLE_ORDER, SIDES
from app.inference.predictor import Predictor
from app.inference.schemas import CompletedDraftRequest
from app.inference.schemas import SideName

app = FastAPI(title="LoL Game Prediction")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lol-game-prediction.pages.dev"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class RoleSelections(BaseModel):
    top: str
    jungle: str
    mid: str
    bot: str
    support: str


class DraftSidePayload(BaseModel):
    team_name: str
    players: RoleSelections
    champions: RoleSelections


class CompletedDraftPayload(BaseModel):
    region: str
    year: int
    split: str
    stage: str
    first_pick_side: SideName
    blue: DraftSidePayload
    red: DraftSidePayload


class PredictionPayload(BaseModel):
    blue_win_probability: float
    predicted_winner: SideName
    model_version: str
    trained_at: str | None = None


def get_predictor() -> Predictor:
    if not hasattr(app.state, "predictor"):
        app.state.predictor = Predictor()
    return app.state.predictor


def _values(frame: Any, columns: list[str]) -> list[str]:
    values = set()
    for column in columns:
        if column in frame:
            values.update(str(value) for value in frame[column].dropna().unique())
    return sorted(values)


def _numbers(frame: Any, column: str) -> list[int]:
    if column not in frame:
        return []
    return sorted({int(value) for value in frame[column].dropna().unique()}, reverse=True)


def _contexts(frame: Any) -> list[dict[str, Any]]:
    columns = ["region", "year", "split", "stage"]
    if any(column not in frame for column in columns):
        return []
    rows = frame.loc[:, columns].dropna().drop_duplicates().sort_values(columns)
    return [
        {
            "region": str(row.region),
            "year": int(row.year),
            "split": str(row.split),
            "stage": str(row.stage),
        }
        for row in rows.itertuples(index=False)
    ]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    predictor = get_predictor()
    return {
        "regions": _values(predictor.history, ["region"]),
        "years": _numbers(predictor.history, "year"),
        "splits": _values(predictor.history, ["split"]),
        "stages": _values(predictor.history, ["stage"]),
        "contexts": _contexts(predictor.history),
        "teams": _values(predictor.history, [f"{side}_team_name" for side in SIDES]),
        "players": _values(predictor.history, [f"{side}_{role}_player" for side in SIDES for role in ROLE_ORDER]),
        "champions": _values(predictor.history, [f"{side}_{role}_champion" for side in SIDES for role in ROLE_ORDER]),
        "model_version": str(
            predictor.metadata.get("mlflow_run_id")
            or predictor.metadata.get("trained_at")
            or predictor.model_dir.name
        ),
    }


@app.post("/predict", response_model=PredictionPayload)
def predict(payload: CompletedDraftPayload) -> PredictionPayload:
    try:
        request = CompletedDraftRequest.from_mapping(payload.model_dump())
        return PredictionPayload(**asdict(get_predictor().predict(request)))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    uvicorn.run("app.api.main:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
