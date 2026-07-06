from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.features.build_features import ROLE_ORDER, SIDES

SideName = Literal["blue", "red"]


def _text(value: Any, field: str) -> str:
    if value is None:
        raise ValueError(f"{field} is required")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


@dataclass(frozen=True)
class DraftSide:
    team_name: str
    players: dict[str, str]
    champions: dict[str, str]

    @classmethod
    def from_mapping(cls, value: Any, side: SideName) -> DraftSide:
        if not isinstance(value, dict):
            raise ValueError(f"{side} must be an object")
        players = value.get("players")
        champions = value.get("champions")
        if not isinstance(players, dict):
            raise ValueError(f"{side}.players must be an object")
        if not isinstance(champions, dict):
            raise ValueError(f"{side}.champions must be an object")
        return cls(
            team_name=_text(value.get("team_name"), f"{side}.team_name"),
            players={role: _text(players.get(role), f"{side}.players.{role}") for role in ROLE_ORDER},
            champions={role: _text(champions.get(role), f"{side}.champions.{role}") for role in ROLE_ORDER},
        )


@dataclass(frozen=True)
class CompletedDraftRequest:
    region: str
    year: int
    split: str
    stage: str
    first_pick_side: SideName
    blue: DraftSide
    red: DraftSide

    @classmethod
    def from_mapping(cls, value: Any) -> CompletedDraftRequest:
        if not isinstance(value, dict):
            raise ValueError("request must be an object")
        first_pick_side = _text(value.get("first_pick_side"), "first_pick_side").lower()
        if first_pick_side not in SIDES:
            raise ValueError("first_pick_side must be blue or red")
        try:
            year = int(value.get("year"))
        except (TypeError, ValueError):
            raise ValueError("year must be an integer") from None
        return cls(
            region=_text(value.get("region"), "region").upper(),
            year=year,
            split=_text(value.get("split"), "split"),
            stage=_text(value.get("stage"), "stage"),
            first_pick_side=first_pick_side,  # type: ignore[arg-type]
            blue=DraftSide.from_mapping(value.get("blue"), "blue"),
            red=DraftSide.from_mapping(value.get("red"), "red"),
        )


@dataclass(frozen=True)
class PredictionResponse:
    blue_win_probability: float
    predicted_winner: SideName
    model_version: str
    trained_at: str | None = None
